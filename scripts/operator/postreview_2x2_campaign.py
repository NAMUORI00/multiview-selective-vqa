"""Prepare and execute the post-review 2x2 semantic-QA campaign.

The preparation path is metadata-only: it reads AI-Hub label JSON and the
historical exposure audit, but never opens video, loads a model, or reads a
model response.  Test media materialization requires an explicit sealed-open
flag.  All targets are derived from frozen annotations and predeclared
geometry rules; this campaign does not claim human semantic sufficiency.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs" / "postreview_acceptance_20260814"
EXPOSURE_AUDIT = (
    ROOT
    / "outputs"
    / "dissection_confirmatory_phase4_20260710"
    / "exclusion_ids.jsonl"
)
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
VLLM_IMAGE = "vllm/vllm-openai:latest"
ABSTAIN_TEXT = "판단할 시각적 근거가 충분하지 않습니다."
SEED = 20260814
PASSES = 3
GRADIENT_ACCUMULATION = 8
SEQUENCE_SLOTS = 1536
BOOTSTRAP_REPLICATES = 4000
SUPPORTED_REASONS = {"active_dev", "reserve_locked_501"}
CONDITIONS = ("available", "temporal_miss", "wrong_region")
PRIMARY_CONDITIONS = ("available", "wrong_region")
ARMS = ("paired_sft", "answer_only_sft", "explicit_prompt", "legacy_lora")


SUPPORTED_EVENT_CLASSES = {
    "경계선을 통한 침입",
    "비정상적인 경로로의 침범",
    "신체적 충돌을 동반한 싸움",
    "오토바이 인도 주행",
    "자전거 인도 주행",
    "전동킥보드 인도 주행",
    "이륜 이동수단 운전자 1인 헬멧 미착용",
    "이륜 이동수단 탑승자 일부 또는 전체 헬멧 미착용",
    "전동킥보드 앞뒤 2인 탑승",
}

PROPERTY_QUESTIONS = {
    "moves_right": "첫 프레임과 비교해 마지막 프레임에서 추적 대상의 중심이 오른쪽으로 이동했는가?",
    "moves_down": "첫 프레임과 비교해 마지막 프레임에서 추적 대상의 중심이 아래쪽으로 이동했는가?",
    "grows_larger": "첫 프레임과 비교해 마지막 프레임에서 추적 대상 영역이 뚜렷하게 커졌는가?",
    "multiple_people": "세 프레임 중 하나 이상에서 두 명 이상의 추적 인물이 함께 표시되는가?",
}

PROPERTY_RULES = {
    "moves_right": "yes if normalized center dx >= 0.05; no if <= -0.05; otherwise excluded",
    "moves_down": "yes if normalized center dy >= 0.05; no if <= -0.05; otherwise excluded",
    "grows_larger": "yes if final/initial bbox area >= 1.25; no if <= 0.80; otherwise excluded",
    "multiple_people": "yes if any evidence obj_id contains at least two IDs; no if all contain exactly one",
}


SYSTEM_PROMPT = (
    "너는 CCTV 영상 프레임 묶음에 실제로 보이는 사건만 판정한다. "
    "질문의 답을 시각적으로 결정할 수 있으면 yes 또는 no를 선택하고, "
    "객체, 관계, 경계, 착용 상태 또는 결정 순간이 보이지 않거나 모호하면 "
    f"abstain을 선택한다. 출력은 정확히 "
    '{"decision":"answer|abstain","answer":"yes|no|abstain"} 형식의 JSON 한 줄이어야 한다.'
)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")
    return sha256_bytes(payload.encode("utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_bytes(row) for row in rows)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def normalize_repo_path(value: str) -> Path:
    path = ROOT / value.replace("\\", "/")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def parse_frame_range(text: str) -> tuple[int, int]:
    match = re.search(r"프레임\s*범위\s*(\d+)\s*[~\-–]\s*(\d+)", text)
    if not match:
        raise ValueError(f"missing evidence frame range: {text}")
    start, end = int(match.group(1)), int(match.group(2))
    if start > end:
        raise ValueError(f"reversed evidence frame range: {text}")
    return start, end


def bbox_area(box: list[Any]) -> float:
    if len(box) < 4:
        return 0.0
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def selected_view(annotation: dict[str, Any]) -> str:
    evidence = annotation["annotations"]["evidence"]
    scored = []
    for view in ("c1", "c2"):
        payload = evidence.get(view)
        if not isinstance(payload, dict):
            continue
        frames = payload.get("frame_id") or []
        boxes = payload.get("obj_bbox") or []
        if (
            len(frames) != 3
            or len(boxes) != 3
            or any(bbox_area(box) <= 0 for box in boxes)
        ):
            continue
        scored.append((sum(bbox_area(box) for box in boxes), view))
    if not scored:
        raise ValueError("no view has exactly three evidence frames and boxes")
    return max(scored)[1]


def tracked_id_count(value: Any) -> int:
    identifiers = {
        token.strip()
        for token in re.split(r"[,;\s]+", str(value or ""))
        if token.strip()
    }
    return len(identifiers)


def property_labels(
    boxes: list[list[int]],
    obj_ids: list[Any],
    width: int,
    height: int,
) -> dict[str, str | None]:
    centers = [((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0) for box in boxes]
    areas = [bbox_area(box) for box in boxes]
    dx = (centers[-1][0] - centers[0][0]) / width
    dy = (centers[-1][1] - centers[0][1]) / height
    ratio = areas[-1] / areas[0]
    counts = [tracked_id_count(value) for value in obj_ids]
    if any(count <= 0 for count in counts):
        raise ValueError("empty tracked obj_id")
    return {
        "moves_right": "yes" if dx >= 0.05 else "no" if dx <= -0.05 else None,
        "moves_down": "yes" if dy >= 0.05 else "no" if dy <= -0.05 else None,
        "grows_larger": "yes" if ratio >= 1.25 else "no" if ratio <= 0.80 else None,
        "multiple_people": "yes" if max(counts) >= 2 else "no",
    }


def source_video_path(label_path: Path, filename: str) -> Path:
    parts = list(label_path.parts)
    try:
        index = parts.index("02.라벨링데이터")
    except ValueError as exc:
        raise ValueError(f"unexpected label path: {label_path}") from exc
    parts[index] = "01.원천데이터"
    candidate = Path(*parts[:-1]) / filename
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def exposure_rows(reason: str) -> list[dict[str, Any]]:
    if reason not in SUPPORTED_REASONS:
        raise ValueError(reason)
    rows = [row for row in read_jsonl(EXPOSURE_AUDIT) if reason in row.get("reasons", [])]
    ids = [str(row["event_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate event ids for {reason}")
    expected = 600 if reason == "active_dev" else 501
    if len(rows) != expected:
        raise RuntimeError(f"{reason} count drift: {len(rows)} != {expected}")
    return rows


def source_record(index: dict[str, Any], partition: str) -> dict[str, Any] | None:
    event_class = str(index["event_class"])
    if event_class not in SUPPORTED_EVENT_CLASSES:
        return None
    label_path = normalize_repo_path(str(index["label_path"]))
    annotation = json.loads(label_path.read_text(encoding="utf-8-sig"))
    if annotation.get("annotations", {}).get("event_class") != event_class:
        raise RuntimeError(f"event-class drift for {index['event_id']}")
    view = selected_view(annotation)
    videos = {str(row["view"]): row for row in annotation["videos"]}
    if set(videos) != {"c1", "c2"}:
        raise RuntimeError(f"camera-view drift for {index['event_id']}")
    evidence = annotation["annotations"]["evidence"][view]
    video = source_video_path(label_path, str(videos[view]["filename"]))
    start, end = parse_frame_range(str(evidence["evidence_text"]))
    frames = [int(value) for value in evidence["frame_id"]]
    boxes = [[int(round(float(value))) for value in box[:4]] for box in evidence["obj_bbox"]]
    if len(frames) != 3 or len(boxes) != 3 or any(bbox_area(box) <= 0 for box in boxes):
        raise RuntimeError(f"invalid evidence triplet for {index['event_id']}/{view}")
    properties = property_labels(
        boxes,
        list(evidence["obj_id"]),
        int(videos[view]["width"]),
        int(videos[view]["height"]),
    )
    width, height = int(videos[view]["width"]), int(videos[view]["height"])
    union = [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]
    available_crop = clamp_box(union, width, height)
    shifted_crop = wrong_region_box(available_crop, union, width, height)
    wrong_region_max_iou = max(iou(shifted_crop, box) for box in boxes)
    return {
        "schema_version": "postreview_2x2_source_v1",
        "partition": partition,
        "event_id": str(index["event_id"]),
        "event_class": event_class,
        "top_class": str(index["top_class"]),
        "dataset_split": str(index["dataset_split"]),
        "label_relpath": label_path.relative_to(ROOT).as_posix(),
        "label_sha256": sha256_file(label_path),
        "view": view,
        "video_relpath": video.relative_to(ROOT).as_posix(),
        "video_bytes": video.stat().st_size,
        "video_content_sha256": "deferred_until_materialization",
        "duration_seconds": float(videos[view]["length"]),
        "evidence_frame_range": [start, end],
        "evidence_frame_ids": frames,
        "evidence_boxes_xyxy": boxes,
        "frame_size": [int(videos[view]["width"]), int(videos[view]["height"])],
        "property_questions": PROPERTY_QUESTIONS,
        "property_labels": properties,
        "wrong_region_max_iou": wrong_region_max_iou,
        "wrong_region_eligible": wrong_region_max_iou <= 0.05,
    }


def stratified_dev(events: list[dict[str, Any]], fraction: float = 0.2) -> set[str]:
    by_class: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in events:
        by_class[str(row["event_class"])].append(row)
    selected: set[str] = set()
    for event_class, rows in sorted(by_class.items()):
        ordered = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{SEED}|dev|{event_class}|{row['event_id']}".encode("utf-8")
            ).hexdigest(),
        )
        count = max(1, int(round(len(ordered) * fraction)))
        if count >= len(ordered):
            count = max(1, len(ordered) - 1)
        selected.update(str(row["event_id"]) for row in ordered[:count])
    return selected


def prepare(args: argparse.Namespace) -> int:
    imported_before = set(sys.modules)
    active = [source_record(row, "development_pool") for row in exposure_rows("active_dev")]
    test = [source_record(row, "prospective_test") for row in exposure_rows("reserve_locked_501")]
    active = [row for row in active if row is not None]
    test = [row for row in test if row is not None]
    if len(active) != 599 or len(test) != 500:
        raise RuntimeError(f"supported-source count drift: active={len(active)} test={len(test)}")
    dev_ids = stratified_dev(active)
    for row in active:
        row["partition"] = "dev" if row["event_id"] in dev_ids else "train"
    sources = sorted(active + test, key=lambda row: (row["partition"], row["event_id"]))
    source_sha = write_jsonl(args.output_root / "manifests" / "source_candidates.jsonl", sources)
    counts = collections.Counter(str(row["partition"]) for row in sources)
    class_counts = {
        partition: dict(
            sorted(collections.Counter(row["event_class"] for row in sources if row["partition"] == partition).items())
        )
        for partition in ("train", "dev", "prospective_test")
    }
    protocol = {
        "schema_version": "postreview_2x2_protocol_draft_v1",
        "status": "DRAFT_NOT_FROZEN_NOT_EXECUTED",
        "created_date": "2026-08-14",
        "research_change": "prospective post-review experiment; does not replace or relabel the original sealed B1 result",
        "primary_design": {
            "unit": "event",
            "cells_per_event": {
                "available_yes": "answer yes",
                "available_no": "answer no",
                "unavailable_yes": "abstain",
                "unavailable_no": "abstain",
            },
            "same_media_for_available_yes_and_no": True,
            "same_media_for_unavailable_yes_and_no": True,
            "annotation_derived_property_questions": PROPERTY_QUESTIONS,
            "property_rules": PROPERTY_RULES,
            "available_yes_no_are_derived_from_frozen_geometry_and_tracking_annotations": True,
            "primary_unavailable_operator": "wrong_region with maximum IoU <= 0.05 against every annotated evidence box",
            "temporal_miss_role": "unscored oracle-dependence diagnostic because actor visibility outside the annotated interval is not guaranteed absent",
            "strict_answer_enum": ["yes", "no", "abstain"],
            "substring_or_affirmative_token_fallback": False,
        },
        "label_contract": {
            "human_judgment_used": False,
            "yes_no_source": "frozen AI-Hub frame IDs, bounding boxes, and tracked object IDs",
            "unavailable_source": "same-event wrong-region geometry rule only",
            "semantic_sufficiency_claimed": False,
        },
        "model_arms": {
            "primary": ["paired_sft", "answer_only_sft", "explicit_prompt"],
            "diagnostic": ["legacy_lora"],
            "base_model": MODEL_ID,
            "paired_sft": "rank-16 LoRA trained only on train; epoch chosen by dev loss before test",
            "answer_only_sft": "same LoRA and update-token budget; available Yes/No rows repeated twice; no abstention target",
            "same_output_schema_and_prompt_contract": True,
        },
        "metrics": {
            "primary": ["four-cell macro exact accuracy", "all-four-rows-correct event rate"],
            "components": ["available yes accuracy", "available no accuracy", "unavailable abstention for each polarity"],
            "shortcut_gates": ["available question-only majority accuracy <= 0.60", "each used property question has both yes and no targets"],
            "statistics": ["paired event-cluster bootstrap 4000 replicates", "exact McNemar", "exact binomial intervals"],
            "release_secondary_only": ["pre-gate exact accuracy", "common-coverage risk curves", "no arm-specific primary threshold"],
        },
        "claim_boundary": {
            "availability_is_semantic_sufficiency": False,
            "human_semantic_sufficiency": "not_claimed",
            "oracle_localized_triplets_primary": True,
            "roi_perturbation_is_rules_derived_unavailability": True,
            "temporal_perturbation_is_unscored_diagnostic": True,
            "perturbations_are_not_end_to_end_localization": True,
            "online_latency_or_deployment_safety_claimed": False,
        },
        "counts": dict(sorted(counts.items())),
        "class_counts": class_counts,
        "seed": SEED,
        "passes": PASSES,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "sequence_slots": SEQUENCE_SLOTS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "source_manifest": {"path": "manifests/source_candidates.jsonl", "sha256": source_sha},
        "script_sha256": sha256_file(Path(__file__)),
        "gpu_execution_started": False,
        "model_loaded": False,
        "sealed_test_video_opened": False,
    }
    protocol_sha = write_json(args.output_root / "protocol_draft.json", protocol)
    command_sequence = {
        "schema_version": "postreview_2x2_commands_v1",
        "working_directory_local": str(ROOT),
        "working_directory_gpu_host": "/workspace/MV-EviRAG",
        "commands": [
            "python scripts/operator/postreview_2x2_campaign.py prepare",
            "python scripts/operator/postreview_2x2_campaign.py materialize --partition train",
            "python scripts/operator/postreview_2x2_campaign.py materialize --partition dev",
            "python scripts/operator/postreview_2x2_campaign.py nuisance-audit --partitions train dev",
            "git add -f scripts/operator/postreview_2x2_campaign.py outputs/postreview_acceptance_20260814/protocol_draft.json outputs/postreview_acceptance_20260814/commands.json outputs/postreview_acceptance_20260814/reproducibility_contract.json",
            "git commit -m 'eval: freeze post-review structured 2x2 campaign'",
            "git push origin main",
            "# sync gpu-host to the frozen commit and verify the checkout is clean before any prospective-test access",
            "# run the following prospective-test materialization locally because licensed source videos remain local",
            "python scripts/operator/postreview_2x2_campaign.py materialize --partition prospective_test --allow-sealed-open",
            "python scripts/operator/postreview_2x2_campaign.py build-manifest",
            "python scripts/operator/postreview_2x2_campaign.py nuisance-audit --partitions prospective_test",
            "python scripts/operator/postreview_2x2_campaign.py transfer-manifest",
            "# transfer only transfer_files.txt entries to /workspace/MV-EviRAG and run verify-transfer there",
            "CUDA_VISIBLE_DEVICES=0 python scripts/operator/postreview_2x2_campaign.py train --arm paired_sft",
            "CUDA_VISIBLE_DEVICES=1 python scripts/operator/postreview_2x2_campaign.py train --arm answer_only_sft",
            "# run infer for every arm and both scopes according to launch_plan.json",
            "python scripts/operator/postreview_2x2_campaign.py aggregate",
        ],
        "note": "commands containing commit, push, sealed-open, train, infer, or aggregate are not executed during preparation",
    }
    commands_sha = write_json(args.output_root / "commands.json", command_sequence)
    reproducibility = {
        "schema_version": "postreview_2x2_reproducibility_contract_v1",
        "status": "DRAFT_BEFORE_EXECUTION",
        "protocol_sha256": protocol_sha,
        "commands_sha256": commands_sha,
        "required_public_or_supplementary_artifacts": [
            "exact execution script and frozen protocol",
            "train/dev/test event IDs when license permits, otherwise ordered salted hashes plus count and class distribution",
            "source-label, media, and final-manifest SHA-256 inventories",
            "base-model revision, processor revision, container image digest, CUDA and package versions",
            "random seed, LoRA configuration, update count, and every checkpoint-selection record",
            "strict prompts and JSON schema",
            "raw model responses, token-probability confidence, parse status, and exact adjudication result",
            "nuisance audit, question-only audit, bootstrap draws or reproducible seed, McNemar inputs, and aggregate script",
            "server commit-sync preflight, GPU resource record, transfer receipt, and sealed-media cleanup receipt",
            "LoRA adapters when redistribution terms permit",
        ],
        "not_redistributed": ["AI-Hub source videos", "private credentials", "unlicensed raw frames"],
        "no_primary_release_threshold": True,
        "raw_outputs_retained_before_aggregation": True,
        "human_judgment_used": False,
        "model_outputs_exist": False,
    }
    write_json(args.output_root / "reproducibility_contract.json", reproducibility)
    launch_plan = {
        "schema_version": "postreview_2x2_launch_plan_v1",
        "status": "BLOCKED_UNTIL_AUTOMATIC_MANIFEST_AND_GIT_SYNC",
        "target": {"alias": "gpu-host", "repo": "/workspace/MV-EviRAG", "gpus": [0, 1, 2]},
        "waves": [
            {"wave": 0, "action": "materialize train/dev locally and pass automatic manifest audits"},
            {"wave": 1, "gpu0": "paired_sft train", "gpu1": "answer_only_sft train", "gpu2": "idle"},
            {"wave": 2, "gpu0": "paired_sft inference", "gpu1": "answer_only_sft inference", "gpu2": "explicit_prompt inference"},
            {"wave": 3, "gpu0": "legacy_lora inference", "gpu1": "idle", "gpu2": "idle"},
            {"wave": 4, "gpu0": "paired_sft temporal diagnostic", "gpu1": "answer_only_sft temporal diagnostic", "gpu2": "explicit_prompt temporal diagnostic"},
            {"wave": 5, "gpu0": "legacy_lora temporal diagnostic", "gpu1": "idle", "gpu2": "idle"},
            {"wave": 6, "action": "recover raw outputs, aggregate on CPU, remove prospective-test media from server"},
        ],
        "required_before_wave_1": [
            "automatic manifest and nuisance-control audits pass before model outputs",
            "protocol and exact code committed and pushed to origin/main",
            "gpu-host checkout equals frozen commit and is clean",
            "manifest hashes match and only consumed media are transferred",
            "free disk remains above 80 GiB safety floor",
        ],
        "protocol_draft_sha256": protocol_sha,
    }
    write_json(args.output_root / "launch_plan.json", launch_plan)
    runtime_modules = sorted(
        name
        for name in set(sys.modules) - imported_before
        if name.split(".")[0] in {"torch", "transformers", "vllm", "peft"}
    )
    if runtime_modules:
        raise RuntimeError(f"prepare loaded model runtimes: {runtime_modules}")
    report = {
        "schema_version": "postreview_2x2_prepare_report_v1",
        "status": "PASS_METADATA_ONLY",
        "source_rows": len(sources),
        "automatic_source_rows": len(sources),
        "automatic_2x2_eligible_by_partition": {
            partition: sum(
                row["partition"] == partition
                and row["wrong_region_eligible"]
                and "yes" in row["property_labels"].values()
                and "no" in row["property_labels"].values()
                for row in sources
            )
            for partition in ("train", "dev", "prospective_test")
        },
        "partitions": dict(sorted(counts.items())),
        "unsupported_events_excluded": {"active_dev": 1, "prospective_test": 1, "event_class": "특정 인물을 뒤따라가며 배회"},
        "model_runtime_modules_loaded": runtime_modules,
        "video_open_count": 0,
        "model_output_read_count": 0,
        "gpu_execution_started": False,
    }
    write_json(args.output_root / "prepare_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def clamp_box(box: list[int], width: int, height: int, margin: float = 0.35) -> list[int]:
    x1, y1, x2, y2 = box
    pad_x = int(round((x2 - x1) * margin))
    pad_y = int(round((y2 - y1) * margin))
    return [max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)]


def iou(left: list[int], right: list[int]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / (left_area + right_area - intersection)


def wrong_region_box(crop: list[int], core: list[int], width: int, height: int) -> list[int]:
    w, h = crop[2] - crop[0], crop[3] - crop[1]
    candidates = [
        [0, 0, w, h],
        [width - w, 0, width, h],
        [0, height - h, w, height],
        [width - w, height - h, width, height],
    ]
    return min(candidates, key=lambda candidate: (iou(candidate, core), candidate))


def temporal_frames(start: int, end: int, total: int, event_id: str) -> list[int]:
    guard = 30
    segments = []
    if start - guard >= 2:
        segments.append((0, start - guard))
    if end + guard <= total - 3:
        segments.append((end + guard, total - 1))
    if not segments:
        raise RuntimeError(f"no temporal-miss segment for {event_id}: event={start}-{end}, total={total}")
    begin, finish = max(segments, key=lambda pair: (pair[1] - pair[0], pair))
    span = finish - begin
    return [begin + span // 4, begin + span // 2, begin + (3 * span) // 4]


def panel(images: list[Any], destination: Path) -> None:
    import cv2
    import numpy as np

    normalized = []
    for image in images:
        if image is None or image.size == 0:
            raise RuntimeError("empty panel image")
        canvas = np.zeros((384, 384, 3), dtype=np.uint8)
        height, width = image.shape[:2]
        scale = min(384 / max(width, 1), 384 / max(height, 1))
        resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
        y = (384 - resized.shape[0]) // 2
        x = (384 - resized.shape[1]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        normalized.append(canvas)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), cv2.hconcat(normalized), [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError(f"failed to write {destination}")


def materialize(args: argparse.Namespace) -> int:
    if args.partition == "prospective_test" and not args.allow_sealed_open:
        raise RuntimeError("prospective-test materialization requires --allow-sealed-open")
    import cv2

    sources = [
        row
        for row in read_jsonl(args.output_root / "manifests" / "source_candidates.jsonl")
        if row["partition"] == args.partition
    ]
    if args.limit:
        sources = sources[: args.limit]
    media_records = []
    skipped = []
    for index, row in enumerate(sources, 1):
        if not row["wrong_region_eligible"]:
            skipped.append({"event_id": row["event_id"], "reason": "wrong_region_max_iou_gt_0.05", "max_iou": row["wrong_region_max_iou"]})
            continue
        if not (
            "yes" in row["property_labels"].values()
            and "no" in row["property_labels"].values()
        ):
            skipped.append(
                {
                    "event_id": row["event_id"],
                    "reason": "no_unambiguous_annotation_derived_yes_no_pair",
                }
            )
            continue
        video = normalize_repo_path(str(row["video_relpath"]))
        capture = cv2.VideoCapture(str(video))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"cannot open {video}")
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if total <= 0 or width <= 0 or height <= 0:
                raise RuntimeError(f"invalid video metadata: {video}")
            event_frames = [int(value) for value in row["evidence_frame_ids"]]
            if any(frame_id < 0 or frame_id >= total for frame_id in event_frames):
                skipped.append(
                    {
                        "event_id": row["event_id"],
                        "reason": "annotation_frame_outside_video",
                        "evidence_frame_ids": event_frames,
                        "video_frame_count": total,
                    }
                )
                continue
            try:
                miss_frames = temporal_frames(*row["evidence_frame_range"], total, str(row["event_id"]))
            except RuntimeError:
                skipped.append(
                    {
                        "event_id": row["event_id"],
                        "reason": "no_temporal_offset_with_30_frame_guard",
                        "evidence_frame_range": row["evidence_frame_range"],
                        "video_frame_count": total,
                    }
                )
                continue
            frame_sets = {"available": event_frames, "temporal_miss": miss_frames, "wrong_region": event_frames}
            boxes = [[int(value) for value in box] for box in row["evidence_boxes_xyxy"]]
            union = [
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            ]
            available_crop = clamp_box(union, width, height)
            shifted_crop = wrong_region_box(available_crop, union, width, height)
            if max(iou(shifted_crop, box) for box in boxes) > 0.05:
                raise RuntimeError(f"wrong-region IoU guard failed: {row['event_id']}")
            for condition, frame_ids in frame_sets.items():
                crops = []
                for position, frame_id in enumerate(frame_ids):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ok, frame = capture.read()
                    if not ok or frame is None or frame.size == 0:
                        raise RuntimeError(f"cannot read {video}:{frame_id}")
                    crop = shifted_crop if condition == "wrong_region" else available_crop
                    x1, y1, x2, y2 = crop
                    image = frame[y1:y2, x1:x2].copy()
                    if image.size == 0:
                        raise RuntimeError(f"empty crop: {row['event_id']}/{condition}/{frame_id}")
                    crops.append(image)
                relative = Path("media") / args.partition / str(row["event_id"]) / f"{condition}_crop_triplet.jpg"
                destination = args.output_root / relative
                panel(crops, destination)
                media_records.append(
                    {
                        "condition_id": f"{row['event_id']}::{condition}",
                        "event_id": row["event_id"],
                        "partition": args.partition,
                        "condition": condition,
                        "frame_ids": frame_ids,
                        "media_relpath": relative.as_posix(),
                        "media_sha256": sha256_file(destination),
                        "media_bytes": destination.stat().st_size,
                        "source_video_sha256": sha256_file(video),
                    }
                )
        finally:
            capture.release()
        if index == 1 or index % 25 == 0 or index == len(sources):
            print(f"materialized {index}/{len(sources)} {args.partition}", flush=True)
    write_jsonl(args.output_root / "manifests" / f"media_{args.partition}.jsonl", media_records)
    write_jsonl(
        args.output_root / "manifests" / f"media_{args.partition}_exclusions.jsonl",
        skipped or [{"status": "none"}],
    )
    print(json.dumps({"status": "PASS", "partition": args.partition, "source_events": len(sources), "materialized_events": len(media_records) // 3, "excluded_events": len(skipped), "media": len(media_records)}))
    return 0


def target_payload(target: str) -> str:
    if target == "abstain":
        return json.dumps({"decision": "abstain", "answer": "abstain"}, separators=(",", ":"))
    return json.dumps({"decision": "answer", "answer": target}, separators=(",", ":"))


def build_manifest(args: argparse.Namespace) -> int:
    sources = read_jsonl(args.output_root / "manifests" / "source_candidates.jsonl")
    media = {}
    for path in args.output_root.glob("manifests/media_*.jsonl"):
        if path.stem.endswith("_exclusions"):
            continue
        media.update({row["condition_id"]: row for row in read_jsonl(path)})
    rows = []
    temporal_diagnostics = []
    exclusions = []
    balance: dict[str, dict[str, collections.Counter[str]]] = {
        partition: {
            question_id: collections.Counter() for question_id in PROPERTY_QUESTIONS
        }
        for partition in ("train", "dev", "prospective_test")
    }
    source_order = sorted(
        sources,
        key=lambda source: hashlib.sha256(
            f"{SEED}|manifest|{source['partition']}|{source['event_id']}".encode()
        ).hexdigest(),
    )
    for source in source_order:
        event_id = str(source["event_id"])
        if any(f"{event_id}::{condition}" not in media for condition in CONDITIONS):
            exclusions.append({"event_id": event_id, "reason": "missing_materialized_condition"})
            continue
        available_values = source["property_labels"]
        yes_questions = [question_id for question_id, value in available_values.items() if value == "yes"]
        no_questions = [question_id for question_id, value in available_values.items() if value == "no"]
        choices = []
        for yes_question in yes_questions:
            for no_question in no_questions:
                if yes_question == no_question:
                    continue
                simulated = {
                    question_id: collections.Counter(values)
                    for question_id, values in balance[str(source["partition"])].items()
                }
                simulated[yes_question]["yes"] += 1
                simulated[no_question]["no"] += 1
                imbalance = sum(
                    abs(values["yes"] - values["no"])
                    for values in simulated.values()
                )
                tie = hashlib.sha256(
                    f"{SEED}|pair|{event_id}|{yes_question}|{no_question}".encode()
                ).hexdigest()
                choices.append((imbalance, tie, yes_question, no_question))
        if not choices:
            exclusions.append({"event_id": event_id, "reason": "no_unambiguous_annotation_derived_yes_no_pair"})
            continue
        _, _, yes_question, no_question = min(choices)
        partition_balance = balance[str(source["partition"])]
        partition_balance[yes_question]["yes"] += 1
        partition_balance[no_question]["no"] += 1
        for condition in PRIMARY_CONDITIONS:
            evidence_state = "available" if condition == "available" else "unavailable"
            media_row = media[f"{event_id}::{condition}"]
            for polarity, question_id in (("yes", yes_question), ("no", no_question)):
                target = polarity if evidence_state == "available" else "abstain"
                row_id = f"{event_id}::{condition}::{polarity}"
                rows.append(
                    {
                        "schema_version": "postreview_2x2_eval_row_v1",
                        "row_id": row_id,
                        "event_id": event_id,
                        "group_id": event_id,
                        "partition": source["partition"],
                        "event_class": source["event_class"],
                        "evidence_state": evidence_state,
                        "question_polarity": polarity,
                        "question_id": question_id,
                        "question": source["property_questions"][question_id],
                        "target": target,
                        "target_text": target_payload(target),
                        "targets": {"answer_abstain_sft": target_payload(target)},
                        "input_condition": "crop_only",
                        "crop_path": media_row["media_relpath"],
                        "full_path": media_row["media_relpath"],
                        "media_sha256": media_row["media_sha256"],
                        "insufficiency_operator": None if evidence_state == "available" else condition,
                        "label_derivation": PROPERTY_RULES[question_id]
                        if evidence_state == "available"
                        else "rules-derived evidence unavailability; answer polarity is not assigned",
                    }
                )
        temporal_media = media[f"{event_id}::temporal_miss"]
        for polarity, question_id in (("yes", yes_question), ("no", no_question)):
            temporal_diagnostics.append(
                {
                    "schema_version": "postreview_temporal_diagnostic_row_v1",
                    "row_id": f"{event_id}::temporal_miss::{polarity}",
                    "event_id": event_id,
                    "group_id": event_id,
                    "partition": source["partition"],
                    "event_class": source["event_class"],
                    "question_id": question_id,
                    "question_polarity": polarity,
                    "question": source["property_questions"][question_id],
                    "source_available_target": polarity,
                    "target": "not_scored",
                    "diagnostic_reason": "outside annotated event interval does not prove semantic insufficiency",
                    "input_condition": "crop_only",
                    "crop_path": temporal_media["media_relpath"],
                    "full_path": temporal_media["media_relpath"],
                    "media_sha256": temporal_media["media_sha256"],
                }
            )
    rows.sort(key=lambda row: row["row_id"])
    by_partition = collections.Counter(row["partition"] for row in rows)
    events_by_partition = collections.Counter()
    for partition in {row["partition"] for row in rows}:
        events_by_partition[partition] = len({row["event_id"] for row in rows if row["partition"] == partition})
    for partition, event_count in events_by_partition.items():
        if by_partition[partition] != event_count * 4:
            raise RuntimeError(f"four-row 2x2 invariant failed for {partition}")
    question_audits = {}
    for partition in ("train", "dev", "prospective_test"):
        available = [
            row
            for row in rows
            if row["partition"] == partition and row["evidence_state"] == "available"
        ]
        question_counts = {
            question_id: collections.Counter(
                row["target"] for row in available if row["question_id"] == question_id
            )
            for question_id in PROPERTY_QUESTIONS
        }
        used_counts = [values for values in question_counts.values() if sum(values.values())]
        question_only_accuracy = (
            sum(max(values["yes"], values["no"]) for values in used_counts)
            / sum(sum(values.values()) for values in used_counts)
            if used_counts
            else None
        )
        gate = bool(used_counts) and all(
            values["yes"] > 0 and values["no"] > 0 for values in used_counts
        ) and question_only_accuracy is not None and question_only_accuracy <= 0.60
        question_audits[partition] = {
            "target_counts": {
                key: dict(sorted(value.items())) for key, value in question_counts.items()
            },
            "question_only_majority_accuracy": question_only_accuracy,
            "gate_max_0_60": gate,
        }
    question_shortcut_gate = all(
        question_audits[partition]["gate_max_0_60"]
        for partition in events_by_partition
    )
    write_jsonl(args.output_root / "manifests" / "final_rows.jsonl", rows)
    write_jsonl(args.output_root / "manifests" / "temporal_diagnostic_rows.jsonl", temporal_diagnostics)
    write_jsonl(args.output_root / "manifests" / "manifest_exclusions.jsonl", exclusions or [{"status": "none"}])
    report = {
        "schema_version": "postreview_2x2_manifest_report_v1",
        "status": "PASS"
        if events_by_partition.get("prospective_test", 0) >= args.minimum_test_events
        and question_shortcut_gate
        else "BLOCKED",
        "rows": len(rows),
        "temporal_diagnostic_rows_not_scored": len(temporal_diagnostics),
        "events_by_partition": dict(sorted(events_by_partition.items())),
        "rows_by_partition": dict(sorted(by_partition.items())),
        "excluded_events": len(exclusions),
        "minimum_test_events": args.minimum_test_events,
        "available_question_audits_by_partition": question_audits,
        "question_shortcut_gate_max_0_60": question_shortcut_gate,
        "model_outputs_read": False,
        "human_judgment_used": False,
    }
    write_json(args.output_root / "manifest_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


def strict_parse(raw: str) -> tuple[str | None, str | None]:
    match = re.fullmatch(r"\s*(\{.*\})\s*", raw, flags=re.DOTALL)
    if not match:
        return None, "not_one_json_object"
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        return None, f"json_decode:{exc.msg}"
    if not isinstance(payload, dict) or set(payload) != {"decision", "answer"}:
        return None, "schema_keys"
    decision = payload["decision"]
    answer = payload["answer"]
    if decision == "answer" and answer in {"yes", "no"}:
        return str(answer), None
    if decision == "abstain" and answer == "abstain":
        return "abstain", None
    return None, "schema_values"


def runtime_rows(args: argparse.Namespace, partition: str) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(args.output_root / "manifests" / "final_rows.jsonl") if row["partition"] == partition]
    if not rows:
        raise RuntimeError(f"no rows for {partition}")
    resolved = []
    for row in rows:
        media = Path(row["crop_path"])
        if not media.is_absolute():
            media = args.output_root / media
        if not media.is_file() or sha256_file(media) != row["media_sha256"]:
            raise RuntimeError(f"media hash mismatch: {row['row_id']}")
        resolved.append(row | {"crop_path": str(media.resolve()), "full_path": str(media.resolve())})
    return resolved


def runtime_temporal_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(args.output_root / "manifests" / "temporal_diagnostic_rows.jsonl")
        if row["partition"] == "prospective_test"
    ]
    if not rows:
        raise RuntimeError("no prospective temporal diagnostic rows")
    resolved = []
    for row in rows:
        media = args.output_root / row["crop_path"]
        if not media.is_file() or sha256_file(media) != row["media_sha256"]:
            raise RuntimeError(f"temporal media hash mismatch: {row['row_id']}")
        resolved.append(row | {"crop_path": str(media.resolve()), "full_path": str(media.resolve())})
    return resolved


def train(args: argparse.Namespace) -> int:
    if args.arm not in {"paired_sft", "answer_only_sft"}:
        raise ValueError("unknown trained arm")
    from scripts.operator import w5a_fair_baselines_runner as w5a

    train_rows = runtime_rows(args, "train")
    dev_rows = runtime_rows(args, "dev")
    if args.arm == "answer_only_sft":
        available_train = [row for row in train_rows if row["evidence_state"] == "available"]
        train_rows = [
            row | {"row_id": f"{row['row_id']}::repeat{repeat}"}
            for repeat in (1, 2)
            for row in available_train
        ]
        dev_rows = [row for row in dev_rows if row["evidence_state"] == "available"]
    w5a.EXPLICIT_SYSTEM_PROMPT = SYSTEM_PROMPT
    w5a.SEED = SEED
    w5a.TRAIN_ROWS = len(train_rows)
    w5a.UPDATE_TOKEN_SLOTS = len(train_rows) * PASSES * SEQUENCE_SLOTS
    w5a.UPDATES_PER_PASS = math.ceil(len(train_rows) / GRADIENT_ACCUMULATION)
    w5a.TOTAL_UPDATES = w5a.UPDATES_PER_PASS * PASSES
    w5a.TRAIN_PASSES = PASSES
    w5a.SEQUENCE_SLOTS = SEQUENCE_SLOTS
    w5a.GRADIENT_ACCUMULATION = GRADIENT_ACCUMULATION
    work_root = args.output_root / f"_{args.arm}_work"
    if work_root.exists():
        raise RuntimeError(f"refusing to overwrite training work root: {work_root}")
    proxy = argparse.Namespace(arm="answer_abstain_sft", output_root=work_root)
    w5a.train_sft(proxy, train_rows, dev_rows)
    source = work_root / "training" / "answer_abstain_sft"
    destination = args.output_root / "training" / args.arm
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite {destination}")
    source.rename(destination)
    selection = json.loads((destination / "checkpoint_selection.json").read_text(encoding="utf-8"))
    old = Path(selection["best_checkpoint"])
    selection["arm"] = args.arm
    selection["best_checkpoint"] = str(destination / old.name)
    selection["training_contract"] = (
        "all four rows per event with abstention targets"
        if args.arm == "paired_sft"
        else "available Yes/No rows repeated twice; same update-token budget; no abstention targets"
    )
    write_json(destination / "checkpoint_selection.json", selection)
    (work_root / "training").rmdir()
    work_root.rmdir()
    return 0


def inference_messages(processor: Any, row: dict[str, Any]) -> tuple[str, list[Any]]:
    from scripts.operator import answerer_alignment_lora as base

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": base.user_content(row)},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, base.row_images(row)


def infer(args: argparse.Namespace) -> int:
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if args.partition == "dev":
        if args.scope != "primary":
            raise RuntimeError("dev pilot supports primary 2x2 inference only")
        rows = runtime_rows(args, "dev")
    else:
        rows = (
            runtime_rows(args, "prospective_test")
            if args.scope == "primary"
            else runtime_temporal_rows(args)
        )
    adapter = None
    if args.arm in {"paired_sft", "answer_only_sft"}:
        selection = json.loads((args.output_root / "training" / args.arm / "checkpoint_selection.json").read_text(encoding="utf-8"))
        adapter = Path(selection["best_checkpoint"])
    elif args.arm == "legacy_lora":
        if args.legacy_adapter is None:
            raise ValueError("--legacy-adapter is required")
        adapter = args.legacy_adapter
    if adapter is not None and not adapter.is_dir():
        raise FileNotFoundError(adapter)
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        enforce_eager=True,
        enable_lora=adapter is not None,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 1},
        max_lora_rank=16,
        max_model_len=4096,
        tensor_parallel_size=1,
        seed=SEED,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32, n=1, seed=SEED, logprobs=1)
    request = LoRARequest(args.arm, 1, lora_path=str(adapter)) if adapter else None
    suffix = "" if args.scope == "primary" else "_temporal"
    path = args.output_root / "responses" / f"{args.arm}{suffix}.jsonl"
    existing = {row["row_id"]: row for row in read_jsonl(path)} if path.is_file() else {}
    pending = [row for row in rows if row["row_id"] not in existing]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            prompts, opened = [], []
            for row in batch:
                prompt, images = inference_messages(processor, row)
                opened.extend(images)
                prompts.append({"prompt": prompt, "multi_modal_data": {"image": images}})
            outputs = llm.generate(prompts, sampling_params=sampling, lora_request=request, use_tqdm=False)
            for row, output in zip(batch, outputs, strict=True):
                candidate = output.outputs[0]
                if not candidate.token_ids:
                    raise RuntimeError(f"empty generation: {row['row_id']}")
                prediction, parse_error = strict_parse(candidate.text)
                result = row | {
                    "evidence_status": "development_pilot_not_registered"
                    if args.partition == "dev"
                    else "prospective_test",
                    "arm": args.arm,
                    "prediction": prediction,
                    "parse_error": parse_error,
                    "exact_correct": prediction == row["target"] if args.scope == "primary" else None,
                    "confidence": math.exp(candidate.cumulative_logprob / len(candidate.token_ids)),
                    "raw_response": candidate.text,
                }
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            for image in opened:
                image.close()
            print(f"{args.arm}/{args.scope} {min(offset + len(batch), len(pending))}/{len(pending)}", flush=True)
    if len(read_jsonl(path)) != len(rows):
        raise RuntimeError(f"incomplete inference for {args.arm}")
    return 0


def metric(rows: list[dict[str, Any]], predicate) -> float:
    if not rows:
        raise ValueError("empty metric subset")
    return sum(bool(predicate(row)) for row in rows) / len(rows)


def arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scipy.stats import beta

    def exact_interval(successes: int, total: int) -> list[float]:
        low = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, total - successes + 1))
        high = 1.0 if successes == total else float(beta.ppf(0.975, successes + 1, total - successes))
        return [low, high]

    cells = {}
    for evidence in ("available", "unavailable"):
        for polarity in ("yes", "no"):
            key = f"{evidence}_{polarity}"
            subset = [row for row in rows if row["evidence_state"] == evidence and row["question_polarity"] == polarity]
            successes = sum(bool(row["exact_correct"]) for row in subset)
            cells[key] = {
                "n": len(subset),
                "correct": successes,
                "exact_accuracy": successes / len(subset),
                "clopper_pearson_95": exact_interval(successes, len(subset)),
            }
    by_event: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_event[str(row["event_id"])].append(row)
    if any(len(values) != 4 for values in by_event.values()):
        raise RuntimeError("incomplete four-row event")
    risk_coverage = {}
    available = [row for row in rows if row["evidence_state"] == "available"]
    ordered = sorted(available, key=lambda row: (-float(row["confidence"]), str(row["row_id"])))
    for coverage in (0.5, 0.75, 0.9, 1.0):
        count = max(1, math.ceil(len(ordered) * coverage))
        selected = ordered[:count]
        risk_coverage[str(coverage)] = {
            "rows": count,
            "realized_coverage": count / len(ordered),
            "exact_risk": 1.0 - metric(selected, lambda row: row["exact_correct"]),
        }
    return {
        "rows": len(rows),
        "events": len(by_event),
        "four_cell_macro_exact_accuracy": sum(value["exact_accuracy"] for value in cells.values()) / 4,
        "all_four_correct_event_rate": sum(all(row["exact_correct"] for row in values) for values in by_event.values()) / len(by_event),
        "available_semantic_accuracy": metric([row for row in rows if row["evidence_state"] == "available"], lambda row: row["exact_correct"]),
        "unavailable_abstention_rate": metric([row for row in rows if row["evidence_state"] == "unavailable"], lambda row: row["prediction"] == "abstain"),
        "parse_error_rate": metric(rows, lambda row: row["parse_error"] is not None),
        "cells": cells,
        "available_common_coverage_risk": risk_coverage,
    }


def bootstrap_delta(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, float]:
    left_by_id = {row["row_id"]: row for row in left}
    right_by_id = {row["row_id"]: row for row in right}
    if set(left_by_id) != set(right_by_id):
        raise RuntimeError("paired response identity mismatch")
    events = sorted({str(row["event_id"]) for row in left})
    rng = random.Random(SEED)
    left_event = {
        event_id: sum(bool(row["exact_correct"]) for row in left if row["event_id"] == event_id) / 4
        for event_id in events
    }
    right_event = {
        event_id: sum(bool(row["exact_correct"]) for row in right if row["event_id"] == event_id) / 4
        for event_id in events
    }
    event_deltas = {event_id: left_event[event_id] - right_event[event_id] for event_id in events}
    observed = sum(event_deltas.values()) / len(events)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [rng.choice(events) for _ in events]
        draws.append(sum(event_deltas[event_id] for event_id in sample) / len(sample))
    draws.sort()
    return {
        "observed": observed,
        "ci95_low": draws[int(0.025 * BOOTSTRAP_REPLICATES)],
        "ci95_high": draws[min(BOOTSTRAP_REPLICATES - 1, int(0.975 * BOOTSTRAP_REPLICATES))],
    }


def exact_mcnemar(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_id = {row["row_id"]: bool(row["exact_correct"]) for row in left}
    right_by_id = {row["row_id"]: bool(row["exact_correct"]) for row in right}
    if set(left_by_id) != set(right_by_id):
        raise RuntimeError("paired response identity mismatch")
    left_only = sum(left_by_id[key] and not right_by_id[key] for key in left_by_id)
    right_only = sum(right_by_id[key] and not left_by_id[key] for key in left_by_id)
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, value) for value in range(min(left_only, right_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "left_correct_right_wrong": left_only,
        "left_wrong_right_correct": right_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def aggregate(args: argparse.Namespace) -> int:
    responses = {arm: read_jsonl(args.output_root / "responses" / f"{arm}.jsonl") for arm in ARMS}
    temporal = {
        arm: read_jsonl(args.output_root / "responses" / f"{arm}_temporal.jsonl")
        for arm in ARMS
    }
    temporal_metrics = {}
    for arm in ARMS:
        available = {
            (row["event_id"], row["question_polarity"]): row
            for row in responses[arm]
            if row["evidence_state"] == "available"
        }
        diagnostic = temporal[arm]
        if len(diagnostic) != len(available):
            raise RuntimeError(f"temporal identity/count mismatch: {arm}")
        paired = [
            (available[(row["event_id"], row["question_polarity"])], row)
            for row in diagnostic
        ]
        temporal_metrics[arm] = {
            "rows": len(diagnostic),
            "scored_for_correctness": False,
            "abstention_rate": metric(diagnostic, lambda row: row["prediction"] == "abstain"),
            "prediction_change_rate_from_available": sum(left["prediction"] != right["prediction"] for left, right in paired) / len(paired),
        }
    report = {
        "schema_version": "postreview_2x2_results_v1",
        "status": "POST_OPEN_ANALYSIS_COMPLETE",
        "primary_has_no_confidence_gate": True,
        "strict_substring_fallback_used": False,
        "arms": {arm: arm_metrics(rows) for arm, rows in responses.items()},
        "paired_sft_minus_explicit_prompt": bootstrap_delta(responses["paired_sft"], responses["explicit_prompt"]),
        "paired_sft_minus_answer_only_sft": bootstrap_delta(responses["paired_sft"], responses["answer_only_sft"]),
        "paired_sft_vs_explicit_prompt_mcnemar": exact_mcnemar(responses["paired_sft"], responses["explicit_prompt"]),
        "paired_sft_vs_answer_only_sft_mcnemar": exact_mcnemar(responses["paired_sft"], responses["answer_only_sft"]),
        "temporal_offset_diagnostic": temporal_metrics,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "claim_boundary": "prospective post-review 2x2 and perturbation evidence; not a replacement for original B1 and not end-to-end localization",
    }
    write_json(args.output_root / "results" / "aggregate.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def nuisance_audit(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = [
        row
        for row in read_jsonl(args.output_root / "manifests" / "final_rows.jsonl")
        if row["partition"] in args.partitions
    ]
    unique = {row["media_sha256"]: row for row in rows}
    features, labels, groups = [], [], []
    feature_names = ["pixel_mean", "pixel_std", "edge_density", "black_pixel_ratio", "jpeg_bytes"]
    for row in unique.values():
        path = args.output_root / row["crop_path"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise RuntimeError(f"cannot read nuisance-audit image: {path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        features.append(
            [
                float(image.mean()),
                float(image.std()),
                float(np.mean(edges > 0)),
                float(np.mean(gray < 8)),
                float(path.stat().st_size),
            ]
        )
        labels.append(int(row["evidence_state"] == "available"))
        groups.append(str(row["event_id"]))
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    group = np.asarray(groups)
    if len(set(group)) < 5 or set(y.tolist()) != {0, 1}:
        raise RuntimeError("nuisance audit requires at least five paired events")
    predictions = np.zeros(len(y), dtype=np.float64)
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for train_index, test_index in splitter.split(x, y, group):
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=SEED,
            ),
        )
        estimator.fit(x[train_index], y[train_index])
        predictions[test_index] = estimator.predict_proba(x[test_index])[:, 1]
    auroc = float(roc_auc_score(y, predictions))
    payload = {
        "schema_version": "postreview_2x2_nuisance_audit_v1",
        "status": "PASS" if auroc <= args.maximum_auroc else "FAIL",
        "partitions": args.partitions,
        "rows": len(y),
        "events": len(set(group)),
        "features": feature_names,
        "estimator": "5-fold StratifiedGroupKFold StandardScaler plus balanced logistic regression",
        "group_key": "event_id",
        "group_cv_auroc": auroc,
        "maximum_auroc": args.maximum_auroc,
        "interpretation": "tests simple size, blankness, and low-level image-statistic shortcuts only; not semantic sufficiency",
        "model_outputs_read": False,
        "human_judgment_used": False,
    }
    write_json(args.output_root / "nuisance_audit.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 2


def resource_plan(args: argparse.Namespace) -> int:
    import statistics

    media_files = list((args.output_root / "media").rglob("*.jpg"))
    if not media_files:
        raise RuntimeError("no materialized development media for resource estimate")
    sizes = [path.stat().st_size for path in media_files]
    sources = read_jsonl(args.output_root / "manifests" / "source_candidates.jsonl")
    test_eligible = sum(
        row["partition"] == "prospective_test"
        and row["wrong_region_eligible"]
        and "yes" in row["property_labels"].values()
        and "no" in row["property_labels"].values()
        for row in sources
    )
    estimated_test_media = int(math.ceil(test_eligible * 3 * statistics.median(sizes)))
    adapter_and_optimizer_budget = 2 * 3 * 64 * 1024 * 1024 + 2 * 1024**3
    response_and_log_budget = 1 * 1024**3
    transfer_and_temp_budget = estimated_test_media * 3
    total_budget = adapter_and_optimizer_budget + response_and_log_budget + transfer_and_temp_budget
    remote = json.loads((args.output_root / "gpu_host_preflight.json").read_text(encoding="utf-8"))
    free_bytes_above_floor = max(0, int(remote["free_gib"] - remote["disk_safety_floor_gib"]) * 1024**3)
    payload = {
        "schema_version": "postreview_2x2_resource_plan_v1",
        "status": "PASS_WITHIN_80_GIB_FLOOR" if total_budget <= free_bytes_above_floor else "BLOCKED_DISK_BUDGET",
        "development_media": {"files": len(media_files), "bytes": sum(sizes), "median_bytes": statistics.median(sizes)},
        "prospective_test": {"eligible_events_before_video_integrity_scan": test_eligible, "panels_per_event": 3, "estimated_media_bytes": estimated_test_media},
        "server": {"free_gib": remote["free_gib"], "safety_floor_gib": remote["disk_safety_floor_gib"], "usable_bytes_above_floor": free_bytes_above_floor},
        "budgets": {
            "two_lora_training_and_checkpoint_work_bytes": adapter_and_optimizer_budget,
            "responses_and_logs_bytes": response_and_log_budget,
            "test_transfer_and_temporary_multiplier_bytes": transfer_and_temp_budget,
            "total_bytes": total_budget,
        },
        "new_model_download_required": False,
        "cached_model_revision": remote["model_revision"],
        "container_image_id": remote["vllm_image_id"],
        "gpu_execution_started": False,
    }
    write_json(args.output_root / "resource_plan.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"].startswith("PASS") else 2


def execution_dry_run(args: argparse.Namespace) -> int:
    imported_before = set(sys.modules)
    rows = read_jsonl(args.output_root / "manifests" / "final_rows.jsonl")
    train_rows = [row for row in rows if row["partition"] == "train"]
    dev_rows = [row for row in rows if row["partition"] == "dev"]
    available_train = [row for row in train_rows if row["evidence_state"] == "available"]
    answer_only_budget_rows = 2 * len(available_train)
    checks = {
        "train_rows_nonzero_multiple_of_4": bool(train_rows) and len(train_rows) % 4 == 0,
        "dev_rows_nonzero_multiple_of_4": bool(dev_rows) and len(dev_rows) % 4 == 0,
        "four_rows_per_train_event": len(train_rows) == 4 * len({row["event_id"] for row in train_rows}),
        "four_rows_per_dev_event": len(dev_rows) == 4 * len({row["event_id"] for row in dev_rows}),
        "paired_and_answer_only_supervised_row_budget_equal": len(train_rows) == answer_only_budget_rows,
        "targets_strict_enum": all(row["target"] in {"yes", "no", "abstain"} for row in rows),
        "available_balanced_per_event": all(
            {row["target"] for row in rows if row["event_id"] == event_id and row["evidence_state"] == "available"}
            == {"yes", "no"}
            for event_id in {row["event_id"] for row in rows}
        ),
        "unavailable_targets_abstain": all(
            row["target"] == "abstain"
            for row in rows
            if row["evidence_state"] == "unavailable"
        ),
        "prospective_test_not_materialized": not (args.output_root / "manifests" / "media_prospective_test.jsonl").exists(),
    }
    runtime_modules = sorted(
        name
        for name in set(sys.modules) - imported_before
        if name.split(".")[0] in {"torch", "transformers", "vllm", "peft"}
    )
    checks["no_model_runtime_loaded"] = not runtime_modules
    payload = {
        "schema_version": "postreview_2x2_execution_dry_run_v1",
        "status": "PASS_PRE_TEST" if all(checks.values()) else "FAIL",
        "checks": checks,
        "train": {"events": len({row["event_id"] for row in train_rows}), "paired_sft_rows": len(train_rows), "answer_only_repeated_rows": answer_only_budget_rows},
        "dev": {"events": len({row["event_id"] for row in dev_rows}), "rows": len(dev_rows)},
        "arms": ARMS,
        "model_runtime_modules_loaded": runtime_modules,
        "gpu_execution_started": False,
        "model_output_read": False,
    }
    write_json(args.output_root / "execution_dry_run.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"].startswith("PASS") else 2


def package_audit(args: argparse.Namespace) -> int:
    names = [
        "protocol_draft.json",
        "commands.json",
        "reproducibility_contract.json",
        "prepare_report.json",
        "launch_plan.json",
        "gpu_host_preflight.json",
        "resource_plan.json",
        "manifest_report.json",
        "nuisance_audit.json",
        "execution_dry_run.json",
        "manifests/source_candidates.jsonl",
        "manifests/media_train.jsonl",
        "manifests/media_train_exclusions.jsonl",
        "manifests/media_dev.jsonl",
        "manifests/media_dev_exclusions.jsonl",
        "manifests/final_rows.jsonl",
        "manifests/temporal_diagnostic_rows.jsonl",
        "manifests/manifest_exclusions.jsonl",
    ]
    artifacts = []
    for name in names:
        path = args.output_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "measurement_output": name.startswith("responses/") or name.startswith("results/"),
            }
        )
    payload = {
        "schema_version": "postreview_2x2_artifact_manifest_draft_v1",
        "status": "PRE_TEST_PACKAGE_COMPLETE",
        "artifacts": artifacts,
        "execution_script": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "bytes": Path(__file__).stat().st_size,
            "sha256": sha256_file(Path(__file__)),
        },
        "prospective_test_artifacts_present": False,
        "model_outputs_present": False,
        "gpu_execution_started": False,
    }
    write_json(args.output_root / "artifact_manifest_draft.json", payload)
    print(json.dumps({"status": payload["status"], "artifacts": len(artifacts), "script_sha256": payload["execution_script"]["sha256"]}, indent=2))
    return 0


def transfer_manifest(args: argparse.Namespace) -> int:
    primary = read_jsonl(args.output_root / "manifests" / "final_rows.jsonl")
    temporal = [
        row
        for row in read_jsonl(args.output_root / "manifests" / "temporal_diagnostic_rows.jsonl")
        if row["partition"] == "prospective_test"
    ]
    if not any(row["partition"] == "prospective_test" for row in primary):
        raise RuntimeError("prospective-test rows are absent")
    relative_paths = {
        str(Path("outputs") / args.output_root.name / row["crop_path"])
        for row in primary + temporal
    }
    support = {
        Path("outputs") / args.output_root.name / name
        for name in (
            "protocol_draft.json",
            "commands.json",
            "reproducibility_contract.json",
            "launch_plan.json",
            "resource_plan.json",
            "manifest_report.json",
            "nuisance_audit.json",
            "manifests/final_rows.jsonl",
            "manifests/temporal_diagnostic_rows.jsonl",
        )
    }
    relative_paths.update(path.as_posix() for path in support)
    rows = []
    for relative in sorted(relative_paths):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": "postreview_2x2_transfer_manifest_v1",
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "destination": "/workspace/MV-EviRAG",
        "source_videos_included": False,
        "prospective_media_only_if_consumed_by_final_manifest": True,
    }
    manifest_path = args.output_root / "transfer_manifest.json"
    write_json(manifest_path, manifest)
    manifest_relative = (Path("outputs") / args.output_root.name / "transfer_manifest.json").as_posix()
    file_list = [*(row["path"] for row in rows), manifest_relative]
    (args.output_root / "transfer_files.txt").write_text("\n".join(file_list) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "files": len(file_list), "bytes": manifest["total_bytes"] + manifest_path.stat().st_size}, indent=2))
    return 0


def verify_transfer(args: argparse.Namespace) -> int:
    manifest = json.loads((args.output_root / "transfer_manifest.json").read_text(encoding="utf-8"))
    mismatches = []
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if not path.is_file():
            mismatches.append({"path": row["path"], "reason": "missing"})
        elif path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            mismatches.append({"path": row["path"], "reason": "size_or_sha256"})
    payload = {
        "schema_version": "postreview_2x2_transfer_verification_v1",
        "status": "PASS" if not mismatches else "FAIL",
        "files": len(manifest["files"]),
        "total_bytes": manifest["total_bytes"],
        "mismatches": mismatches,
    }
    write_json(args.output_root / "transfer_verification.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not mismatches else 2


def remote_preflight(args: argparse.Namespace) -> int:
    command = """
set -eu
cd /workspace/MV-EviRAG
printf 'HOST=%s\\n' "$(hostname)"
printf 'HEAD=%s\\n' "$(git rev-parse HEAD)"
printf 'ORIGIN=%s\\n' "$(git rev-parse origin/main)"
printf 'STATUS_BEGIN\\n'
git status --porcelain
printf 'STATUS_END\\n'
printf 'GPU_BEGIN\\n'
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader,nounits
printf 'GPU_END\\n'
printf 'DISK_BEGIN\\n'
df -BG / | tail -n 1
printf 'DISK_END\\n'
test -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
printf 'MODEL_CACHE=present\\n'
printf 'MODEL_REV=%s\\n' "$(cat "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/refs/main")"
printf 'VLLM_IMAGE_ID=%s\\n' "$(docker image inspect vllm/vllm-openai:latest --format '{{.Id}}')"
""".strip()
    completed = subprocess.run(
        ["ssh", "gpu-host", command],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    lines = completed.stdout.splitlines()

    def scalar(prefix: str) -> str:
        values = [line.removeprefix(prefix) for line in lines if line.startswith(prefix)]
        if len(values) != 1:
            raise RuntimeError(f"remote preflight missing {prefix}")
        return values[0]

    def section(begin: str, end: str) -> list[str]:
        first, last = lines.index(begin), lines.index(end)
        return lines[first + 1 : last]

    status_lines = section("STATUS_BEGIN", "STATUS_END")
    gpu_lines = section("GPU_BEGIN", "GPU_END")
    disk_line = section("DISK_BEGIN", "DISK_END")
    if len(gpu_lines) != 3 or len(disk_line) != 1:
        raise RuntimeError("remote resource shape drift")
    available_match = re.search(r"\s(\d+)G\s+\d+%\s+/$", disk_line[0])
    if not available_match:
        raise RuntimeError(f"cannot parse remote disk line: {disk_line[0]}")
    free_gib = int(available_match.group(1))
    head, origin = scalar("HEAD="), scalar("ORIGIN=")
    payload = {
        "schema_version": "postreview_2x2_gpu_host_preflight_v1",
        "status": "RESOURCE_READY_GIT_SYNC_BLOCKED" if free_gib >= 80 else "BLOCKED_DISK_SAFETY_FLOOR",
        "host": scalar("HOST="),
        "repo": "/workspace/MV-EviRAG",
        "head": head,
        "origin_main": origin,
        "head_matches_origin": head == origin,
        "status_lines": status_lines,
        "clean": not status_lines,
        "gpus": gpu_lines,
        "disk_df": disk_line[0],
        "free_gib": free_gib,
        "disk_safety_floor_gib": 80,
        "model_cache": scalar("MODEL_CACHE="),
        "model_revision": scalar("MODEL_REV="),
        "vllm_image": VLLM_IMAGE,
        "vllm_image_id": scalar("VLLM_IMAGE_ID="),
        "gpu_process_launched": False,
        "model_loaded": False,
    }
    write_json(args.output_root / "gpu_host_preflight.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if free_gib >= 80 else 2


def preflight(args: argparse.Namespace) -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    origin = subprocess.run(["git", "rev-parse", "origin/main"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(Path(__file__).resolve().relative_to(ROOT))],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).returncode == 0
    ready = head == origin and not status and tracked
    payload = {
        "schema_version": "postreview_2x2_local_preflight_v1",
        "status": "READY" if ready else "BLOCKED_GIT_SYNC_OR_UNTRACKED_CODE",
        "head": head,
        "origin_main": origin,
        "clean": not status,
        "execution_script_tracked": tracked,
        "status_lines": status,
        "protocol_sha256": sha256_file(args.output_root / "protocol_draft.json"),
        "script_sha256": sha256_file(Path(__file__)),
        "model_loaded": False,
        "gpu_execution_started": False,
    }
    write_json(args.output_root / "local_preflight.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ready else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    materialize_parser = sub.add_parser("materialize")
    materialize_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    materialize_parser.add_argument("--partition", choices=("train", "dev", "prospective_test"), required=True)
    materialize_parser.add_argument("--allow-sealed-open", action="store_true")
    materialize_parser.add_argument("--limit", type=int, default=0)
    manifest_parser = sub.add_parser("build-manifest")
    manifest_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    manifest_parser.add_argument("--minimum-test-events", type=int, default=300)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    train_parser.add_argument("--arm", choices=("paired_sft", "answer_only_sft"), required=True)
    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    infer_parser.add_argument("--arm", choices=ARMS, required=True)
    infer_parser.add_argument("--legacy-adapter", type=Path)
    infer_parser.add_argument("--batch-size", type=int, default=8)
    infer_parser.add_argument(
        "--partition",
        choices=("dev", "prospective_test"),
        default="prospective_test",
    )
    infer_parser.add_argument("--scope", choices=("primary", "temporal"), default="primary")
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    nuisance_parser = sub.add_parser("nuisance-audit")
    nuisance_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    nuisance_parser.add_argument(
        "--partitions",
        nargs="+",
        choices=("train", "dev", "prospective_test"),
        default=["train", "dev"],
    )
    nuisance_parser.add_argument("--maximum-auroc", type=float, default=0.60)
    resource_parser = sub.add_parser("resource-plan")
    resource_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    dry_parser = sub.add_parser("execution-dry-run")
    dry_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    package_parser = sub.add_parser("package-audit")
    package_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    transfer_parser = sub.add_parser("transfer-manifest")
    transfer_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    verify_parser = sub.add_parser("verify-transfer")
    verify_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    remote_parser = sub.add_parser("remote-preflight")
    remote_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    return {
        "prepare": prepare,
        "materialize": materialize,
        "build-manifest": build_manifest,
        "train": train,
        "infer": infer,
        "aggregate": aggregate,
        "nuisance-audit": nuisance_audit,
        "resource-plan": resource_plan,
        "execution-dry-run": execution_dry_run,
        "package-audit": package_audit,
        "transfer-manifest": transfer_manifest,
        "verify-transfer": verify_transfer,
        "preflight": preflight,
        "remote-preflight": remote_preflight,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
