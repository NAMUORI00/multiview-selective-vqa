"""Exploratory answerer-alignment LoRA track (ignored operator code).

This script deliberately accepts only the frozen active-development artifacts
listed below.  It never discovers or reads heldout/reserve artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SEED = 20260710
ABSTAIN = "주어진 근거로는 판단 불가"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_OUTPUT = Path("outputs/answerer_alignment_lora_exploratory_20260710")
CORE_PATH = Path("outputs/dissection_empirical_package_20260710/step1_cross_core_benchmark.jsonl")
POSITIVE_INDEX_PATH = Path(
    "outputs/dissection_empirical_package_20260710/step1_positive_crop_index_private.jsonl"
)
HARD_NEGATIVE_PATH = Path(
    "outputs/dissection_empirical_package_20260710/required_augmentation/"
    "step3_same_category_cross_template_diagnostic.jsonl"
)
ACTIVE_EXAMPLES_PATH = Path("outputs/active_dev_examples.jsonl")
COUNTERFACTUAL_PATH = Path(
    "outputs/integrated_core_sufficiency_rebuild_20260709/counterfactual_rows_private.jsonl"
)
CONTEXT_PATH = Path(
    "outputs/dissection_empirical_package_20260710/step4_three_stage_table.jsonl"
)

FORBIDDEN_DATA_TOKENS = (
    "heldout-b",
    "heldout_b",
    "heldout-c",
    "heldout_c",
    "heldout-d",
    "heldout_d",
    "reserve",
)

EVENT_TARGETS = {
    "경계선을 통한 침입": (
        "사람이 표시된 경계선을 넘어 침입하고 있는가?",
        "사람이 표시된 경계선을 넘어 침입했다.",
    ),
    "비정상적인 경로로의 침범": (
        "사람이 비정상적인 경로로 침범하고 있는가?",
        "사람이 비정상적인 경로로 침범했다.",
    ),
    "신체적 충돌을 동반한 싸움": (
        "사람들이 신체적 충돌을 동반해 싸우고 있는가?",
        "사람들이 신체적 충돌을 동반해 싸우고 있다.",
    ),
    "오토바이 인도 주행": (
        "오토바이가 인도를 주행하고 있는가?",
        "오토바이가 인도를 주행하고 있다.",
    ),
    "이륜 이동수단 운전자 1인 헬멧 미착용": (
        "이륜 이동수단 운전자가 헬멧을 착용하지 않았는가?",
        "운전자는 헬멧을 착용하지 않았다.",
    ),
    "이륜 이동수단 탑승자 일부 또는 전체 헬멧 미착용": (
        "이륜 이동수단 탑승자 중 헬멧을 착용하지 않은 사람이 있는가?",
        "탑승자 중 일부 또는 전원이 헬멧을 착용하지 않았다.",
    ),
    "자전거 인도 주행": (
        "자전거가 인도를 주행하고 있는가?",
        "자전거가 인도를 주행하고 있다.",
    ),
    "전동킥보드 앞뒤 2인 탑승": (
        "전동킥보드에 두 사람이 앞뒤로 탑승했는가?",
        "전동킥보드에 두 사람이 앞뒤로 탑승했다.",
    ),
    "전동킥보드 인도 주행": (
        "전동킥보드가 인도를 주행하고 있는가?",
        "전동킥보드가 인도를 주행하고 있다.",
    ),
}

SYSTEM_PROMPT = (
    "당신은 제공된 시각 근거만 사용하는 안전사건 답변자다. 외부 지식이나 추측을 사용하지 않는다. "
    "근거가 질문의 사건을 직접 뒷받침하면 짧은 사실 문장으로 답하고, 직접 판단할 수 없으면 "
    f"정확히 '{ABSTAIN}'라고 답한다. 반드시 JSON 한 줄만 출력한다. "
    '형식은 {"decision":"answer 또는 abstain","answer":"짧은 답변"}이다.'
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL: {path}:{line_number}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fraction(text: str) -> float:
    raw = hashlib.sha256(f"{SEED}:{text}".encode()).digest()[:8]
    return int.from_bytes(raw, "big") / 2**64


def assert_no_forbidden_data(rows: Iterable[dict[str, Any]], label: str) -> None:
    for index, row in enumerate(rows):
        rendered = json.dumps(row, ensure_ascii=False).lower()
        hit = next((token for token in FORBIDDEN_DATA_TOKENS if token in rendered), None)
        if hit:
            raise RuntimeError(f"firewall violation in {label}[{index}]: {hit}")


def answer_payload(target_type: str, target_answer: str) -> str:
    decision = "answer" if target_type == "answer" else "abstain"
    answer = target_answer if target_type == "answer" else ABSTAIN
    return json.dumps({"decision": decision, "answer": answer}, ensure_ascii=False, separators=(",", ":"))


def question_and_target(event_class: str) -> tuple[str, str]:
    try:
        return EVENT_TARGETS[event_class]
    except KeyError as exc:
        raise ValueError(f"unsupported event class: {event_class}") from exc


def relative_media_path(output: Path, kind: str, name: str) -> Path:
    return output / "media" / kind / f"{name}.jpg"


def copy_image(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == sha256_file(source):
        return
    shutil.copy2(source, destination)


def extract_frame(video_path: Path, frame_id: int, destination: Path) -> None:
    import cv2

    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError(f"cannot read frame {frame_id}: {video_path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"cannot write frame: {destination}")
    finally:
        capture.release()


def ensure_media(
    output: Path,
    positive: dict[str, Any],
    active_example: dict[str, Any],
) -> tuple[str, str]:
    sample_id = positive["source_sample_id"]
    crop_destination = relative_media_path(output, "crop", sample_id)
    full_destination = relative_media_path(output, "full", sample_id)
    copy_image(Path(positive["crop_path"]), crop_destination)

    full_source_text = positive.get("full_path")
    full_source = Path(full_source_text) if full_source_text else None
    if full_source is not None and full_source.is_file():
        copy_image(full_source, full_destination)
    elif not full_destination.is_file():
        view = positive["core_view"]
        videos = active_example["main_input"]["videos"]
        matches = [item for item in videos if item["view"] == view]
        if len(matches) != 1:
            raise RuntimeError(f"expected one video for {sample_id}/{view}, got {len(matches)}")
        extract_frame(Path(matches[0]["video_path"]), int(positive["frame_id"]), full_destination)

    if not crop_destination.is_file() or not full_destination.is_file():
        raise RuntimeError(f"media construction failed: {sample_id}")
    return crop_destination.as_posix(), full_destination.as_posix()


def build_row(
    source: dict[str, Any],
    split: str,
    source_type: str,
    question_event_class: str,
    image_event_class: str,
    crop_path: str,
    full_path: str,
    target_type: str,
    row_id: str,
    input_condition: str | None = None,
) -> dict[str, Any]:
    question, target_answer = question_and_target(question_event_class)
    if input_condition is None:
        input_condition = "full_native_crop" if stable_fraction(row_id) < 0.8 else "crop_only"
    return {
        "category": source["image_category"] if "image_category" in source else source["category"],
        "crop_path": crop_path,
        "event_class": image_event_class,
        "full_path": full_path,
        "group_id": source["group_id"],
        "input_condition": input_condition,
        "question": question,
        "question_event_class": question_event_class,
        "row_id": row_id,
        "source_type": source_type,
        "split": split,
        "target_answer": target_answer if target_type == "answer" else ABSTAIN,
        "target_text": answer_payload(target_type, target_answer),
        "target_type": target_type,
    }


def build_command(args: argparse.Namespace) -> None:
    from sklearn.model_selection import train_test_split

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    explicit_inputs = (
        CORE_PATH,
        POSITIVE_INDEX_PATH,
        HARD_NEGATIVE_PATH,
        ACTIVE_EXAMPLES_PATH,
        COUNTERFACTUAL_PATH,
        CONTEXT_PATH,
    )
    for path in explicit_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    core = read_jsonl(CORE_PATH)
    positives = read_jsonl(POSITIVE_INDEX_PATH)
    hard_all = read_jsonl(HARD_NEGATIVE_PATH)
    active_examples = read_jsonl(ACTIVE_EXAMPLES_PATH)
    counterfactual = read_jsonl(COUNTERFACTUAL_PATH)
    hard_negatives = [row for row in hard_all if int(row["label"]) == 0]

    if len(core) != 1198 or len(positives) != 599 or len(hard_negatives) != 224:
        raise RuntimeError(
            f"frozen count mismatch: core={len(core)} positives={len(positives)} hard={len(hard_negatives)}"
        )
    core_ids = {row["group_id"] for row in core}
    positive_ids = {row["group_id"] for row in positives}
    hard_ids = {row["group_id"] for row in hard_negatives}
    if len(core_ids) != 599 or core_ids != positive_ids or not hard_ids <= core_ids:
        raise RuntimeError("frozen incident universe mismatch")
    if Counter(int(row["label"]) for row in core) != Counter({0: 599, 1: 599}):
        raise RuntimeError("core pair labels are not one positive and one negative per incident")

    positive_by_id = {row["group_id"]: row for row in positives}
    active_by_id = {row["sample_id"]: row for row in active_examples}
    if not core_ids <= active_by_id.keys():
        raise RuntimeError("active-development annotation coverage is incomplete")

    ids = sorted(core_ids)
    strata = [positive_by_id[sample_id]["event_class_private"] for sample_id in ids]
    train_ids_raw, val_ids_raw = train_test_split(
        ids, test_size=0.30, random_state=SEED, shuffle=True, stratify=strata
    )
    train_ids = set(train_ids_raw)
    val_ids = set(val_ids_raw)
    if len(train_ids) != 419 or len(val_ids) != 180 or train_ids & val_ids:
        raise RuntimeError("incident split invariant failed")

    media: dict[str, tuple[str, str]] = {}
    for index, sample_id in enumerate(ids, 1):
        media[sample_id] = ensure_media(output, positive_by_id[sample_id], active_by_id[sample_id])
        if index % 50 == 0:
            print(f"media {index}/599", flush=True)

    supervised: list[dict[str, Any]] = []
    for source in core:
        sample_id = source["group_id"]
        split = "train" if sample_id in train_ids else "val"
        is_positive = int(source["label"]) == 1
        supervised.append(
            build_row(
                source=source,
                split=split,
                source_type="native_positive" if is_positive else "cross_category_negative",
                question_event_class=source["question_event_class"],
                image_event_class=source["image_event_class"],
                crop_path=media[sample_id][0],
                full_path=media[sample_id][1],
                target_type="answer" if is_positive else "abstain",
                row_id=f"core::{source['row_id']}",
            )
        )
    for source in hard_negatives:
        sample_id = source["group_id"]
        split = "train" if sample_id in train_ids else "val"
        supervised.append(
            build_row(
                source=source,
                split=split,
                source_type="same_category_hard_negative",
                question_event_class=source["question_event_class"],
                image_event_class=source["image_event_class"],
                crop_path=media[sample_id][0],
                full_path=media[sample_id][1],
                target_type="abstain",
                row_id=f"hard::{source['row_id']}",
            )
        )

    train_rows = sorted((row for row in supervised if row["split"] == "train"), key=lambda row: row["row_id"])
    val_rows = sorted((row for row in supervised if row["split"] == "val"), key=lambda row: row["row_id"])
    if {row["group_id"] for row in train_rows} & {row["group_id"] for row in val_rows}:
        raise RuntimeError("derived-row incident leakage")
    assert_no_forbidden_data(train_rows, "train")
    assert_no_forbidden_data(val_rows, "val")

    eval_core = []
    for source in core:
        if source["group_id"] not in val_ids:
            continue
        is_positive = int(source["label"]) == 1
        eval_core.append(
            build_row(
                source=source,
                split="val",
                source_type="native_positive" if is_positive else "cross_category_negative",
                question_event_class=source["question_event_class"],
                image_event_class=source["image_event_class"],
                crop_path=media[source["group_id"]][0],
                full_path=media[source["group_id"]][1],
                target_type="answer" if is_positive else "abstain",
                row_id=f"core::{source['row_id']}",
                input_condition="full_native_crop",
            )
        )
    for source in hard_negatives:
        if source["group_id"] not in val_ids:
            continue
        eval_core.append(
            build_row(
                source=source,
                split="val",
                source_type="same_category_hard_negative",
                question_event_class=source["question_event_class"],
                image_event_class=source["image_event_class"],
                crop_path=media[source["group_id"]][0],
                full_path=media[source["group_id"]][1],
                target_type="abstain",
                row_id=f"hard::{source['row_id']}",
                input_condition="full_native_crop",
            )
        )

    eval_counterfactual: list[dict[str, Any]] = []
    counterfactual_ids = {row["group_id"] for row in counterfactual}
    reusable_val_ids = val_ids & counterfactual_ids
    for source in counterfactual:
        sample_id = source["group_id"]
        if sample_id not in reusable_val_ids:
            continue
        media_name = re.sub(r"[^A-Za-z0-9_.-]", "_", source["row_id"])
        crop_destination = relative_media_path(output, "counterfactual_crop", media_name)
        full_destination = relative_media_path(output, "counterfactual_full", media_name)
        copy_image(Path(source["crop_path"]), crop_destination)
        copy_image(Path(source["full_path"]), full_destination)
        is_positive = int(source["label"]) == 1
        eval_counterfactual.append(
            build_row(
                source=source,
                split="val",
                source_type=f"counterfactual_{source['counterfactual_role']}",
                question_event_class=source["event_class_private"],
                image_event_class=source["event_class_private"],
                crop_path=crop_destination.as_posix(),
                full_path=full_destination.as_posix(),
                target_type="answer" if is_positive else "abstain",
                row_id=f"counterfactual::{source['row_id']}",
                input_condition="full_native_crop",
            )
            | {
                "counterfactual_role": source["counterfactual_role"],
                "operator": source["operator"],
                "sufficiency_condition": source["sufficiency_condition"],
            }
        )

    eval_core.sort(key=lambda row: row["row_id"])
    eval_counterfactual.sort(key=lambda row: row["row_id"])
    assert_no_forbidden_data(eval_core, "eval_core")
    assert_no_forbidden_data(eval_counterfactual, "eval_counterfactual")

    split_payload = {
        "exploratory": True,
        "seed": SEED,
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
    }
    write_json(output / "incident_split.json", split_payload)
    write_jsonl(output / "train.jsonl", train_rows)
    write_jsonl(output / "val.jsonl", val_rows)
    write_jsonl(output / "eval_core_val.jsonl", eval_core)
    write_jsonl(output / "eval_counterfactual_val.jsonl", eval_counterfactual)

    def counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[key]) for row in rows).items()))

    stats = {
        "exploratory": True,
        "firewall": {
            "api_used": False,
            "forbidden_split_rows": 0,
            "incident_overlap": len(train_ids & val_ids),
            "manuscript_modified": False,
            "reserve_used": False,
        },
        "hard_negative_events": len(hard_ids),
        "media": {
            "counterfactual_crop_files": len(list((output / "media/counterfactual_crop").glob("*.jpg"))),
            "counterfactual_full_files": len(list((output / "media/counterfactual_full").glob("*.jpg"))),
            "crop_files": len(list((output / "media/crop").glob("*.jpg"))),
            "full_files": len(list((output / "media/full").glob("*.jpg"))),
        },
        "reusable_counterfactual_val_events": len(reusable_val_ids),
        "train": {
            "events": len({row["group_id"] for row in train_rows}),
            "input_condition": counts(train_rows, "input_condition"),
            "rows": len(train_rows),
            "source_type": counts(train_rows, "source_type"),
            "target_type": counts(train_rows, "target_type"),
        },
        "val": {
            "events": len({row["group_id"] for row in val_rows}),
            "input_condition": counts(val_rows, "input_condition"),
            "rows": len(val_rows),
            "source_type": counts(val_rows, "source_type"),
            "target_type": counts(val_rows, "target_type"),
        },
    }
    write_json(output / "data_statistics.json", stats)

    design = {
        "base_model": MODEL_ID,
        "checkpoint_selection": "minimum mixed-validation teacher-forced loss; development val only",
        "comparison": "tuned-8B versus zero-shot-8B under identical deterministic prompts and decoding",
        "decision_rule": {
            "improvement_established_exploratory": (
                "paired incident-bootstrap E+ gain CI95 > 0 and tuned gate delta Risk@50 > 0 "
                "with zero-shot <= 0, without shortcut flag"
            ),
            "partial": "significant E+ gain without gate transition and without shortcut flag",
            "shortcut_flag": (
                "cross-category abstention minus same-category-hard-negative abstention > 0.10 "
                "with incident-bootstrap CI95 lower bound > 0"
            ),
        },
        "exploratory": True,
        "firewall": {
            "allowed_input_files": {str(path): sha256_file(path) for path in explicit_inputs},
            "api_calls": 0,
            "forbidden_split_families": list(FORBIDDEN_DATA_TOKENS),
            "selection_universe": "599 active-development incidents split 70/30 by incident",
        },
        "input_mix": "deterministic 80% full frame (max side 1024) + native crop; 20% native crop only",
        "lora": {
            "alpha": 32,
            "dropout": 0.05,
            "epochs_max": 3,
            "rank": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "primary_metrics": [
            "E+ answer accuracy",
            "E- abstention rate",
            "P- assertion rate",
            "Pr recovery after P- abstention",
            "true-label gate delta Risk@50 versus confidence ranking",
            "same-category hard-negative abstention accuracy",
        ],
        "precedent": {
            "paper": "Can Video LLMs Refuse to Answer? Alignment for Answerability in Video Large Language Models",
            "use": "motivation only; this track uses supervised LoRA rather than claiming protocol equivalence",
        },
        "target_supervision": "annotation-derived event answer for E+; fixed abstention for E-",
    }
    write_json(output / "design_summary.json", design)

    manifest = {
        "artifacts": {
            path.name: sha256_file(path)
            for path in (
                output / "incident_split.json",
                output / "train.jsonl",
                output / "val.jsonl",
                output / "eval_core_val.jsonl",
                output / "eval_counterfactual_val.jsonl",
                output / "data_statistics.json",
                output / "design_summary.json",
            )
        },
        "builder_sha256": sha256_file(Path(__file__)),
        "exploratory": True,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": SEED,
    }
    write_json(output / "build_manifest.json", manifest)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


def load_pil(path: str, max_side: int | None = None):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if max_side is not None and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def user_content(row: dict[str, Any]) -> list[dict[str, str]]:
    if row["input_condition"] == "full_native_crop":
        return [
            {"type": "text", "text": "전체 프레임:"},
            {"type": "image"},
            {"type": "text", "text": "원해상도 판단영역:"},
            {"type": "image"},
            {"type": "text", "text": f"질문: {row['question']}"},
        ]
    if row["input_condition"] == "crop_only":
        return [
            {"type": "text", "text": "원해상도 판단영역:"},
            {"type": "image"},
            {"type": "text", "text": f"질문: {row['question']}"},
        ]
    raise ValueError(f"unexpected input condition: {row['input_condition']}")


def row_images(row: dict[str, Any]):
    if row["input_condition"] == "full_native_crop":
        return [load_pil(row["full_path"], max_side=1024), load_pil(row["crop_path"])]
    return [load_pil(row["crop_path"])]


def encode_training_rows(processor: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    prompt_texts = []
    full_texts = []
    image_batches = []
    for row in rows:
        prompt_messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content(row)},
        ]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": row["target_text"]}]}
        ]
        prompt_texts.append(
            processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        )
        full_texts.append(
            processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        )
        image_batches.append(row_images(row))
    prompt = processor(
        text=prompt_texts, images=image_batches, padding=True, return_tensors="pt"
    )
    encoded = processor(text=full_texts, images=image_batches, padding=True, return_tensors="pt")
    labels = encoded["input_ids"].clone()
    prompt_lengths = prompt["attention_mask"].sum(dim=1)
    for index, prompt_length_tensor in enumerate(prompt_lengths):
        prompt_length = int(prompt_length_tensor)
        if prompt_length >= int(encoded["attention_mask"][index].sum()):
            raise RuntimeError(f"no supervised tokens for {rows[index]['row_id']}")
        labels[index, :prompt_length] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    encoded["labels"] = labels
    if int((labels != -100).sum()) == 0:
        raise RuntimeError("empty label mask")
    for images in image_batches:
        for image in images:
            image.close()
    return {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}


def encode_training_row(processor: Any, row: dict[str, Any]) -> dict[str, Any]:
    return encode_training_rows(processor, [row])


class JsonlDataset:
    def __init__(self, path: Path):
        self.rows = read_jsonl(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def environment_payload(torch: Any) -> dict[str, Any]:
    gpu = torch.cuda.get_device_properties(0)
    return {
        "cuda": torch.version.cuda,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": gpu.name,
        "gpu_total_memory": gpu.total_memory,
        "hostname": os.uname().nodename,
        "python": sys.version,
        "torch": torch.__version__,
    }


def load_model_and_processor(adapter: Path | None, trainable: bool):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    # This host's system cuDNN 8 and PyTorch wheel cuDNN 9 collide on Conv3d.
    # Qwen3-VL uses Conv3d only for visual patch embedding; native CUDA is exact
    # enough for this exploratory track and avoids the broken host library path.
    torch.backends.cudnn.enabled = False
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    model.to("cuda:0")
    return model, processor


def train_command(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    output = args.output
    train_path = output / "train.jsonl"
    val_path = output / "val.jsonl"
    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    assert_no_forbidden_data(train_rows, "train")
    assert_no_forbidden_data(val_rows, "val")
    if {row["group_id"] for row in train_rows} & {row["group_id"] for row in val_rows}:
        raise RuntimeError("training firewall: incident leakage")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bf16 GPU is required")

    torch.manual_seed(SEED)
    random.seed(SEED)
    model, processor = load_model_and_processor(adapter=None, trainable=True)
    model.config.use_cache = False
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_dataset = JsonlDataset(train_path)
    val_dataset = JsonlDataset(val_path)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda rows: rows,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.validation_batch_size,
        shuffle=False,
        collate_fn=lambda rows: rows,
    )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * 0.03)),
        num_training_steps=total_updates,
    )

    run_dir = output / "training"
    run_dir.mkdir(parents=True, exist_ok=True)
    curve: list[dict[str, Any]] = []
    best_epoch: int | None = None
    best_val = float("inf")
    stale_epochs = 0
    global_update = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        accumulated = 0
        for step, rows in enumerate(train_loader, 1):
            batch = {
                key: value.to("cuda:0")
                for key, value in encode_training_rows(processor, rows).items()
            }
            output_obj = model(**batch)
            if not torch.isfinite(output_obj.loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch} step={step}")
            loss = output_obj.loss / args.gradient_accumulation
            loss.backward()
            epoch_losses.append(float(output_obj.loss.detach().cpu()))
            accumulated += 1
            should_update = accumulated == args.gradient_accumulation or step == len(train_loader)
            if should_update:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                accumulated = 0
                curve.append(
                    {
                        "epoch": epoch,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "split": "train_update",
                        "step": global_update,
                        "value": sum(epoch_losses[-args.gradient_accumulation :])
                        / min(args.gradient_accumulation, len(epoch_losses)),
                    }
                )
                if global_update % 10 == 0:
                    print(
                        f"epoch={epoch} update={global_update}/{total_updates} "
                        f"loss={curve[-1]['value']:.6f}",
                        flush=True,
                    )

        model.eval()
        val_loss_sum = 0.0
        val_tokens = 0
        with torch.inference_mode():
            for index, rows in enumerate(val_loader, 1):
                batch = {
                    key: value.to("cuda:0")
                    for key, value in encode_training_rows(processor, rows).items()
                }
                output_obj = model(**batch)
                supervised_tokens = int((batch["labels"] != -100).sum())
                val_loss_sum += float(output_obj.loss.detach().cpu()) * supervised_tokens
                val_tokens += supervised_tokens
                if index % 50 == 0:
                    print(f"validation epoch={epoch} {index}/{len(val_loader)}", flush=True)
        if val_tokens == 0:
            raise RuntimeError("validation produced zero supervised tokens")
        val_loss = val_loss_sum / val_tokens
        curve.append({"epoch": epoch, "split": "validation", "step": global_update, "value": val_loss})
        checkpoint = run_dir / f"epoch-{epoch}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        processor.save_pretrained(checkpoint)
        print(f"epoch={epoch} val_loss={val_loss:.6f} checkpoint={checkpoint}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stopping_patience:
                print(f"early stopping after epoch {epoch}", flush=True)
                break

    if best_epoch is None:
        raise RuntimeError("no checkpoint selected")
    selection = {
        "best_checkpoint": str(run_dir / f"epoch-{best_epoch}"),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "elapsed_seconds": time.time() - started,
        "environment": environment_payload(torch),
        "exploratory": True,
        "selection_data": "development incident validation split only",
        "training_data_sha256": sha256_file(train_path),
        "validation_data_sha256": sha256_file(val_path),
    }
    write_json(run_dir / "checkpoint_selection.json", selection)
    write_json(run_dir / "training_curve.json", curve)
    with (run_dir / "training_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "split", "step", "value", "learning_rate"])
        writer.writeheader()
        for row in curve:
            writer.writerow(row)
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)


def generation_prompts(processor: Any, rows: list[dict[str, Any]]) -> tuple[Any, list[list[Any]]]:
    texts = []
    image_batches = []
    for row in rows:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content(row)},
        ]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        image_batches.append(row_images(row))
    processor.tokenizer.padding_side = "left"
    encoded = processor(
        text=texts, images=image_batches, padding=True, return_tensors="pt"
    )
    return encoded, image_batches


def parse_answer(raw: str) -> tuple[str, str, bool]:
    raw = raw.strip()
    parsed: dict[str, Any] | None = None
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            candidate = json.loads(match.group(0))
            if isinstance(candidate, dict):
                parsed = candidate
        except json.JSONDecodeError:
            parsed = None
    decision = str(parsed.get("decision", "")) if parsed else ""
    answer = str(parsed.get("answer", raw)) if parsed else raw
    normalized = re.sub(r"\s+", "", answer.lower())
    abstained = decision.lower() == "abstain" or re.sub(r"\s+", "", ABSTAIN) in normalized
    return decision, answer, abstained


def positive_correct(answer: str, decision: str, abstained: bool, expected: str) -> bool:
    if abstained or decision.lower() == "abstain":
        return False
    normalized = re.sub(r"[\s\"'.,!?{}:]", "", answer.lower())
    expected_normalized = re.sub(r"[\s\"'.,!?{}:]", "", expected.lower())
    if expected_normalized in normalized:
        return True
    affirmative = normalized in {"예", "네", "yes", "그렇다", "발생함", "발생했다"}
    negative_markers = ("아니다", "않았다가아니다", "발생하지", "없다", "no")
    return affirmative and not any(marker in normalized for marker in negative_markers)


def evaluation_rows_and_adapter(output: Path, arm: str) -> tuple[list[dict[str, Any]], Path | None]:
    eval_rows = read_jsonl(output / "eval_core_val.jsonl") + read_jsonl(
        output / "eval_counterfactual_val.jsonl"
    )
    assert_no_forbidden_data(eval_rows, "evaluation")
    row_ids = [row["row_id"] for row in eval_rows]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("duplicate evaluation row ids")
    adapter: Path | None = None
    if arm == "tuned":
        selection = json.loads((output / "training/checkpoint_selection.json").read_text(encoding="utf-8"))
        adapter = Path(selection["best_checkpoint"])
        if not adapter.is_dir():
            raise FileNotFoundError(adapter)
    return eval_rows, adapter


def evaluate_vllm_command(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    output = args.output
    eval_rows, adapter = evaluation_rows_and_adapter(output, args.arm)
    if args.limit:
        eval_rows = eval_rows[: args.limit]
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        enforce_eager=True,
        enable_lora=adapter is not None,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 2},
        max_lora_rank=16,
        max_model_len=4096,
        tensor_parallel_size=1,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=32, logprobs=1)
    lora_request = (
        LoRARequest("tuned", 1, lora_path=str(adapter.resolve())) if adapter is not None else None
    )
    result_path = output / "evaluation" / f"{args.arm}.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if result_path.is_file():
        existing = {row["row_id"]: row for row in read_jsonl(result_path)}
    pending = [row for row in eval_rows if row["row_id"] not in existing]
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with result_path.open("a", encoding="utf-8", newline="\n") as handle:
        for offset in range(0, len(pending), args.batch_size):
            rows = pending[offset : offset + args.batch_size]
            prompts = []
            image_batches = []
            for row in rows:
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": user_content(row)},
                ]
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                images = row_images(row)
                image_batches.append(images)
                prompts.append({"prompt": prompt, "multi_modal_data": {"image": images}})
            generated = llm.generate(
                prompts,
                sampling_params=sampling,
                lora_request=lora_request,
                use_tqdm=False,
            )
            if len(generated) != len(rows):
                raise RuntimeError("vLLM output count mismatch")
            for row, request_output in zip(rows, generated, strict=True):
                if len(request_output.outputs) != 1:
                    raise RuntimeError(f"unexpected candidates: {row['row_id']}")
                candidate = request_output.outputs[0]
                raw = candidate.text.strip()
                token_count = len(candidate.token_ids)
                if token_count == 0:
                    raise RuntimeError(f"no generated tokens: {row['row_id']}")
                confidence = math.exp(float(candidate.cumulative_logprob) / token_count)
                decision, answer, abstained = parse_answer(raw)
                event_correct = (
                    positive_correct(answer, decision, abstained, row["target_answer"])
                    if row["target_type"] == "answer"
                    else None
                )
                correct = event_correct if row["target_type"] == "answer" else abstained
                result = row | {
                    "arm": args.arm,
                    "answer": answer,
                    "confidence": confidence,
                    "decision": decision,
                    "event_answer_correct": event_correct,
                    "raw_response": raw,
                    "target_correct": bool(correct),
                    "was_abstained": bool(abstained),
                }
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            for images in image_batches:
                for image in images:
                    image.close()
            completed = min(offset + len(rows), len(pending))
            print(f"{args.arm} {completed}/{len(pending)}", flush=True)

    final_rows = read_jsonl(result_path)
    if {row["row_id"] for row in final_rows} != {row["row_id"] for row in eval_rows}:
        raise RuntimeError(f"evaluation incomplete: {args.arm}")
    manifest = {
        "adapter": str(adapter) if adapter else None,
        "arm": args.arm,
        "backend": "vllm_offline",
        "data_rows": len(eval_rows),
        "environment": environment_payload(torch),
        "exploratory": True,
        "external_api_calls": 0,
        "model": MODEL_ID,
        "result_sha256": sha256_file(result_path),
    }
    write_json(output / "evaluation" / f"{args.arm}_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def evaluate_hf_command(args: argparse.Namespace) -> None:
    import torch

    output = args.output
    eval_rows, adapter = evaluation_rows_and_adapter(output, args.arm)
    if args.limit:
        eval_rows = eval_rows[: args.limit]
    row_ids = [row["row_id"] for row in eval_rows]
    model, processor = load_model_and_processor(adapter=adapter, trainable=False)
    model.eval()
    result_path = output / "evaluation" / f"{args.arm}.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if result_path.is_file():
        existing = {row["row_id"]: row for row in read_jsonl(result_path)}
    result_path.parent.mkdir(parents=True, exist_ok=True)

    pending = [row for row in eval_rows if row["row_id"] not in existing]
    with result_path.open("a", encoding="utf-8", newline="\n") as handle:
        for offset in range(0, len(pending), args.batch_size):
            rows = pending[offset : offset + args.batch_size]
            encoded, image_batches = generation_prompts(processor, rows)
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=32,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            prompt_length = encoded["input_ids"].shape[1]
            new_tokens = generated.sequences[:, prompt_length:]
            raw_responses = processor.batch_decode(new_tokens, skip_special_tokens=True)
            transition = model.compute_transition_scores(
                generated.sequences, generated.scores, normalize_logits=True
            )
            for batch_index, row in enumerate(rows):
                raw = raw_responses[batch_index].strip()
                token_ids = new_tokens[batch_index]
                eos_positions = (token_ids == processor.tokenizer.eos_token_id).nonzero()
                score_count = (
                    int(eos_positions[0]) + 1 if eos_positions.numel() else transition.shape[1]
                )
                if score_count == 0:
                    raise RuntimeError(f"no generated tokens: {row['row_id']}")
                confidence = float(
                    torch.exp(transition[batch_index, :score_count].mean()).detach().cpu()
                )
                decision, answer, abstained = parse_answer(raw)
                event_correct = (
                    positive_correct(answer, decision, abstained, row["target_answer"])
                    if row["target_type"] == "answer"
                    else None
                )
                correct = event_correct if row["target_type"] == "answer" else abstained
                result = row | {
                    "arm": args.arm,
                    "answer": answer,
                    "confidence": confidence,
                    "decision": decision,
                    "event_answer_correct": event_correct,
                    "raw_response": raw,
                    "target_correct": bool(correct),
                    "was_abstained": bool(abstained),
                }
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            for images in image_batches:
                for image in images:
                    image.close()
            completed = min(offset + len(rows), len(pending))
            if completed % 20 == 0 or completed == len(pending):
                print(f"{args.arm} {completed}/{len(pending)}", flush=True)

    final_rows = read_jsonl(result_path)
    if {row["row_id"] for row in final_rows} != set(row_ids):
        raise RuntimeError(f"evaluation incomplete: {args.arm}")
    manifest = {
        "adapter": str(adapter) if adapter else None,
        "arm": args.arm,
        "backend": "transformers_generate",
        "data_rows": len(eval_rows),
        "environment": environment_payload(torch),
        "exploratory": True,
        "model": MODEL_ID,
        "result_sha256": sha256_file(result_path),
    }
    write_json(output / "evaluation" / f"{args.arm}_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def evaluate_command(args: argparse.Namespace) -> None:
    if args.backend == "vllm":
        evaluate_vllm_command(args)
    elif args.backend == "hf":
        evaluate_hf_command(args)
    else:
        raise AssertionError(args.backend)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def metric_ci(values: list[float]) -> dict[str, float]:
    return {
        "ci95_high": percentile(values, 0.975),
        "ci95_low": percentile(values, 0.025),
        "mean": sum(values) / len(values),
    }


def mean(items: Iterable[bool | float]) -> float:
    values = [float(item) for item in items]
    return sum(values) / len(values) if values else float("nan")


def arm_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    core = [row for row in rows if row["row_id"].startswith("core::")]
    hard = [row for row in rows if row["source_type"] == "same_category_hard_negative"]
    cf = [row for row in rows if row["row_id"].startswith("counterfactual::")]
    eplus = [row for row in cf if row["counterfactual_role"] == "P+"]
    eminus = [row for row in cf if row["sufficiency_condition"] == "E-"]
    pminus = [row for row in cf if row["counterfactual_role"] == "P-"]
    prestored = {row["group_id"]: row for row in cf if row["counterfactual_role"] == "Pr"}
    recovery_candidates = [row for row in pminus if row["was_abstained"]]
    recovery = [prestored[row["group_id"]]["event_answer_correct"] for row in recovery_candidates]
    core_positive = [row for row in core if row["source_type"] == "native_positive"]
    core_negative = [row for row in core if row["source_type"] == "cross_category_negative"]
    return {
        "broad_core_eplus_accuracy": mean(row["target_correct"] for row in core_positive),
        "broad_core_eminus_abstention": mean(row["was_abstained"] for row in core_negative),
        "eplus_accuracy": mean(row["event_answer_correct"] for row in eplus),
        "eminus_abstention": mean(row["was_abstained"] for row in eminus),
        "pminus_assertion_rate": mean(not row["was_abstained"] for row in pminus),
        "pr_recovery_candidates": float(len(recovery_candidates)),
        "pr_recovery_rate": mean(recovery),
        "same_category_hard_negative_abstention": mean(row["was_abstained"] for row in hard),
    }


def gate_delta(rows: list[dict[str, Any]]) -> float:
    core = [
        row
        for row in rows
        if row["row_id"].startswith("core::")
        and row["source_type"] in {"native_positive", "cross_category_negative"}
    ]
    if len(core) % 2:
        raise RuntimeError("gate core must contain paired rows")
    selected = len(core) // 2
    confidence_ranked = sorted(core, key=lambda row: (-float(row["confidence"]), row["row_id"]))[:selected]
    oracle_ranked = sorted(
        core,
        key=lambda row: (
            0 if row["source_type"] == "native_positive" else 1,
            -float(row["confidence"]),
            row["row_id"],
        ),
    )[:selected]
    confidence_risk = mean(not row["target_correct"] for row in confidence_ranked)
    oracle_risk = mean(not row["target_correct"] for row in oracle_ranked)
    return confidence_risk - oracle_risk


def analyze_command(args: argparse.Namespace) -> None:
    output = args.output
    arms = {
        arm: read_jsonl(output / "evaluation" / f"{arm}.jsonl")
        for arm in ("zero_shot", "tuned")
    }
    ids_by_arm = {arm: {row["row_id"] for row in rows} for arm, rows in arms.items()}
    if ids_by_arm["zero_shot"] != ids_by_arm["tuned"]:
        raise RuntimeError("evaluation arms do not have identical row ids")

    metrics = {arm: arm_metrics(rows) for arm, rows in arms.items()}
    metrics["zero_shot"]["gate_delta_risk_at_50"] = gate_delta(arms["zero_shot"])
    metrics["tuned"]["gate_delta_risk_at_50"] = gate_delta(arms["tuned"])
    by_arm_group = {
        arm: defaultdict(list)
        for arm in arms
    }
    for arm, rows in arms.items():
        for row in rows:
            by_arm_group[arm][row["group_id"]].append(row)
    groups = sorted(set(by_arm_group["zero_shot"]) & set(by_arm_group["tuned"]))
    if not groups:
        raise RuntimeError("no paired incidents for bootstrap")

    rng = random.Random(SEED)
    bootstrap: dict[str, list[float]] = defaultdict(list)
    for _ in range(args.bootstrap):
        sampled = [rng.choice(groups) for _ in groups]
        sampled_arms: dict[str, list[dict[str, Any]]] = {}
        for arm in arms:
            sampled_rows = []
            for draw_index, group_id in enumerate(sampled):
                for row in by_arm_group[arm][group_id]:
                    sampled_rows.append(row | {"row_id": f"{row['row_id']}::draw{draw_index}"})
            sampled_arms[arm] = sampled_rows
        sampled_metrics = {arm: arm_metrics(rows) for arm, rows in sampled_arms.items()}
        for key in (
            "eplus_accuracy",
            "eminus_abstention",
            "pminus_assertion_rate",
            "pr_recovery_rate",
            "same_category_hard_negative_abstention",
        ):
            left = sampled_metrics["tuned"][key]
            right = sampled_metrics["zero_shot"][key]
            if not math.isnan(left) and not math.isnan(right):
                bootstrap[f"delta_{key}"].append(left - right)
        bootstrap["zero_shot_gate_delta_risk_at_50"].append(gate_delta(sampled_arms["zero_shot"]))
        bootstrap["tuned_gate_delta_risk_at_50"].append(gate_delta(sampled_arms["tuned"]))

        tuned_core_negative = [
            row
            for row in sampled_arms["tuned"]
            if row["source_type"] == "cross_category_negative"
        ]
        tuned_hard = [
            row
            for row in sampled_arms["tuned"]
            if row["source_type"] == "same_category_hard_negative"
        ]
        if tuned_hard:
            shortcut_gap = mean(row["was_abstained"] for row in tuned_core_negative) - mean(
                row["was_abstained"] for row in tuned_hard
            )
            bootstrap["tuned_shortcut_gap"].append(shortcut_gap)

    cis = {key: metric_ci(values) for key, values in bootstrap.items() if values}
    eplus_significant = cis["delta_eplus_accuracy"]["ci95_low"] > 0
    zero_gate = metrics["zero_shot"]["gate_delta_risk_at_50"]
    tuned_gate = metrics["tuned"]["gate_delta_risk_at_50"]
    gate_transition = zero_gate <= 0 < tuned_gate
    shortcut_gap = (
        metrics["tuned"]["broad_core_eminus_abstention"]
        - metrics["tuned"]["same_category_hard_negative_abstention"]
    )
    shortcut_flag = shortcut_gap > 0.10 and cis["tuned_shortcut_gap"]["ci95_low"] > 0

    if shortcut_flag:
        verdict = "지름길 학습 징후로 개선 불성립(탐색적)"
    elif eplus_significant and gate_transition:
        verdict = "개선 성립(탐색적)"
    elif eplus_significant:
        verdict = "부분 개선(탐색적)"
    else:
        verdict = "유의한 개선 없음(탐색적)"

    context_rows = [row for row in read_jsonl(CONTEXT_PATH) if row["input_condition"] == "full_native_crop"]
    context: dict[str, dict[str, float]] = {}
    for model in sorted({row["model"] for row in context_rows}):
        model_rows = [row for row in context_rows if row["model"] == model]
        context[model] = {
            "eplus_accuracy": sum(row["eplus_accuracy"] * row["eplus_n"] for row in model_rows)
            / sum(row["eplus_n"] for row in model_rows),
            "pminus_assertion_rate": sum(
                row["pminus_assertion_rate"] * row["pminus_n"] for row in model_rows
            )
            / sum(row["pminus_n"] for row in model_rows),
            "pr_recovery_rate": sum(
                row["pr_recovery_rate"] * row["pr_recovery_candidate_n"] for row in model_rows
            )
            / sum(row["pr_recovery_candidate_n"] for row in model_rows),
        }

    result = {
        "bootstrap": {"ci": "paired incident percentile 95%", "n": args.bootstrap, "seed": SEED},
        "confidence": "geometric mean generated-token probability",
        "context_only_existing_large_zero_shot": context,
        "decision": {
            "eplus_significant": eplus_significant,
            "gate_transition": gate_transition,
            "shortcut_flag": shortcut_flag,
            "shortcut_gap": shortcut_gap,
            "verdict_ko": verdict,
        },
        "exploratory": True,
        "metric_cis": cis,
        "metrics": metrics,
    }
    evaluation_dir = output / "evaluation"
    write_json(evaluation_dir / "final_evaluation.json", result)

    table_rows = []
    for arm in ("zero_shot", "tuned"):
        table_rows.append({"arm": arm, **metrics[arm]})
    with (evaluation_dir / "comparison_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    lines = [
        "답변자 정렬 LoRA 탐색 트랙 최종 판정",
        "",
        f"판정: {verdict}",
        f"E+ 유의 상승: {'예' if eplus_significant else '아니오'}",
        f"게이트 ΔRisk@50 양수 전환: {'예' if gate_transition else '아니오'} "
        f"(zero-shot={zero_gate:.4f}, tuned={tuned_gate:.4f})",
        f"동일범주 지름길 징후: {'있음' if shortcut_flag else '없음'} (격차={shortcut_gap:.4f})",
        "",
        "비교표 (개발 30% 사건 홀드아웃, 탐색적)",
    ]
    for arm in ("zero_shot", "tuned"):
        item = metrics[arm]
        lines.append(
            f"- {arm}: E+ 정답률={item['eplus_accuracy']:.4f}, "
            f"E- 보류율={item['eminus_abstention']:.4f}, "
            f"P- 단정률={item['pminus_assertion_rate']:.4f}, "
            f"Pr 회복률={item['pr_recovery_rate']:.4f}, "
            f"동일범주 음성 보류 정확도={item['same_category_hard_negative_abstention']:.4f}"
        )
    lines += [
        "",
        "기존 3개 대형모델 수치는 맥락용 zero-shot 수치이며 우월성 비교 주장이 아니다.",
        "준확증 B/C, heldout-D, reserve는 사용하지 않았고 외부 API 호출은 0회다.",
    ]
    (evaluation_dir / "final_report_ko.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build")
    train = subparsers.add_parser("train")
    train.add_argument("--epochs", type=int, default=3, choices=(1, 2, 3))
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--validation-batch-size", type=int, default=2)
    train.add_argument("--gradient-accumulation", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--early-stopping-patience", type=int, default=1)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--arm", choices=("zero_shot", "tuned"), required=True)
    evaluate.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--limit", type=int, default=0)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "analyze":
        analyze_command(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
