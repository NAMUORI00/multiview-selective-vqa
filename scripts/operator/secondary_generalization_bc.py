"""Pre-registered exposed-history secondary generalization pipeline for B/C.

Stage 1 may run ``build``, ``selector`` and ``smoke`` only.  ``evaluate`` has an
explicit confirmation token because it performs B/C model inference.  All
analysis is incident-paired and failures are fatal; the only resume mechanism
is append-only completion of an identical evaluation shard.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "secondary_generalization_bc_20260712"
QUASI = ROOT / "outputs" / "dissection_quasi_confirmatory_bc_20260710"
DEV = ROOT / "outputs" / "dissection_empirical_package_20260710"
DEV_REAL = ROOT / "outputs" / "integrated_core_sufficiency_rebuild_20260709"
LORA_DEV = ROOT / "outputs" / "answerer_alignment_lora_exploratory_20260710"
INPUT = QUASI / "step4_answerer_input_private.jsonl"
SEED = 20260712
BOOTSTRAP = 4000
EXECUTION_TOKEN = "SECONDARY_GENERALIZATION_BC_20260712"
ABSTAIN = "주어진 근거로는 판단 불가"
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SYSTEM_PROMPT = (
    "당신은 제공된 시각 근거만 사용하는 안전사건 답변자다. 외부 지식이나 추측을 사용하지 않는다. "
    "근거가 질문의 사건을 직접 뒷받침하면 짧은 사실 문장으로 답하고, 직접 판단할 수 없으면 "
    f"정확히 '{ABSTAIN}'라고 답한다. 반드시 JSON 한 줄만 출력한다. "
    '형식은 {"decision":"answer 또는 abstain","answer":"짧은 답변"}이다.'
)
FRESH_DINO_KEYS = (
    "fresh_dino_total_regions",
    "fresh_dino_person_regions",
    "fresh_dino_max_conf",
    "fresh_dino_max_person_conf",
    "fresh_dino_region_area_sum",
)
LARGE_DEV_FILES = (
    DEV_REAL / "answerer_vqa" / "gemma4_31b" / "full_native_crop.jsonl",
    DEV_REAL / "answerer_vqa" / "internvl3_38b" / "full_native_crop.jsonl",
    DEV_REAL / "answerer_vqa" / "qwen3vl_32b" / "full_native_crop.jsonl",
)
LARGE_BC_FILES = (
    QUASI / "answerer_results" / "gemma3_27b" / "full_native_crop.jsonl",
    QUASI / "answerer_results" / "internvl3_38b" / "full_native_crop.jsonl",
    QUASI / "answerer_results" / "qwen3vl_32b" / "full_native_crop.jsonl",
)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL: {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_ids(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(set(values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_fields(row: dict[str, Any], fields: Iterable[str], label: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"missing fields in {label}: {missing}")


def assert_unique(rows: list[dict[str, Any]], key: str, label: str) -> None:
    values = [str(row[key]) for row in rows]
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate {key} in {label}: {duplicates[:5]}")


def stable_score(text: str, seed: int = SEED) -> float:
    raw = hashlib.sha256(f"{seed}:{text}".encode()).digest()[:8]
    return int.from_bytes(raw, "big") / 2**64


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def ci(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "ci95_low": percentile(values, 0.025),
        "ci95_high": percentile(values, 0.975),
    }


def logit(value: float) -> float:
    clipped = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def build_command(args: argparse.Namespace) -> None:
    output = args.output
    rows = read_jsonl(INPUT)
    required = (
        "row_id",
        "group_id",
        "category",
        "counterfactual_role",
        "operator",
        "label",
        "question",
        "event_question",
        "full_path",
        "crop_path",
        "image_width",
        "image_height",
    )
    for index, row in enumerate(rows):
        require_fields(row, required, f"B/C input row {index}")
    assert_unique(rows, "row_id", "B/C input")
    events = sorted({str(row["group_id"]) for row in rows})
    roles = Counter(str(row["counterfactual_role"]) for row in rows)
    if len(rows) != 576 or len(events) != 96:
        raise RuntimeError(f"unexpected B/C shape rows={len(rows)} events={len(events)}")
    if roles != Counter({"P+": 96, "P-": 96, "P0": 288, "Pr": 96}):
        raise RuntimeError(f"unexpected counterfactual roles: {roles}")
    expected_hash = read_json(output / "prereg_design.json")["event_set"]
    if sha256_ids(events) != expected_hash["event_ids_sha256_lf_sorted"]:
        raise RuntimeError("event ID hash differs from preregistration")
    if sha256_file(INPUT) != expected_hash["source_input_sha256"]:
        raise RuntimeError("B/C input hash differs from preregistration")

    dispositions = read_jsonl(QUASI / "step1_event_disposition_private.jsonl")
    disposition_by_id = {str(row["sample_id"]): row for row in dispositions}
    system_split_counts = Counter(str(disposition_by_id[event]["legacy_split"]) for event in events)
    routing_event_ids = sorted(
        {str(row["group_id"]) for row in read_jsonl(QUASI / "step1_cross_core_benchmark.jsonl")}
    )
    routing_split_counts = Counter(
        str(disposition_by_id[event]["legacy_split"]) for event in routing_event_ids
    )
    if system_split_counts != Counter({"heldout-C": 96}):
        raise RuntimeError(f"unexpected system split composition: {system_split_counts}")
    if routing_split_counts != Counter({"heldout-C": 798}):
        raise RuntimeError(f"unexpected routing split composition: {routing_split_counts}")

    all_paths: dict[str, list[Path]] = {}
    for field in ("full_path", "crop_path"):
        paths = sorted({ROOT / str(row[field]) for row in rows})
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing {field}: {missing[:5]}")
        all_paths[field] = paths

    event_to_shard = {event: index % 3 for index, event in enumerate(events)}
    built = [dict(row, shard_index=event_to_shard[str(row["group_id"])]) for row in rows]
    build_dir = output / "build"
    write_jsonl(build_dir / "bc_input.jsonl", built)
    for shard_index in range(3):
        shard = [row for row in built if row["shard_index"] == shard_index]
        if len(shard) != 192 or len({row["group_id"] for row in shard}) != 32:
            raise RuntimeError(f"unexpected shard {shard_index} shape")
        write_jsonl(build_dir / f"bc_input_shard{shard_index}.jsonl", shard)

    signal_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["counterfactual_role"] == "Pr":
            continue
        signal_rows.append(
            {
                **row,
                "source_sample_id": str(row["row_id"]),
                "source_row_id": str(row["row_id"]),
            }
        )
    if len(signal_rows) != 480:
        raise RuntimeError("availability signal input must contain 480 non-restored rows")
    write_jsonl(build_dir / "availability_signal_input.jsonl", signal_rows)

    large_inventory: dict[str, Any] = {}
    for path in LARGE_BC_FILES:
        model_rows = read_jsonl(path)
        assert_unique(model_rows, "row_id", path.name)
        missing_confidence = sum(row.get("confidence") is None for row in model_rows)
        parse_errors = sum(row.get("parse_error") is not None for row in model_rows)
        if len(model_rows) != 576 or len({row["group_id"] for row in model_rows}) != 96:
            raise RuntimeError(f"incomplete large-model result: {path}")
        if missing_confidence or parse_errors:
            raise RuntimeError(f"invalid large-model result: {path}")
        large_inventory[path.parent.name] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "rows": len(model_rows),
            "events": len({row["group_id"] for row in model_rows}),
            "confidence_present": len(model_rows),
            "confidence_missing": 0,
            "parse_errors": 0,
        }

    inventory = {
        "schema_version": "secondary_generalization_bc_inventory_v1",
        "large_model_answers": large_inventory,
        "counterfactual_exposure": {
            "event_ids": events,
            "event_ids_sha256_lf_sorted": sha256_ids(events),
            "events": len(events),
            "rows": len(rows),
            "pminus_event_ids": sorted(
                {str(row["group_id"]) for row in rows if row["counterfactual_role"] == "P-"}
            ),
            "pminus_rows": roles["P-"],
            "legacy_split_counts": {"heldout-B": 0, "heldout-C": 96},
        },
        "historical_source_pool": {
            "legacy_split_counts": dict(
                sorted(Counter(str(row["legacy_split"]) for row in dispositions).items())
            ),
            "eligible_routing_core_events": len(routing_event_ids),
            "eligible_routing_core_legacy_split_counts": {
                "heldout-B": 0,
                "heldout-C": 798,
            },
            "scope_warning": "the frozen eligible system and routing surfaces are heldout-C only",
        },
        "images": {
            field: {
                "unique_files": len(paths),
                "bytes": sum(path.stat().st_size for path in paths),
                "local_missing": 0,
            }
            for field, paths in all_paths.items()
        },
        "remote_gpu_host": {
            "root": "/workspace/MV-EviRAG",
            "checked_at": "2026-07-12T00:00:00+09:00",
            "existing_files": 0,
            "missing_files": 960,
            "bytes": 0,
            "transfer_required": True,
            "deterministic_reextraction_required": False,
            "transfer_plan": "send the 960 existing local files (359535087 bytes) by an LF file-list tar|ssh stream; verify count and du before and after",
        },
        "server_checkpoint": {
            "epoch": 3,
            "adapter_config_sha256": "ff273b37648189c1a526cbc289318478c48fd20e1a7a9d8664f313dc6386887f",
            "adapter_model_sha256": "3490c8851463441a9453e3de52411200cbbff8ae99b231b3b4c0f817a45d7edf",
        },
    }
    write_json(output / "inventory.json", inventory)
    docker_plan = docker_launch_plan()
    write_json(output / "docker_launch_plan.json", docker_plan)
    write_prereg_manifest(output)
    print(json.dumps({"status": "BUILD_READY_NO_INFERENCE", "rows": 576, "events": 96}, ensure_ascii=False))


def docker_launch_plan() -> dict[str, Any]:
    commands = []
    for gpu in range(3):
        port = 8100 + gpu
        commands.append(
            {
                "gpu": gpu,
                "port": port,
                "command": (
                    f"docker run --rm --name mvevirag-secondary-bc-gpu{gpu} --ipc host --shm-size 8g "
                    f"--gpus '\"device={gpu}\"' -p {port}:8000 "
                    "-v /workspace/MV-EviRAG:/workspace/MV-EviRAG "
                    "-v /cache/huggingface:/cache/huggingface "
                    "-e HF_HUB_OFFLINE=1 -e HF_HOME=/cache/huggingface "
                    "vllm/vllm-openai:latest --model Qwen/Qwen3-VL-8B-Instruct "
                    "--served-model-name Qwen/Qwen3-VL-8B-Instruct --dtype bfloat16 "
                    "--max-model-len 4096 --gpu-memory-utilization 0.92 --enforce-eager "
                    "--limit-mm-per-prompt '{\"image\":2}' --enable-lora "
                    "--lora-modules aligned=/workspace/MV-EviRAG/outputs/"
                    "answerer_alignment_lora_exploratory_20260710/training/epoch-3"
                ),
            }
        )
    return {
        "image": "vllm/vllm-openai:latest",
        "observed_image_id": "sha256:4091d5593f77189cafa013299e927ee5cb979e7c38330cbd5e30ea54c8fea4fd",
        "commands": commands,
        "worker_pattern": (
            "python scripts/operator/secondary_generalization_bc.py evaluate --arm {zero_shot_8b|aligned_8b} "
            "--shard-index {0|1|2} --base-url http://127.0.0.1:{8100|8101|8102}/v1 "
            f"--confirm-bc-inference {EXECUTION_TOKEN}"
        ),
    }


def write_prereg_manifest(output: Path) -> None:
    names = ("prereg_design.json", "arm_specs.json", "analysis_plan.json")
    artifacts = {name: sha256_file(output / name) for name in names}
    write_json(
        output / "prereg_manifest.json",
        {
            "schema_version": "secondary_generalization_bc_prereg_manifest_v1",
            "artifacts": artifacts,
            "combined_sha256": sha256_json(artifacts),
        },
    )


def image_data_url(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"unsupported image type: {path}")
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def call_vllm(base_url: str, model: str, row: dict[str, Any]) -> tuple[str, float]:
    content = [
        {"type": "text", "text": "전체 프레임:"},
        {"type": "image_url", "image_url": {"url": image_data_url(ROOT / row["full_path"])}},
        {"type": "text", "text": "원해상도 판단영역:"},
        {"type": "image_url", "image_url": {"url": image_data_url(ROOT / row["crop_path"])}},
        {"type": "text", "text": f"질문: {row['event_question']}"},
    ]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": 128,
        "seed": 20260710,
        "logprobs": True,
        "top_logprobs": 1,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"vLLM request failed for {row['row_id']}") from exc
    try:
        choice = result["choices"][0]
        raw = str(choice["message"]["content"])
        token_rows = choice["logprobs"]["content"]
        logprobs = [finite_float(token["logprob"], "token logprob") for token in token_rows]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"malformed vLLM response for {row['row_id']}") from exc
    if not logprobs:
        raise RuntimeError(f"missing generated-token logprobs for {row['row_id']}")
    return raw, math.exp(sum(logprobs) / len(logprobs))


def parse_answer(raw: str) -> tuple[str, str, bool]:
    match = re.search(r"\{.*\}", raw.strip(), flags=re.DOTALL)
    if not match:
        raise ValueError(f"answer is not a JSON object: {raw[:120]!r}")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"answer JSON parse failed: {raw[:120]!r}") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"decision", "answer"}:
        raise ValueError(f"answer schema mismatch: {parsed!r}")
    decision = str(parsed["decision"]).strip().lower()
    answer = str(parsed["answer"]).strip()
    if decision not in {"answer", "abstain"} or not answer:
        raise ValueError(f"invalid decision or empty answer: {parsed!r}")
    normalized = re.sub(r"\s+", "", answer.lower())
    abstained = decision == "abstain" or re.sub(r"\s+", "", ABSTAIN) in normalized
    return decision, answer, abstained


def positive_correct(answer: str, decision: str, abstained: bool) -> bool:
    if abstained or decision == "abstain":
        return False
    normalized = re.sub(r"[\s\"'.,!?{}:]", "", answer.lower())
    affirmative = normalized in {"예", "네", "yes", "그렇다", "발생함", "발생했다"}
    event_statement = any(
        marker in normalized
        for marker in ("침입", "넘어", "착용하지않", "미착용", "두사람", "2인", "함께탑승")
    )
    negative = any(marker in normalized for marker in ("아니다", "발생하지", "없다", "no"))
    return (affirmative or event_statement) and not negative


def evaluation_result(row: dict[str, Any], arm: str, raw: str, confidence: float) -> dict[str, Any]:
    decision, answer, abstained = parse_answer(raw)
    label = int(row["label"])
    success = positive_correct(answer, decision, abstained) if label == 1 else abstained
    return {
        **row,
        "arm": arm,
        "raw_response": raw,
        "decision": decision,
        "answer": answer,
        "abstained": abstained,
        "confidence": confidence,
        "answer_success": bool(success),
        "event_answer_correct": bool(success) if label == 1 else False,
        "prompt_contract_sha256": "a85f987456e9633eaa7abd2cae1f7216ed313b8cfbad225f2b27d7968bfaafa7",
        "decoding_sha256": "7e6bb0b51a49fa43e469ef58b2556335d6092a33bfef7d9f6b28127778061fb1",
    }


def evaluate_command(args: argparse.Namespace) -> None:
    if args.confirm_bc_inference != EXECUTION_TOKEN:
        raise RuntimeError("B/C inference is locked; the exact confirmation token is required")
    shard_path = args.output / "build" / f"bc_input_shard{args.shard_index}.jsonl"
    rows = read_jsonl(shard_path)
    model = BASE_MODEL if args.arm == "zero_shot_8b" else "aligned"
    result_path = args.output / "evaluation" / args.arm / f"shard{args.shard_index}.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if result_path.exists():
        prior = read_jsonl(result_path)
        assert_unique(prior, "row_id", str(result_path))
        for item in prior:
            if item.get("arm") != args.arm:
                raise RuntimeError("checkpoint arm mismatch")
            completed[str(item["row_id"])] = item
    expected_ids = {str(row["row_id"]) for row in rows}
    if set(completed) - expected_ids:
        raise RuntimeError("checkpoint contains stale row IDs")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with result_path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            row_id = str(row["row_id"])
            if row_id in completed:
                continue
            raw, confidence = call_vllm(args.base_url, model, row)
            result = evaluation_result(row, args.arm, raw, confidence)
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            completed[row_id] = result
            print(f"{args.arm} shard={args.shard_index} {len(completed)}/{len(rows)}", flush=True)
    if set(completed) != expected_ids:
        raise RuntimeError("evaluation shard is incomplete")
    manifest = {
        "arm": args.arm,
        "shard_index": args.shard_index,
        "rows": len(rows),
        "events": len({row["group_id"] for row in rows}),
        "result_sha256": sha256_file(result_path),
        "elapsed_seconds_this_invocation": time.time() - started,
        "resume_only": True,
        "external_api_calls": 0,
    }
    write_json(result_path.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def response_features(row: dict[str, Any]) -> list[float]:
    answer = str(row.get("answer", ""))
    evidence = str(row.get("evidence", ""))
    raw = str(row.get("raw_response", ""))
    return [
        finite_float(row.get("confidence"), "confidence"),
        math.log1p(len(answer)),
        math.log1p(len(answer.split())),
        float(bool(row.get("abstained"))),
        math.log1p(len(evidence)),
        math.log1p(len(raw)),
    ]


def fit_logistic(features: list[list[float]], labels: list[int], *, c_value: float) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[0] != y.size or set(y.tolist()) != {0, 1}:
        raise ValueError("invalid logistic training data")
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
        random_state=20260710,
    ).fit(scaler.transform(x), y)
    return {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "training_rows": int(y.size),
        "positive_rows": int(y.sum()),
    }


def predict_logistic(model: dict[str, Any], features: list[float]) -> float:
    if not (len(features) == len(model["scaler_mean"]) == len(model["scaler_scale"]) == len(model["coef"])):
        raise ValueError("frozen logistic feature dimension mismatch")
    z = finite_float(model["intercept"], "intercept")
    for value, mean, scale, weight in zip(
        features, model["scaler_mean"], model["scaler_scale"], model["coef"], strict=True
    ):
        scale_value = finite_float(scale, "scale")
        if scale_value == 0:
            raise ValueError("zero frozen scale")
        z += finite_float(weight, "weight") * (finite_float(value, "feature") - float(mean)) / scale_value
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def selector_command(args: argparse.Namespace) -> None:
    b3_rows: list[dict[str, Any]] = []
    for path in LARGE_DEV_FILES:
        if any(token in str(path).lower() for token in ("heldout", "reserve", "quasi_confirmatory")):
            raise RuntimeError(f"forbidden B3 training source: {path}")
        rows = read_jsonl(path)
        if len(rows) != 576:
            raise RuntimeError(f"unexpected B3 dev rows: {path}")
        b3_rows.extend(rows)
    b3 = fit_logistic(
        [response_features(row) for row in b3_rows],
        [int(bool(row["answer_success"])) for row in b3_rows],
        c_value=1.0,
    )
    b3.update(
        {
            "feature_names": [
                "confidence",
                "log1p_answer_chars",
                "log1p_answer_whitespace_tokens",
                "abstained",
                "log1p_evidence_chars",
                "log1p_raw_response_chars",
            ],
            "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in LARGE_DEV_FILES},
            "forbidden_features": ["model_id", "category", "operator", "counterfactual_role", "event_id"],
        }
    )

    tuned = read_jsonl(LORA_DEV / "evaluation" / "tuned.jsonl")
    availability_rows = read_jsonl(DEV_REAL / "real_step1_selector_predictions.jsonl")
    availability = {str(row["row_id"]): row for row in availability_rows}
    fusion_features: list[list[float]] = []
    fusion_labels: list[int] = []
    fusion_groups: set[str] = set()
    for row in tuned:
        row_id = re.sub(r"^(core|counterfactual)::", "", str(row["row_id"]))
        signal = availability.get(row_id)
        if signal is None:
            continue
        fusion_features.append(
            [
                logit(finite_float(row["confidence"], "confidence")),
                logit(finite_float(signal["real_selector_score"], "availability")),
            ]
        )
        fusion_labels.append(int(bool(row["target_correct"])))
        fusion_groups.add(str(row["group_id"]))
    if len(fusion_features) != 565 or len(fusion_groups) != 113:
        raise RuntimeError(
            f"unexpected fusion dev join rows={len(fusion_features)} groups={len(fusion_groups)}"
        )
    fusion = fit_logistic(fusion_features, fusion_labels, c_value=1.0)
    fusion.update(
        {
            "feature_names": ["logit_confidence", "logit_availability"],
            "training_groups": len(fusion_groups),
            "source_sha256": {
                str((LORA_DEV / "evaluation" / "tuned.jsonl").relative_to(ROOT)): sha256_file(
                    LORA_DEV / "evaluation" / "tuned.jsonl"
                ),
                str((DEV_REAL / "real_step1_selector_predictions.jsonl").relative_to(ROOT)): sha256_file(
                    DEV_REAL / "real_step1_selector_predictions.jsonl"
                ),
            },
        }
    )
    availability_model = fit_availability_transfer_model()
    payload = {
        "schema_version": "secondary_generalization_bc_selector_models_v1",
        "fit_surface": "development_only",
        "bc_rows_used_for_fit": 0,
        "b3": b3,
        "fusion": fusion,
        "availability_transfer": availability_model,
    }
    write_json(args.output / "selector_models.json", payload)
    write_prereg_manifest(args.output)
    print(
        json.dumps(
            {
                "status": "SELECTORS_FROZEN_DEV_ONLY",
                "b3_rows": b3["training_rows"],
                "fusion_rows": fusion["training_rows"],
                "availability_rows": availability_model["training_rows"],
            },
            ensure_ascii=False,
        )
    )


def feature_arrays(
    rows: list[dict[str, Any]],
    positives: list[dict[str, Any]],
    feature_path: Path,
    dino_path: Path,
) -> dict[str, Any]:
    import numpy as np

    cache = np.load(feature_path, allow_pickle=True)
    source_ids = [str(value) for value in cache["source_ids"].tolist()]
    image_by_source = {
        source_id: cache["image"][index].astype(np.float32) for index, source_id in enumerate(source_ids)
    }
    query_by_text = {
        str(text): cache["query"][index].astype(np.float32)
        for index, text in enumerate(cache["question_texts"].tolist())
    }
    dino_raw = {str(row["row_id"]): row for row in read_jsonl(dino_path)}
    positive_by_source = {str(row["source_sample_id"]): row for row in positives}
    if set(source_ids) != set(positive_by_source):
        raise RuntimeError("availability feature source IDs do not match positives")
    image = np.stack([image_by_source[str(row["source_sample_id"])] for row in rows]).astype(np.float32)
    query = np.stack([query_by_text[str(row["question"])] for row in rows]).astype(np.float32)
    dino = np.asarray(
        [
            [
                float(dino_raw[str(positive_by_source[str(row["source_sample_id"])]["row_id"])][key])
                for key in FRESH_DINO_KEYS
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    return {"image": image, "query": query, "dino": dino}


def dense_features(arrays: dict[str, Any]) -> Any:
    import numpy as np

    return np.concatenate(
        [
            arrays["image"],
            arrays["query"],
            arrays["image"] * arrays["query"],
            np.abs(arrays["image"] - arrays["query"]),
            arrays["dino"],
        ],
        axis=1,
    ).astype(np.float32)


def fit_availability_transfer_model() -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rows = read_jsonl(DEV / "step1_cross_core_benchmark.jsonl")
    positives = read_jsonl(DEV / "step1_positive_crop_index_private.jsonl")
    arrays = feature_arrays(
        rows,
        positives,
        DEV / "step1_siglip2_features.npz",
        DEV / "step1_fresh_dino_features.jsonl",
    )
    x = dense_features(arrays)
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=0.08,
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
        random_state=20260710,
    ).fit(scaler.transform(x), y)
    return {
        "feature_definition": "SigLIP2 image + raw question + product + absolute difference + fresh DINO scalars",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "training_rows": len(rows),
        "positive_rows": int(y.sum()),
        "C": 0.08,
        "source_sha256": {
            "step1_cross_core_benchmark.jsonl": sha256_file(DEV / "step1_cross_core_benchmark.jsonl"),
            "step1_positive_crop_index_private.jsonl": sha256_file(
                DEV / "step1_positive_crop_index_private.jsonl"
            ),
            "step1_siglip2_features.npz": sha256_file(DEV / "step1_siglip2_features.npz"),
            "step1_fresh_dino_features.jsonl": sha256_file(DEV / "step1_fresh_dino_features.jsonl"),
        },
    }


def apply_availability_features(output: Path, feature_dir: Path) -> list[dict[str, Any]]:
    import numpy as np

    model = read_json(output / "selector_models.json")["availability_transfer"]
    rows = read_jsonl(output / "build" / "availability_signal_input.jsonl")
    cache = np.load(feature_dir / "step2_siglip2_features.npz", allow_pickle=True)
    source_ids = [str(value) for value in cache["source_ids"].tolist()]
    if source_ids != [str(row["row_id"]) for row in rows]:
        raise RuntimeError("availability SigLIP checkpoint row order mismatch")
    query_by_text = {
        str(text): cache["query"][index].astype(np.float32)
        for index, text in enumerate(cache["question_texts"].tolist())
    }
    dino_by_id = {
        str(row["row_id"]): row for row in read_jsonl(feature_dir / "step2_fresh_dino_features.jsonl")
    }
    image = cache["image"].astype(np.float32)
    query = np.stack([query_by_text[str(row["question"])] for row in rows]).astype(np.float32)
    dino = np.asarray(
        [[finite_float(dino_by_id[str(row["row_id"])][key], key) for key in FRESH_DINO_KEYS] for row in rows],
        dtype=np.float32,
    )
    dense = dense_features({"image": image, "query": query, "dino": dino})
    scores = [predict_logistic(model, vector.tolist()) for vector in dense]
    semantic = []
    for image_vector, query_vector in zip(image, query, strict=True):
        denominator = float(np.linalg.norm(image_vector) * np.linalg.norm(query_vector))
        if denominator <= 0:
            raise RuntimeError("zero norm in semantic-relevance proxy")
        semantic.append(float(np.dot(image_vector, query_vector) / denominator))
    result = [
        {
            "row_id": str(row["row_id"]),
            "group_id": str(row["group_id"]),
            "category": str(row["category"]),
            "availability_score": scores[index],
            "semantic_relevance_proxy": semantic[index],
        }
        for index, row in enumerate(rows)
    ]
    by_id = {row["row_id"]: row for row in result}
    for row in read_jsonl(INPUT):
        if row["counterfactual_role"] != "Pr":
            continue
        positive_id = f"{row['group_id']}__{row['category']}__Epos"
        source = by_id.get(positive_id)
        if source is None:
            raise RuntimeError(f"missing positive score for restored row: {row['row_id']}")
        result.append({**source, "row_id": str(row["row_id"])})
    assert_unique(result, "row_id", "availability scores")
    if len(result) != 576:
        raise RuntimeError("availability scores must contain 576 rows")
    write_jsonl(output / "availability_scores.jsonl", result)
    return result


def routing_rows_real() -> list[dict[str, Any]]:
    import numpy as np

    rows = read_jsonl(QUASI / "step1_cross_core_benchmark.jsonl")
    selector = {str(row["row_id"]): row for row in read_jsonl(QUASI / "step3_selector_predictions.jsonl")}
    cache = np.load(QUASI / "step2_siglip2_features.npz", allow_pickle=True)
    source_ids = [str(value) for value in cache["source_ids"].tolist()]
    image_by_source = {source: cache["image"][index] for index, source in enumerate(source_ids)}
    query_by_text = {
        str(text): cache["query"][index] for index, text in enumerate(cache["question_texts"].tolist())
    }
    dev_rates_rows = read_jsonl(DEV / "step1_cross_core_benchmark.jsonl")
    dev_counts: dict[str, list[int]] = defaultdict(list)
    for row in dev_rates_rows:
        dev_counts[str(row["question_category"])].append(int(row["label"]))
    category_rates = {
        category: (sum(labels) + 1.0) / (len(labels) + 2.0) for category, labels in dev_counts.items()
    }
    result = []
    for row in rows:
        row_id = str(row["row_id"])
        signal = selector.get(row_id)
        if signal is None:
            raise RuntimeError(f"missing routing selector score: {row_id}")
        image = image_by_source[str(row["source_sample_id"])]
        query = query_by_text[str(row["question"])]
        denominator = float(np.linalg.norm(image) * np.linalg.norm(query))
        if denominator <= 0:
            raise RuntimeError(f"zero semantic norm: {row_id}")
        label = int(row["label"])
        result.append(
            {
                "row_id": row_id,
                "group_id": str(row["group_id"]),
                "category": str(row["question_category"]),
                "label": label,
                "grounded_correct": label,
                "random_score": stable_score(row_id),
                "category_stratified_random_score": stable_score(row_id),
                "category_only_score": category_rates[str(row["question_category"])],
                "semantic_relevance_proxy": float(np.dot(image, query) / denominator),
                "availability_score": finite_float(signal["dev_transfer_score"], "dev_transfer_score"),
                "true_label_score": float(label),
            }
        )
    assert_unique(result, "row_id", "routing rows")
    return result


def routing_rows_smoke() -> list[dict[str, Any]]:
    rows = read_jsonl(DEV_REAL / "real_step1_selector_predictions.jsonl")
    labels_by_category: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        labels_by_category[str(row["category"])].append(int(row["label"]))
    rates = {
        category: (sum(labels) + 1.0) / (len(labels) + 2.0) for category, labels in labels_by_category.items()
    }
    result = []
    for row in rows:
        row_id = str(row["row_id"])
        label = int(row["label"])
        result.append(
            {
                "row_id": row_id,
                "group_id": str(row["group_id"]),
                "category": str(row["category"]),
                "label": label,
                "grounded_correct": label,
                "random_score": stable_score(row_id),
                "category_stratified_random_score": stable_score(row_id),
                "category_only_score": rates[str(row["category"])],
                "semantic_relevance_proxy": finite_float(
                    row["real_shuffled_question_score"], "smoke semantic surrogate"
                ),
                "availability_score": finite_float(row["real_selector_score"], "availability"),
                "true_label_score": float(label),
            }
        )
    return result


POLICY_KEYS = {
    "random": "random_score",
    "category_stratified_random": "category_stratified_random_score",
    "category_only": "category_only_score",
    "siglip2_semantic_relevance_proxy": "semantic_relevance_proxy",
    "availability": "availability_score",
    "true_label_upper_bound": "true_label_score",
}


def selected_rows(
    rows: list[dict[str, Any]], score_key: str, coverage: float, scope: str
) -> list[dict[str, Any]]:
    if not 0.0 <= coverage <= 1.0:
        raise ValueError(f"invalid coverage: {coverage}")
    if coverage == 0 or not rows:
        return []
    if scope == "global":
        count = min(len(rows), math.ceil(coverage * len(rows)))
        return sorted(rows, key=lambda row: (-finite_float(row[score_key], score_key), str(row["row_id"])))[
            :count
        ]
    if scope != "within_category":
        raise ValueError(f"invalid ranking scope: {scope}")
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    selected: list[dict[str, Any]] = []
    for category in sorted(by_category):
        category_rows = by_category[category]
        count = min(len(category_rows), math.ceil(coverage * len(category_rows)))
        selected.extend(
            sorted(
                category_rows,
                key=lambda row: (-finite_float(row[score_key], score_key), str(row["row_id"])),
            )[:count]
        )
    return selected


def metric_at(rows: list[dict[str, Any]], score_key: str, coverage: float, scope: str) -> dict[str, Any]:
    selected = selected_rows(rows, score_key, coverage, scope)
    positives = sum(int(row["label"]) for row in rows)
    if positives <= 0:
        raise ValueError("routing metric needs positive rows")
    selected_positive = sum(int(row["label"]) for row in selected)
    grounded = sum(int(row["grounded_correct"]) for row in selected)
    return {
        "selected_calls": len(selected),
        "answerability_recall": selected_positive / positives,
        "call_waste_rate": (
            sum(1 - int(row["label"]) for row in selected) / len(selected) if selected else 0.0
        ),
        "evidence_supported_correct_yield": grounded,
        "calls_per_correct_answer": len(selected) / grounded if grounded else None,
    }


def routing_auc(rows: list[dict[str, Any]], score_key: str, scope: str) -> float:
    positives = sum(int(row["label"]) for row in rows)
    if positives <= 0:
        raise ValueError("routing AUC needs positive rows")
    fractions = {0.0, 0.5}
    ordered_groups: list[list[dict[str, Any]]] = []
    if scope == "within_category":
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_category[str(row["category"])].append(row)
        for category in sorted(by_category):
            ordered = sorted(
                by_category[category],
                key=lambda row: (-finite_float(row[score_key], score_key), str(row["row_id"])),
            )
            ordered_groups.append(ordered)
            fractions.update(index / len(ordered) for index in range(1, math.floor(0.5 * len(ordered)) + 1))
    elif scope == "global":
        ordered = sorted(
            rows,
            key=lambda row: (-finite_float(row[score_key], score_key), str(row["row_id"])),
        )
        ordered_groups.append(ordered)
        fractions.update(index / len(ordered) for index in range(1, math.floor(0.5 * len(ordered)) + 1))
    else:
        raise ValueError(f"invalid routing scope: {scope}")
    prefixes: list[list[int]] = []
    for ordered in ordered_groups:
        running = [0]
        for row in ordered:
            running.append(running[-1] + int(row["label"]))
        prefixes.append(running)
    points = []
    for fraction in sorted(value for value in fractions if value <= 0.5):
        selected_positive = 0
        if fraction:
            for ordered, prefix in zip(ordered_groups, prefixes, strict=True):
                selected_positive += prefix[min(len(ordered), math.ceil(fraction * len(ordered)))]
        points.append((fraction, selected_positive / positives))
    area = 0.0
    for left, right in zip(points, points[1:], strict=False):
        area += (right[0] - left[0]) * (right[1] + left[1]) / 2.0
    return area / 0.5


def routing_summary(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    bootstrap: int,
    generation_seconds: float | None,
    preprocessing_seconds: float | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for policy, score_key in POLICY_KEYS.items():
        points = {str(value): metric_at(rows, score_key, value, scope) for value in (0.1, 0.25, 0.5)}
        net_savings = None
        if generation_seconds is not None and preprocessing_seconds is not None:
            selected_calls = points["0.5"]["selected_calls"]
            net_savings = (
                len(rows) * generation_seconds - selected_calls * generation_seconds - preprocessing_seconds
            )
        metrics[policy] = {
            "at_coverage": points,
            "answerability_recall_auc_0_50": routing_auc(rows, score_key, scope),
            "net_time_savings_seconds_at_50": net_savings,
        }

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_id"])].append(row)
    group_ids = sorted(groups)
    if not group_ids:
        raise ValueError("routing bootstrap needs incidents")
    rng = random.Random(SEED)
    comparisons: dict[str, list[float]] = defaultdict(list)
    for _ in range(bootstrap):
        sampled: list[dict[str, Any]] = []
        for draw_index in range(len(group_ids)):
            group_id = rng.choice(group_ids)
            for row in groups[group_id]:
                sampled.append({**row, "row_id": f"{row['row_id']}::draw{draw_index}"})
        availability_auc = routing_auc(sampled, "availability_score", scope)
        availability_yield = metric_at(sampled, "availability_score", 0.5, scope)[
            "evidence_supported_correct_yield"
        ]
        for name, key in (
            ("random", "random_score"),
            ("semantic", "semantic_relevance_proxy"),
        ):
            comparisons[f"auc_availability_minus_{name}"].append(
                availability_auc - routing_auc(sampled, key, scope)
            )
            comparisons[f"yield_availability_minus_{name}"].append(
                availability_yield - metric_at(sampled, key, 0.5, scope)["evidence_supported_correct_yield"]
            )
    return {
        "scope": scope,
        "rows": len(rows),
        "events": len(group_ids),
        "bootstrap_replicates": bootstrap,
        "policies": metrics,
        "paired_differences": {name: ci(values) for name, values in comparisons.items()},
    }


def routing_command(args: argparse.Namespace) -> None:
    if args.availability_feature_dir is not None:
        apply_availability_features(args.output, args.availability_feature_dir)
    rows = routing_rows_smoke() if args.smoke else routing_rows_real()
    summary = routing_summary(
        rows,
        scope=args.scope,
        bootstrap=args.bootstrap,
        generation_seconds=args.generation_seconds,
        preprocessing_seconds=args.preprocessing_seconds,
    )
    summary.update(
        {
            "smoke_only": bool(args.smoke),
            "not_for_tuning_or_selection": bool(args.smoke),
            "semantic_name": "SigLIP2 semantic-relevance proxy",
        }
    )
    destination = args.output / ("smoke/routing_summary.json" if args.smoke else "routing_summary.json")
    write_json(destination, summary)
    print(json.dumps({"status": "ROUTING_COMPLETE", "output": str(destination)}, ensure_ascii=False))


def normalize_smoke_system_rows(output: Path) -> list[dict[str, Any]]:
    models = read_json(output / "selector_models.json")
    availability = {
        str(row["row_id"]): row for row in read_jsonl(DEV_REAL / "real_step1_selector_predictions.jsonl")
    }
    arms = {
        "zero_shot_8b": read_jsonl(LORA_DEV / "evaluation" / "zero_shot.jsonl"),
        "aligned_8b": read_jsonl(LORA_DEV / "evaluation" / "tuned.jsonl"),
    }
    result = []
    for arm, rows in arms.items():
        for row in rows:
            source_id = re.sub(r"^(core|counterfactual)::", "", str(row["row_id"]))
            signal = availability.get(source_id)
            if signal is None or not str(row["source_type"]).startswith("counterfactual_P"):
                continue
            role = str(row["source_type"]).split("counterfactual_", 1)[1]
            confidence = finite_float(row["confidence"], "confidence")
            availability_score = finite_float(signal["real_selector_score"], "availability")
            fusion_score = predict_logistic(models["fusion"], [logit(confidence), logit(availability_score)])
            result.append(
                {
                    "arm": arm,
                    "row_id": source_id,
                    "group_id": str(row["group_id"]),
                    "category": str(row["category"]),
                    "role": role,
                    "correct": bool(row["target_correct"]),
                    "abstained": bool(row["was_abstained"]),
                    "confidence": confidence,
                    "availability": availability_score,
                    "fusion": fusion_score,
                }
            )
    return result


def assemble_system_results(output: Path) -> list[dict[str, Any]]:
    models = read_json(output / "selector_models.json")
    availability_rows = read_jsonl(output / "availability_scores.jsonl")
    availability = {str(row["row_id"]): row for row in availability_rows}
    if len(availability) != 576:
        raise RuntimeError("final assembly requires 576 availability rows")
    result: list[dict[str, Any]] = []
    for arm in ("zero_shot_8b", "aligned_8b"):
        arm_rows_raw: list[dict[str, Any]] = []
        for shard_index in range(3):
            arm_rows_raw.extend(read_jsonl(output / "evaluation" / arm / f"shard{shard_index}.jsonl"))
        assert_unique(arm_rows_raw, "row_id", f"assembled {arm}")
        if len(arm_rows_raw) != 576:
            raise RuntimeError(f"assembled {arm} must contain 576 rows")
        for row in arm_rows_raw:
            signal = availability.get(str(row["row_id"]))
            if signal is None:
                raise RuntimeError(f"missing availability for {row['row_id']}")
            confidence = finite_float(row["confidence"], "confidence")
            availability_score = finite_float(signal["availability_score"], "availability")
            result.append(
                {
                    "arm": arm,
                    "row_id": str(row["row_id"]),
                    "group_id": str(row["group_id"]),
                    "category": str(row["category"]),
                    "role": str(row["counterfactual_role"]),
                    "correct": bool(row["answer_success"]),
                    "abstained": bool(row["abstained"]),
                    "confidence": confidence,
                    "availability": availability_score,
                    "fusion": predict_logistic(
                        models["fusion"], [logit(confidence), logit(availability_score)]
                    ),
                }
            )
    for path in LARGE_BC_FILES:
        for row in read_jsonl(path):
            result.append(
                {
                    "arm": "large_3_zero_shot_reuse",
                    "model_key": str(row["model_key"]),
                    "row_id": f"{row['model_key']}::{row['row_id']}",
                    "source_row_id": str(row["row_id"]),
                    "group_id": str(row["group_id"]),
                    "category": str(row["category"]),
                    "role": str(row["counterfactual_role"]),
                    "correct": bool(row["answer_success"]),
                    "abstained": bool(row["abstained"]),
                    "confidence": finite_float(row["confidence"], "large confidence"),
                }
            )
            result.append(
                {
                    "arm": "reliable_vqa_B3",
                    "model_key": str(row["model_key"]),
                    "row_id": f"{row['model_key']}::{row['row_id']}",
                    "source_row_id": str(row["row_id"]),
                    "group_id": str(row["group_id"]),
                    "category": str(row["category"]),
                    "role": str(row["counterfactual_role"]),
                    "correct": bool(row["answer_success"]),
                    "abstained": bool(row["abstained"]),
                    "b3": predict_logistic(models["b3"], response_features(row)),
                }
            )
    destination = output / "analysis" / "system_results.jsonl"
    write_jsonl(destination, result)
    return result


def arm_rows(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    result = [row for row in rows if row["arm"] == arm]
    if not result:
        raise ValueError(f"missing arm: {arm}")
    assert_unique(result, "row_id", arm)
    return result


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        raise ValueError(f"empty metric rows for {key}")
    return sum(bool(row[key]) for row in rows) / len(rows)


def risk_at_half(rows: list[dict[str, Any]], score: str) -> float:
    selected = selected_rows(rows, score, 0.5, "within_category")
    return 1.0 - mean_bool(selected, "correct")


def gate_statistics(rows: list[dict[str, Any]], bootstrap: int) -> dict[str, Any]:
    zero = arm_rows(rows, "zero_shot_8b")
    aligned = arm_rows(rows, "aligned_8b")
    if {row["row_id"] for row in zero} != {row["row_id"] for row in aligned}:
        raise RuntimeError("8B arms are not exactly paired")
    by_arm_group: dict[str, dict[str, list[dict[str, Any]]]] = {
        "zero_shot_8b": defaultdict(list),
        "aligned_8b": defaultdict(list),
    }
    for row in rows:
        by_arm_group[str(row["arm"])][str(row["group_id"])].append(row)
    groups = sorted(set(by_arm_group["zero_shot_8b"]) & set(by_arm_group["aligned_8b"]))
    rng = random.Random(SEED)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(bootstrap):
        sampled: dict[str, list[dict[str, Any]]] = {"zero_shot_8b": [], "aligned_8b": []}
        for draw_index in range(len(groups)):
            group = rng.choice(groups)
            for arm in sampled:
                sampled[arm].extend(
                    {**row, "row_id": f"{row['row_id']}::draw{draw_index}"}
                    for row in by_arm_group[arm][group]
                )
        z = sampled["zero_shot_8b"]
        a = sampled["aligned_8b"]
        samples["g1_delta_eplus"].append(
            mean_bool([row for row in a if row["role"] == "P+"], "correct")
            - mean_bool([row for row in z if row["role"] == "P+"], "correct")
        )
        samples["g2_delta_eminus_abstention"].append(
            mean_bool([row for row in a if row["role"] in {"P-", "P0"}], "abstained")
            - mean_bool([row for row in z if row["role"] in {"P-", "P0"}], "abstained")
        )
        samples["g2_delta_pminus_assertion"].append(
            (1.0 - mean_bool([row for row in a if row["role"] == "P-"], "abstained"))
            - (1.0 - mean_bool([row for row in z if row["role"] == "P-"], "abstained"))
        )
        aligned_risk_all = 1.0 - mean_bool(a, "correct")
        samples["g3_confidence_risk_difference"].append(risk_at_half(a, "confidence") - aligned_risk_all)
        samples["g4_confidence_minus_fusion_risk"].append(
            risk_at_half(a, "confidence") - risk_at_half(a, "fusion")
        )
    intervals = {name: ci(values) for name, values in samples.items()}
    return {
        "intervals": intervals,
        "decisions": {
            "G1": intervals["g1_delta_eplus"]["ci95_low"] > 0,
            "G2": (
                intervals["g2_delta_eminus_abstention"]["ci95_low"] >= -0.10
                and intervals["g2_delta_pminus_assertion"]["ci95_high"] <= 0.05
            ),
            "G3": intervals["g3_confidence_risk_difference"]["ci95_high"] < 0,
            "G4": intervals["g4_confidence_minus_fusion_risk"]["ci95_low"] > 0,
        },
    }


def partial_dominance_rows(
    system_rows: list[dict[str, Any]], routing: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for system, score, budget in (
        ("zero_shot_8b", None, "not_applicable"),
        ("aligned_8b", None, "not_applicable"),
        ("aligned_8b_confidence_control", "confidence", "structurally_unavailable_post_generation"),
        ("availability_only_selection", "availability", "available_pre_generation"),
        ("confidence_availability_fusion", "fusion", "structurally_unavailable_post_generation"),
        ("reliable_vqa_B3", "b3", "structurally_unavailable_post_generation"),
        ("large_3_zero_shot_reuse", None, "not_applicable"),
    ):
        base_arm = (
            "aligned_8b"
            if system not in {"zero_shot_8b", "large_3_zero_shot_reuse", "reliable_vqa_B3"}
            else system
        )
        available_rows = [row for row in system_rows if row.get("arm") == base_arm]
        risk: Any = "not_available_in_current_input"
        eplus: Any = "not_available_in_current_input"
        pminus: Any = "not_available_in_current_input"
        if available_rows:
            if score and score in available_rows[0]:
                risk = risk_at_half(available_rows, score)
            else:
                risk = 1.0 - mean_bool(available_rows, "correct")
            eplus = mean_bool([row for row in available_rows if row["role"] == "P+"], "correct")
            pminus = mean_bool([row for row in available_rows if row["role"] == "P-"], "abstained")
        rows.append(
            {
                "system": system,
                "risk_at_coverage": risk,
                "eplus_utilization": eplus,
                "pminus_suppression": pminus,
                "budget_recall": budget,
                "cost_per_call": "measure_in_stage_2",
                "preparation_cost": "measure_in_stage_2",
            }
        )
    return rows


def gates_command(args: argparse.Namespace) -> None:
    if args.smoke:
        system_rows = normalize_smoke_system_rows(args.output)
        routing_path = args.output / "smoke" / "routing_summary.json"
    else:
        system_rows = (
            read_jsonl(args.system_results)
            if args.system_results is not None
            else assemble_system_results(args.output)
        )
        routing_path = args.output / "routing_summary.json"
    routing = read_json(routing_path)
    gates = gate_statistics(system_rows, args.bootstrap)
    diffs = routing["paired_differences"]
    availability_policy = routing["policies"]["availability"]
    net = availability_policy["net_time_savings_seconds_at_50"]
    g5 = (
        diffs["auc_availability_minus_random"]["ci95_low"] > 0
        and diffs["auc_availability_minus_semantic"]["ci95_low"] > 0
        and diffs["yield_availability_minus_random"]["ci95_low"] > 0
        and diffs["yield_availability_minus_semantic"]["ci95_low"] > 0
        and net is not None
        and net > 0
    )
    gates["decisions"]["G5"] = g5
    gates["routing_evidence"] = {
        "paired_differences": diffs,
        "availability_net_time_savings_seconds_at_50": net,
    }
    gates["smoke_only"] = bool(args.smoke)
    gates["not_for_tuning_or_selection"] = bool(args.smoke)
    destination_dir = args.output / "smoke" if args.smoke else args.output / "analysis"
    write_json(destination_dir / "gate_results.json", gates)
    matrix = partial_dominance_rows(system_rows, routing)
    write_csv(destination_dir / "partial_dominance_matrix.csv", matrix)
    print(json.dumps({"status": "GATES_COMPLETE", "decisions": gates["decisions"]}, ensure_ascii=False))


def smoke_command(args: argparse.Namespace) -> None:
    models_path = args.output / "selector_models.json"
    if not models_path.is_file():
        selector_command(argparse.Namespace(output=args.output))
    routing_args = argparse.Namespace(
        output=args.output,
        smoke=True,
        scope="within_category",
        bootstrap=args.bootstrap,
        generation_seconds=None,
        preprocessing_seconds=None,
        availability_feature_dir=None,
    )
    routing_command(routing_args)
    gates_command(
        argparse.Namespace(
            output=args.output,
            smoke=True,
            bootstrap=args.bootstrap,
            system_results=None,
        )
    )
    artifacts = [
        args.output / "selector_models.json",
        args.output / "smoke" / "routing_summary.json",
        args.output / "smoke" / "gate_results.json",
        args.output / "smoke" / "partial_dominance_matrix.csv",
    ]
    manifest = {
        "schema_version": "secondary_generalization_bc_smoke_v1",
        "status": "PASS",
        "data_surface": "stored development/validation only",
        "bc_model_inference": False,
        "external_api_calls": 0,
        "bootstrap_replicates": args.bootstrap,
        "not_for_tuning_or_selection": True,
        "artifacts": {str(path.relative_to(args.output)): sha256_file(path) for path in artifacts},
    }
    write_json(args.output / "smoke" / "smoke_manifest.json", manifest)
    prereg_manifest = read_json(args.output / "prereg_manifest.json")
    inventory = read_json(args.output / "inventory.json")
    checklist = {
        "schema_version": "secondary_generalization_bc_go_checklist_v1",
        "status": "CONDITIONAL_GO_AFTER_STAGE2_PREFLIGHT",
        "completed": {
            "preregistration_json_sealed": True,
            "large_model_reuse_coverage_verified": True,
            "local_images_complete": True,
            "epoch3_checkpoint_hashes_verified_on_server": True,
            "three_incident_shards_built": True,
            "dev_val_analysis_smoke_passed": True,
            "bc_model_inference_calls": 0,
            "external_api_calls": 0,
        },
        "inventory": {
            "large_models": 3,
            "rows_per_large_model": 576,
            "events_per_large_model": 96,
            "confidence_present_per_large_model": 576,
            "system_panel_legacy_split_counts": {"heldout-B": 0, "heldout-C": 96},
            "local_image_files": 960,
            "local_image_bytes": 359535087,
            "remote_image_files": inventory["remote_gpu_host"]["existing_files"],
            "remote_missing_image_files": inventory["remote_gpu_host"]["missing_files"],
            "deterministic_reextraction_required": False,
        },
        "preregistration_hashes": prereg_manifest,
        "stage2_required_preflight": [
            "commit and push the exact execution code/protocol state to origin/main, then verify the server checkout hash matches",
            "leave the two pre-existing untracked LoRA runner files untouched or use a clean execution worktree",
            "transfer only the 960 existing image files by LF file-list tar|ssh and verify count plus bytes",
            "run a dev-only vLLM Docker endpoint/logprob/LoRA preflight before opening any B/C inference shard",
            "verify the vllm image ID and epoch-3 adapter hashes still equal the preregistered values",
        ],
        "stage2_sequence": [
            "sync exact committed code to gpu-host",
            "transfer the 359535087-byte image bundle",
            "extract frozen SigLIP2 and fresh-DINO signals for the 480 unique non-restored rows, then reuse E+ scores for Pr duplicates",
            "launch three vllm/vllm-openai containers on GPU 0,1,2 and run 32 incidents per GPU for zero-shot and aligned arms",
            "train/apply no new selector: use selector_models.json exactly as sealed",
            "run routing and gates with 4000 incident bootstrap replicates and measured preprocessing/generation wall time",
        ],
        "estimated_elapsed_minutes": {
            "transfer_and_verification": [2, 5],
            "frozen_signal_extraction": [5, 15],
            "docker_start_and_dev_preflight": [3, 8],
            "two_8b_arms_three_gpu": [5, 10],
            "analysis_and_recovery": [2, 5],
            "total": [17, 43],
        },
        "remaining_risks": [
            "the frozen eligible system panel and routing core are heldout-C only despite the historical B/C source-pool name",
            "OpenAI-compatible vLLM token logprobs with multimodal LoRA must pass the dev-only preflight",
            "the epoch-3 adapter exists on the server but not in the recovered local output tree",
            "vllm/vllm-openai:latest is mutable, so the observed image ID must be rechecked",
            "G5 cannot pass without complete measured preprocessing time and mean generation time",
        ],
    }
    write_json(args.output / "go_checklist.json", checklist)
    stage1_paths = [
        args.output / "prereg_design.json",
        args.output / "arm_specs.json",
        args.output / "analysis_plan.json",
        args.output / "prereg_manifest.json",
        args.output / "inventory.json",
        args.output / "docker_launch_plan.json",
        args.output / "selector_models.json",
        args.output / "go_checklist.json",
        args.output / "build" / "bc_input.jsonl",
        args.output / "build" / "bc_input_shard0.jsonl",
        args.output / "build" / "bc_input_shard1.jsonl",
        args.output / "build" / "bc_input_shard2.jsonl",
        args.output / "build" / "availability_signal_input.jsonl",
        args.output / "smoke" / "routing_summary.json",
        args.output / "smoke" / "gate_results.json",
        args.output / "smoke" / "partial_dominance_matrix.csv",
        args.output / "smoke" / "smoke_manifest.json",
    ]
    stage1 = {
        "schema_version": "secondary_generalization_bc_stage1_seal_v1",
        "status": "STAGE_1_PREPARED_NO_BC_INFERENCE",
        "bc_model_inference_calls": 0,
        "external_api_calls": 0,
        "reserve_or_heldout_d_data_used": False,
        "artifacts": {
            str(path.relative_to(args.output)).replace("\\", "/"): sha256_file(path) for path in stage1_paths
        },
        "operator_script": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "read_only_source_hashes": {
            str(INPUT.relative_to(ROOT)).replace("\\", "/"): sha256_file(INPUT),
            **{str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in LARGE_BC_FILES},
        },
    }
    write_json(args.output / "stage1_manifest.json", stage1)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build")
    commands.add_parser("selector")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--arm", choices=("zero_shot_8b", "aligned_8b"), required=True)
    evaluate.add_argument("--shard-index", type=int, choices=(0, 1, 2), required=True)
    evaluate.add_argument("--base-url", required=True)
    evaluate.add_argument("--confirm-bc-inference", required=True)
    routing = commands.add_parser("routing")
    routing.add_argument("--scope", choices=("within_category", "global"), default="within_category")
    routing.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    routing.add_argument("--generation-seconds", type=float)
    routing.add_argument("--preprocessing-seconds", type=float)
    routing.add_argument("--availability-feature-dir", type=Path)
    routing.add_argument("--smoke", action="store_true")
    gates = commands.add_parser("gates")
    gates.add_argument("--system-results", type=Path)
    gates.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    gates.add_argument("--smoke", action="store_true")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--bootstrap", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "selector":
        selector_command(args)
    elif args.command == "routing":
        routing_command(args)
    elif args.command == "gates":
        gates_command(args)
    elif args.command == "smoke":
        smoke_command(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
