"""Run the sealed W4 cohort once, then replay the frozen policies.

``dry-run`` is the only command intended before the runner, registration, and
checksum sidecar are committed.  It reads the sealed cohort metadata and the
already completed train/dev selection artifacts, but never opens new-cohort
media, imports model runtimes, or reads new-cohort model outputs.

The live commands form one canonical path: ``preflight``, ``materialize``,
``transfer``, ``features``, both ``answerer`` arms, ``replay``, and ``seal``.
Every long stage resumes only from append-only or atomic checkpoints.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import w4_dev_comparator_runner as dev
from scripts.operator.w3w5a_common import ABSTAIN, ACTIVITY_TEXT

SEED = 20260714
BOOTSTRAP_REPLICATES = 4000
EXPECTED_EVENTS = 260
EXPECTED_VIEWS = 1056
MIN_FREE_BYTES = 80 * 1024**3
PRIMARY_ARM = "answer_abstain_sft"
CONTROL_ARM = "explicit_abstention_prompt"
ARMS = (PRIMARY_ARM, CONTROL_ARM)
RELEASE_THRESHOLDS = {
    PRIMARY_ARM: 0.981128697326919,
    CONTROL_ARM: 0.9491225851331859,
}
POLICY_ORDER = dev.POLICY_ORDER
FOCAL_POLICY = "availability"
COMPARATOR_POLICY = "task_matched_selector"
EXPECTED_GPU = {PRIMARY_ARM: 0, CONTROL_ARM: 1}
EXPECTED_COHORT_SHA256 = "d2f8fc73483850d5131551cddf3f6f45fc387588c467dc598212a2d53974af21"
EXPECTED_CONTRACT_SHA256 = "a1c359e2905210e0e499f38c452166fdd7f5974a6cbe890ebb30e34da3b45aa3"
EXPECTED_DEV_SELECTION_SHA256 = "eb235723964355585e974b5f669a2e5eeb225794f6a2307999db5ee6cb89d042"
EXPECTED_W5A_SELECTION_SHA256 = "afd607ecc221fa00fe28b7610039bed9a19ca46a89c69031c6405c5e444dfc1f"
EXPECTED_AVAILABILITY_MODEL_SHA256 = dev.AVAILABILITY_MODEL_SHA256
DEFAULT_COHORT_MANIFEST = (
    ROOT / "outputs" / "meva_full_20260713" / "w4_prereg_20260714" / "w4_cohort_manifest.json"
)
DEFAULT_REGISTRATION = (
    ROOT / "outputs" / "meva_full_20260713" / "w4_prereg_20260714" / "w4_amended_registration.json"
)
DEFAULT_CONTRACT = (
    ROOT / "outputs" / "meva_full_20260713" / "w4_prereg_20260714" / "w4_feature_extraction_contract.json"
)
DEFAULT_DEV_ROOT = ROOT / "outputs" / "meva_full_20260713" / "w4_dev_comparator_20260714"
DEFAULT_DEV_SELECTION = DEFAULT_DEV_ROOT / "features_host_v2" / "w4_dev_comparator_selection.json"
DEFAULT_W5A_SELECTION = (
    ROOT / "outputs" / "meva_full_20260713" / "w5a_fair_baselines" / "w5a_dev_selection.json"
)
VIDEO_ORIENTATION_AUDIT = (
    ROOT / "outputs" / "meva_full_20260713" / "video_orientation_audit_v2.json"
)
EXPECTED_VIDEO_ORIENTATION_AUDIT_SHA256 = (
    "1d7e0d033e22e9801163be1328ed9bae2cc94872d5c335f7ce88f5b1c6c0d5c0"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "meva_full_20260713" / "w4_oneshot_20260714"
DEFAULT_AVAILABILITY_MODEL = dev.DEFAULT_AVAILABILITY_MODEL
STAGE_SCHEMA = (
    {
        "stage": "preflight",
        "environment": "server host conda mvevirag",
        "checkpoint": "checkpoints/preflight.json",
        "requires": ["HEAD=origin/main", "sealed SHA checks", "80 GiB free", "startup regression gate"],
    },
    {
        "stage": "materialize",
        "environment": "local canonical-video host; GPU=0",
        "checkpoint": "checkpoints/materialize.json",
        "requires": ["260 events", "1056 views", "semantic_valid written in-row"],
    },
    {
        "stage": "transfer",
        "environment": "local tar piped to SSH",
        "checkpoint": "checkpoints/transfer.json",
        "requires": ["LF file list", "resume by size", "count/bytes/sample SHA verification"],
    },
    {
        "stage": "features",
        "environment": "server host conda mvevirag; GPU2",
        "checkpoint": "checkpoints/features.json",
        "requires": ["Grounding DINO", "SigLIP2 image FP32", "DINOv2 CLS FP32", "text max_length=64", "M4"],
    },
    {
        "stage": "answerer",
        "environment": "vLLM container; arm-dedicated GPU0/GPU1",
        "checkpoint": "checkpoints/answerer.<arm>.json",
        "requires": ["1056 views once per arm", "append-only resume", "nohup-compatible CLI"],
    },
    {
        "stage": "replay",
        "environment": "CPU analysis",
        "checkpoint": "checkpoints/replay.json",
        "requires": ["six policies", "U(k)/V(k)", "H1/H2", "role table", "Green/Amber/Red"],
    },
    {
        "stage": "seal",
        "environment": "CPU hashing",
        "checkpoint": "checkpoints/seal.json",
        "requires": ["artifact SHA manifest", "result JSON", "registration result-hash diff"],
    },
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    return dev.sha256_file(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return dev.read_jsonl(path)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl_fsync(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_checkpoint(output_root: Path, stage: str, payload: dict[str, Any]) -> Path:
    path = output_root / "checkpoints" / f"{stage}.json"
    atomic_write_json(
        path,
        {
            "schema_version": "meva_w4_oneshot_checkpoint_v1",
            "stage": stage,
            "status": "completed",
            "completed_at_utc": utc_now(),
            "runner_sha256": sha256_file(Path(__file__)),
            **payload,
        },
    )
    return path


def require_checkpoint(output_root: Path, stage: str) -> dict[str, Any]:
    path = output_root / "checkpoints" / f"{stage}.json"
    if not path.is_file():
        raise FileNotFoundError(f"required checkpoint is absent: {path}")
    checkpoint = load_json(path)
    if checkpoint.get("stage") != stage or checkpoint.get("status") != "completed":
        raise RuntimeError(f"invalid checkpoint: {path}")
    return checkpoint


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{label} SHA-256 drift: {observed} != {expected}")


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise RuntimeError(f"invalid SHA sidecar: {sidecar}")
    observed = sha256_file(path)
    if fields[0] != observed:
        raise RuntimeError(f"SHA sidecar mismatch: {path}")
    return observed


def write_probe(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".w4_probe_", dir=path, delete=False) as handle:
        probe = Path(handle.name)
        handle.write(b"w4-oneshot-write-probe\n")
        handle.flush()
        os.fsync(handle.fileno())
    probe.unlink()


def load_sealed_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    require_sha(args.cohort_manifest, EXPECTED_COHORT_SHA256, "cohort manifest")
    require_sha(args.runtime_contract, EXPECTED_CONTRACT_SHA256, "runtime contract")
    require_sha(args.dev_selection, EXPECTED_DEV_SELECTION_SHA256, "dev selection")
    require_sha(args.w5a_selection, EXPECTED_W5A_SELECTION_SHA256, "W5a selection")
    registration_sha = verify_sidecar(args.registration)
    manifest = load_json(args.cohort_manifest)
    registration = load_json(args.registration)
    if registration["cohort"]["manifest_sha256"] != EXPECTED_COHORT_SHA256:
        raise RuntimeError("registration cohort SHA drift")
    if registration["seal_requirements"]["feature_extraction_contract"]["sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("registration contract SHA drift")
    clarification = registration.get("execution_semantics_clarification")
    if not isinstance(clarification, dict):
        raise RuntimeError("registration execution-semantics clarification is absent")
    if clarification.get("result_observation_state") != "STOP_NOT_RUN":
        raise RuntimeError("registration is no longer in STOP_NOT_RUN state")
    if clarification.get("runner", {}).get("sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("registration runner SHA drift")
    return manifest, registration | {"_verified_sha256": registration_sha}


def cohort_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != "meva_w4_unused_cohort_manifest_v1":
        raise RuntimeError("cohort manifest schema drift")
    events = manifest.get("events")
    if not isinstance(events, list) or len(events) != EXPECTED_EVENTS:
        raise RuntimeError("cohort event count drift")
    rows: list[dict[str, Any]] = []
    for event in events:
        views = event["views"]
        positives = [view for view in views if view["role"] == "positive"]
        if len(positives) != 1 or len(views) != event["n_cand"]:
            raise RuntimeError(f"cohort event view structure drift: {event['event_id']}")
        positive = positives[0]
        question, affirmative = ACTIVITY_TEXT[event["activity"]]
        common = {
            "event_id": event["event_id"],
            "group_id": event["event_id"],
            "split": "cohort",
            "activity": event["activity"],
            "question": question,
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "positive_anchor_event_uid": event["anchor_positive_event_uid"],
            "positive_anchor_actor_id": int(positive["selection_actor_id"]),
            "positive_anchor_clip": positive["clip"],
            "positive_anchor_frame": int(positive["native_frame"]),
            "positive_anchor_model_file": positive["model_file"],
            "reference_frame": int(positive["reference_frame"]),
            "n_cand": int(event["n_cand"]),
        }
        for view in views:
            role = view["role"]
            row = common | {
                "view_id": view["view_id"],
                "row_id": view["view_id"],
                "camera_id": view["camera_id"],
                "role": role,
                "label": int(role == "positive"),
                "clip": view["clip"],
                "native_frame": int(view["native_frame"]),
                "model_file": view["model_file"],
                "target_answer": affirmative if role == "positive" else ABSTAIN,
                "target_type": "answer" if role == "positive" else "abstain",
                "sealed_view_spec_sha256": view["view_spec_sha256"],
                "source_video": view["s3"],
            }
            if role != "positive":
                row["minimum_outside_clearance_deg_beyond_buffer"] = float(
                    view["minimum_outside_clearance_deg_beyond_buffer"]
                )
            rows.append(row)
    rows.sort(key=lambda row: (row["event_id"], row["camera_id"]))
    if len(rows) != EXPECTED_VIEWS or len({row["row_id"] for row in rows}) != EXPECTED_VIEWS:
        raise RuntimeError("cohort candidate-view coverage drift")
    return rows


def logistic_scores(matrix: np.ndarray, selector: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(selector["scaler_mean"], dtype=np.float64)
    scale = np.asarray(selector["scaler_scale"], dtype=np.float64)
    coefficient = np.asarray(selector["coef"], dtype=np.float64)
    standardized = (matrix.astype(np.float64) - mean) / scale
    logits = standardized @ coefficient + float(selector["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))


def deterministic_policy_order(policy: str, event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = {
        "person_count_area": lambda row: (-row["person_count"], -row["person_area"], row["camera_id"]),
        "siglip_semantic_relevance": lambda row: (-row["semantic_score"], row["camera_id"]),
        "task_matched_selector": lambda row: (-row["task_matched_score"], row["camera_id"]),
        "availability": lambda row: (-row["availability_score"], row["camera_id"]),
    }
    if policy in keys:
        return sorted(event_rows, key=keys[policy])
    if policy != "availability_plus_diversity":
        raise ValueError(f"unknown deterministic policy: {policy}")
    remaining = list(event_rows)
    selected = [min(remaining, key=keys["availability"])]
    remaining.remove(selected[0])
    while remaining:

        def diversity_key(row: dict[str, Any]) -> tuple[float, str]:
            redundancy = max(
                float(np.dot(row["image_embedding"], prior["image_embedding"])) for prior in selected
            )
            return (-(row["availability_score"] - redundancy), row["camera_id"])

        next_row = min(remaining, key=diversity_key)
        selected.append(next_row)
        remaining.remove(next_row)
    return selected


def subset_outcome(rows: Sequence[dict[str, Any]], arm: str) -> tuple[float, float, float]:
    return (
        float(any(row["arm_metrics"][arm]["correct_release"] for row in rows)),
        float(any(row["arm_metrics"][arm]["violation_release"] for row in rows)),
        float(sum(row["arm_metrics"][arm]["vlm_gpu_seconds"] for row in rows)),
    )


def exact_random_event(rows: list[dict[str, Any]], k: int, arm: str) -> tuple[float, float, float]:
    outcomes = [subset_outcome(subset, arm) for subset in combinations(rows, k)]
    if not outcomes:
        raise RuntimeError("random exact expectation has no subsets")
    return tuple(float(np.mean([item[index] for item in outcomes])) for index in range(3))  # type: ignore[return-value]


def paired_bootstrap(
    left: Sequence[float], right: Sequence[float], *, seed: int = SEED, replicates: int = BOOTSTRAP_REPLICATES
) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.ndim != 1 or not left_array.size:
        raise ValueError("paired bootstrap inputs must be non-empty equal-length vectors")
    difference = left_array - right_array
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, difference.size, size=(replicates, difference.size))
    estimates = difference[indices].mean(axis=1)
    return {
        "delta": float(difference.mean()),
        "two_sided_95_ci": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "one_sided_95_upper": float(np.quantile(estimates, 0.95)),
        "replicates": replicates,
        "seed": seed,
    }


def branch_color(h1: dict[str, Any], h2: dict[str, Any]) -> str:
    common = h1["ci_pass"] and h1["safety_pass"] and h1["cost_pass"] and h2["pass"]
    if common and h1["effect_pass"]:
        return "Green"
    if common and 0.07 <= h1["delta_u"] < 0.10:
        return "Amber"
    return "Red"


def _dev_replay(args: argparse.Namespace) -> dict[str, Any]:
    root = args.dev_root
    selection = load_json(args.dev_selection)
    rows = read_jsonl(root / "candidate_views.materialized.jsonl")
    validity = {
        row["row_id"]: row for row in read_jsonl(root / "features_host_v2" / "semantic_validity.jsonl")
    }
    answers = {row["row_id"]: row for row in read_jsonl(root / "answerer_outputs.jsonl")}
    dino = {row["row_id"]: row for row in read_jsonl(root / "features" / "dino.jsonl")}
    availability = {
        row["row_id"]: float(row["availability_score"])
        for row in read_jsonl(root / "features_host_v2" / "m4_availability_scores.jsonl")
    }
    if not all(
        set(table) == {row["row_id"] for row in rows} for table in (validity, answers, dino, availability)
    ):
        raise RuntimeError("dev replay sealed row coverage drift")
    cache = np.load(root / "features_host_v2" / "siglip2.npz", allow_pickle=True)
    row_ids = cache["row_ids"].tolist()
    if row_ids != [row["row_id"] for row in rows]:
        raise RuntimeError("dev replay feature row order drift")
    image = np.asarray(cache["image"], dtype=np.float32)
    questions = cache["questions"].tolist()
    query_map = {
        question: np.asarray(cache["query"][index], dtype=np.float32)
        for index, question in enumerate(questions)
    }
    query = np.stack([query_map[row["question"]] for row in rows])
    dino_matrix = np.asarray(
        [[float(dino[row["row_id"]][key]) for key in dev.DINO_KEYS] for row in rows], dtype=np.float64
    )
    combined = np.concatenate(
        [
            image.astype(np.float64),
            query.astype(np.float64),
            image * query,
            np.abs(image - query),
            dino_matrix,
        ],
        axis=1,
    )
    task = logistic_scores(combined, selection["task_selector"])
    semantic = dev.semantic_scores(
        image,
        query,
        np.asarray([validity[row["row_id"]]["semantic_candidate_valid"] is True for row in rows]),
    )
    enriched: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        answer = answers[row["row_id"]]
        released = (
            not bool(answer["was_abstained"])
            and float(answer["confidence"]) >= RELEASE_THRESHOLDS[PRIMARY_ARM]
        )
        enriched.append(
            row
            | {
                "correct_release": float(row["role"] == "positive" and released and answer["target_correct"]),
                "violation_release": float(row["role"] != "positive" and released),
                "person_count": float(dino[row["row_id"]]["fresh_dino_person_regions"]),
                "person_area": float(dino[row["row_id"]]["person_area_sum"]),
                "semantic_score": float(semantic[index]),
                "task_matched_score": float(task[index]),
                "availability_score": availability[row["row_id"]],
                "image_embedding": image[index],
            }
        )
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        if row["split"] == "dev":
            by_event[row["event_id"]].append(row)
    observed: dict[str, float] = {}
    observed[POLICY_ORDER[0]] = float(
        np.mean([np.mean([row["correct_release"] for row in values]) for values in by_event.values()])
    )
    for policy in POLICY_ORDER[1:]:
        chosen = [deterministic_policy_order(policy, values)[0] for values in by_event.values()]
        observed[policy] = float(np.mean([row["correct_release"] for row in chosen]))
    expected = {policy: float(selection["policy_results"][policy]["U_1"]) for policy in POLICY_ORDER}
    maximum_error = max(abs(observed[policy] - expected[policy]) for policy in POLICY_ORDER)
    if maximum_error > 1e-12:
        raise RuntimeError(f"dev U(1) replay mismatch: max_abs_error={maximum_error}")
    return {
        "events": len(by_event),
        "observed_u_1": observed,
        "expected_u_1": expected,
        "max_abs_error": maximum_error,
    }


def _synthetic_formula_tests() -> dict[str, Any]:
    random_rows = [
        {
            "arm_metrics": {
                PRIMARY_ARM: {"correct_release": value, "violation_release": 0.0, "vlm_gpu_seconds": 1.0}
            }
        }
        for value in (1.0, 0.0, 0.0)
    ]
    random_u, random_v, random_cost = exact_random_event(random_rows, 2, PRIMARY_ARM)
    if not math.isclose(random_u, 2.0 / 3.0) or random_v != 0.0 or random_cost != 2.0:
        raise RuntimeError("exact-random formula test failed")
    h1_u = paired_bootstrap(np.full(40, 0.20), np.zeros(40))
    h1_v = paired_bootstrap(np.zeros(40), np.zeros(40))
    h2_u = paired_bootstrap(np.full(40, 0.10), np.zeros(40))
    h2_v = paired_bootstrap(np.zeros(40), np.zeros(40))
    if h1_u["two_sided_95_ci"][0] <= 0 or h1_v["one_sided_95_upper"] >= 0.05:
        raise RuntimeError("H1 bootstrap formula test failed")
    if h2_u["two_sided_95_ci"][0] <= 0 or h2_v["one_sided_95_upper"] >= 0.05:
        raise RuntimeError("H2 bootstrap formula test failed")
    green = branch_color(
        {"ci_pass": True, "safety_pass": True, "cost_pass": True, "effect_pass": True, "delta_u": 0.10},
        {"pass": True},
    )
    amber = branch_color(
        {"ci_pass": True, "safety_pass": True, "cost_pass": True, "effect_pass": False, "delta_u": 0.08},
        {"pass": True},
    )
    red = branch_color(
        {"ci_pass": False, "safety_pass": True, "cost_pass": True, "effect_pass": True, "delta_u": 0.12},
        {"pass": True},
    )
    if (green, amber, red) != ("Green", "Amber", "Red"):
        raise RuntimeError("branch-color formula test failed")
    return {
        "exact_random_k2": {"U": random_u, "V": random_v, "selected_vlm_gpu_seconds": random_cost},
        "H1": {"delta_u": h1_u, "delta_v": h1_v, "pass": True},
        "H2": {"delta_yield": h2_u, "delta_v": h2_v, "pass": True},
        "branch_colors": [green, amber, red],
    }


def dry_run(args: argparse.Namespace) -> int:
    manifest, registration = load_sealed_inputs(args)
    rows = cohort_rows(manifest)
    contract = load_json(args.runtime_contract)
    w5a = load_json(args.w5a_selection)
    checks = {
        "cohort_260_events": len(manifest["events"]) == EXPECTED_EVENTS,
        "cohort_1056_views": len(rows) == EXPECTED_VIEWS,
        "manifest_sha256": registration["cohort"]["manifest_sha256"] == EXPECTED_COHORT_SHA256,
        "policy_order": tuple(registration["routing"]["deployable_policy_order"]) == POLICY_ORDER,
        "focal_policy": registration["routing"]["focal_policy"] == FOCAL_POLICY,
        "comparator_policy": registration["dev_comparator_selection"]["selected_primary_comparator"]
        == COMPARATOR_POLICY,
        "primary_threshold": registration["answer_and_release"]["release_threshold"]
        == RELEASE_THRESHOLDS[PRIMARY_ARM],
        "control_threshold": w5a["selected_dev_rule_per_arm"][CONTROL_ARM]["threshold"]
        == RELEASE_THRESHOLDS[CONTROL_ARM],
        "H1_bootstrap": registration["hypotheses"]["H1"]["bootstrap_replicates"] == BOOTSTRAP_REPLICATES,
        "H1_seed": registration["hypotheses"]["H1"]["bootstrap_seed"] == SEED,
        "H1_effect": registration["hypotheses"]["H1"]["effect_gate"] == ">= +0.10 absolute",
        "H1_safety": registration["hypotheses"]["H1"]["safety"].endswith("< +0.05"),
        "H2_arms": registration["hypotheses"]["H2"]["comparison"] == f"{PRIMARY_ARM} minus {CONTROL_ARM}",
        "H2_safety": registration["hypotheses"]["H2"]["safety"].endswith("< +0.05"),
        "text_padding_64": contract["text_preprocessing"]
        == {"max_length": 64, "padding": "max_length", "truncation": True},
        "STOP_NOT_RUN": registration["execution_semantics_clarification"]["result_observation_state"]
        == "STOP_NOT_RUN",
        "full_view_replay_clarified": registration["execution_semantics_clarification"]["clarification"]
        == "답변자 전 뷰 실측 + 정책 재생은 dev 선정과 동일 방법론이며 전수 비용 실측 정의를 충족",
        "runner_sha_registered": registration["execution_semantics_clarification"]["runner"]["sha256"]
        == sha256_file(Path(__file__)),
        "stage_schema": [item["stage"] for item in STAGE_SCHEMA]
        == ["preflight", "materialize", "transfer", "features", "answerer", "replay", "seal"],
        "new_cohort_result_absent": not (args.output_root / "w4_results.json").exists(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"registration checklist failed: {failed}")
    dev_replay = _dev_replay(args)
    formula_tests = _synthetic_formula_tests()
    model_modules = sorted(name for name in ("torch", "transformers", "vllm") if name in sys.modules)
    if model_modules:
        raise RuntimeError(f"dry-run imported model runtimes: {model_modules}")
    report = {
        "schema_version": "meva_w4_oneshot_dry_run_v1",
        "status": "PASS",
        "mode": "dry-run",
        "result_observation_state": "STOP_NOT_RUN",
        "cohort": {"events": len(manifest["events"]), "candidate_views": len(rows)},
        "stage_schema": STAGE_SCHEMA,
        "dev_u1_replay": dev_replay,
        "synthetic_formula_tests": formula_tests,
        "registration_checklist": {name: "PASS" for name in checks},
        "new_cohort_media_opened": 0,
        "new_cohort_model_outputs_read": 0,
        "new_cohort_model_inferences": 0,
        "gpu_jobs_started": 0,
        "model_modules_loaded": model_modules,
        "writes_performed": 0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout.strip()


def preflight(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest, registration = load_sealed_inputs(args)
    rows = cohort_rows(manifest)
    head = git_output("rev-parse", "HEAD")
    origin_main = git_output("rev-parse", "origin/main")
    if head != origin_main:
        raise RuntimeError(f"HEAD is not origin/main: {head} != {origin_main}")
    protected_paths = [
        "scripts/w4_oneshot_runner.py",
        str(args.registration.relative_to(ROOT)).replace("\\", "/"),
        str(args.registration.with_suffix(args.registration.suffix + ".sha256").relative_to(ROOT)).replace(
            "\\", "/"
        ),
    ]
    for path in protected_paths:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", path], cwd=ROOT, check=True, capture_output=True
        )
        if subprocess.run(["git", "diff", "--quiet", "--", path], cwd=ROOT).returncode != 0:
            raise RuntimeError(f"live execution path is dirty: {path}")
        if subprocess.run(["git", "diff", "--cached", "--quiet", "--", path], cwd=ROOT).returncode != 0:
            raise RuntimeError(f"live execution path is staged but not committed: {path}")
    free_bytes = shutil.disk_usage(output_root).free
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"free space below 80 GiB: {free_bytes}")
    features_host = output_root / "features_host"
    write_probe(features_host)
    dev.require_host_conda_environment()
    runtime_args = argparse.Namespace(
        runtime_contract=args.runtime_contract,
        availability_model=args.availability_model,
        device=args.device,
    )
    startup_gate = dev.runtime_contract_gate(runtime_args)
    checkpoint = write_checkpoint(
        output_root,
        "preflight",
        {
            "head": head,
            "origin_main": origin_main,
            "free_bytes": free_bytes,
            "events": EXPECTED_EVENTS,
            "candidate_views": len(rows),
            "cohort_manifest_sha256": EXPECTED_COHORT_SHA256,
            "registration_sha256": registration["_verified_sha256"],
            "runtime_contract_sha256": EXPECTED_CONTRACT_SHA256,
            "startup_regression_gate": startup_gate,
            "features_host_write_probe": "passed",
        },
    )
    print(json.dumps({"status": "PASS", "checkpoint": str(checkpoint), "head": head}, indent=2))
    return 0


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def write_image_same_or_new(path: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if sha256_file(path) != digest:
            raise RuntimeError(f"materialized image drift on resume: {path}")
        return digest
    atomic_write_bytes(path, payload)
    return digest


def load_video_orientation_rules() -> dict[str, str]:
    require_sha(
        VIDEO_ORIENTATION_AUDIT,
        EXPECTED_VIDEO_ORIENTATION_AUDIT_SHA256,
        "video orientation audit",
    )
    audit = load_json(VIDEO_ORIENTATION_AUDIT)
    if (
        audit.get("schema_version") != "meva_m2_v2_video_orientation_audit"
        or audit.get("finding")
        != "G639 r13 AVI frames are stored as upright 1072x1920 portrait images, while the MEVA "
        "KRTD and supplied camera-model reference frame use a 1920x1072 canonical coordinate image."
        or audit.get("canonicalization")
        != "Rotate portrait-decoded frames 90 degrees counterclockwise before applying KRTD "
        "projections or annotation-coordinate crops."
    ):
        raise RuntimeError("video orientation audit semantics drift")
    return {"G639": "rotate_90_counterclockwise"}


def normalize_video_frame(
    frame: Any, camera_id: str, orientation_rules: dict[str, str]
) -> Any:
    import cv2

    rule = orientation_rules.get(camera_id)
    if rule is not None:
        if rule != "rotate_90_counterclockwise":
            raise RuntimeError(f"unsupported audited orientation rule: {camera_id}: {rule}")
        if frame.shape[1] != 1072 or frame.shape[0] != 1920:
            raise RuntimeError(f"unexpected audited source frame shape: {camera_id}: {frame.shape}")
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if frame.shape[1] != 1920 or frame.shape[0] not in (1072, 1080):
        raise RuntimeError(f"unexpected normalized frame shape: {camera_id}: {frame.shape}")
    if frame.shape[0] == 1072:
        frame = cv2.copyMakeBorder(frame, 0, 8, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return frame


def read_video_frame(
    path: Path,
    frame_index: int,
    camera_id: str,
    orientation_rules: dict[str, str],
) -> Any:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    actual = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
    capture.release()
    if not ok or frame is None or actual != frame_index:
        raise RuntimeError(f"frame seek mismatch: {path.name} {frame_index=} {actual=}")
    return normalize_video_frame(frame, camera_id, orientation_rules)


def validate_materialized_prefix(
    output_root: Path, planned: list[dict[str, Any]], completed: list[dict[str, Any]]
) -> None:
    if len(completed) > len(planned):
        raise RuntimeError("materialization checkpoint has excess rows")
    expected_prefix = [row["row_id"] for row in planned[: len(completed)]]
    if [row["row_id"] for row in completed] != expected_prefix:
        raise RuntimeError("materialization checkpoint is not an ordered prefix")
    for row in completed:
        if not isinstance(row.get("semantic_valid"), bool):
            raise RuntimeError(f"semantic_valid is absent: {row['row_id']}")
        for key in ("full", "crop"):
            path = output_root / row[f"{key}_path"]
            if not path.is_file() or sha256_file(path) != row[f"{key}_sha256"]:
                raise RuntimeError(f"materialization resume artifact drift: {path}")


def materialize(args: argparse.Namespace) -> int:
    import cv2

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest, registration = load_sealed_inputs(args)
    planned = cohort_rows(manifest)
    videos = dev.video_index(args.video_root.resolve())
    missing = sorted({row["clip"] for row in planned} - set(videos))
    if missing:
        raise FileNotFoundError(f"missing canonical videos: {len(missing)}")
    orientation_rules = load_video_orientation_rules()
    journal = output_root / "candidate_views.materialized.jsonl"
    completed = read_jsonl(journal) if journal.exists() else []
    validate_materialized_prefix(output_root, planned, completed)
    anchors: dict[str, dict[str, Any]] = {}
    stage_started = time.perf_counter()
    for index, row in enumerate(planned[len(completed) :], len(completed) + 1):
        row_started = time.perf_counter()
        anchor = anchors.get(row["event_id"])
        if anchor is None:
            anchor = dev.anchor_geometry(row, args.meva_repo.resolve())
            anchors[row["event_id"]] = anchor
        frame = read_video_frame(
            videos[row["clip"]],
            int(row["native_frame"]),
            row["camera_id"],
            orientation_rules,
        )
        if row["role"] == "positive":
            box = anchor["bbox"]
            center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        else:
            center = dev.negative_center(row, anchor, args.meva_repo.resolve())
        crop, bounds, semantic_valid = dev.square_crop(frame, center, anchor["side"])
        scale = 1024.0 / max(frame.shape[:2])
        full_for_answerer = cv2.resize(
            frame,
            (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        stem = row["row_id"].replace("|", "__")
        full_relative = Path("media") / "cohort" / f"{stem}_full.png"
        crop_relative = Path("media") / "cohort" / f"{stem}_crop.png"
        full_ok, full_buffer = cv2.imencode(".png", full_for_answerer)
        crop_ok, crop_buffer = cv2.imencode(".png", crop)
        if not full_ok or not crop_ok:
            raise RuntimeError(f"PNG encoding failed: {row['row_id']}")
        full_sha = write_image_same_or_new(output_root / full_relative, full_buffer.tobytes())
        crop_sha = write_image_same_or_new(output_root / crop_relative, crop_buffer.tobytes())
        item = row | {
            "full_path": full_relative.as_posix(),
            "full_sha256": full_sha,
            "crop_path": crop_relative.as_posix(),
            "crop_sha256": crop_sha,
            "crop_bounds_xyxy": bounds,
            "semantic_valid": bool(semantic_valid),
            "input_condition": "full_native_crop",
            "full_source_resolution": [int(frame.shape[1]), int(frame.shape[0])],
            "full_materialized_resolution": [
                int(full_for_answerer.shape[1]),
                int(full_for_answerer.shape[0]),
            ],
            "materialization_wall_seconds": time.perf_counter() - row_started,
        }
        append_jsonl_fsync(journal, item)
        if index % 25 == 0 or index == len(planned):
            print(f"materialize {index}/{len(planned)}", flush=True)
    materialized = read_jsonl(journal)
    validate_materialized_prefix(output_root, planned, materialized)
    if len(materialized) != EXPECTED_VIEWS:
        raise RuntimeError("materialization is incomplete")
    summary = {
        "schema_version": "meva_w4_oneshot_materialization_v1",
        "events": EXPECTED_EVENTS,
        "candidate_views": EXPECTED_VIEWS,
        "semantic_valid": sum(row["semantic_valid"] for row in materialized),
        "semantic_invalid": sum(not row["semantic_valid"] for row in materialized),
        "candidate_manifest_sha256": sha256_file(journal),
        "cohort_manifest_sha256": EXPECTED_COHORT_SHA256,
        "registration_sha256": registration["_verified_sha256"],
        "full_frame_target": "aspect-preserving max-side 1024",
        "crop_target": "frozen native square crop",
        "wall_seconds": time.perf_counter() - stage_started,
        "gpu_jobs": 0,
    }
    atomic_write_json(output_root / "materialization_summary.json", summary)
    checkpoint = write_checkpoint(output_root, "materialize", summary)
    print(json.dumps({"status": "PASS", "checkpoint": str(checkpoint), **summary}, indent=2))
    return 0


REMOTE_INVENTORY_SCRIPT = r"""
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
request = json.load(sys.stdin)
sizes = {}
for relative in request["relative_paths"]:
    path = root / relative
    if path.is_file():
        sizes[relative] = path.stat().st_size
samples = {}
for relative in request["samples"]:
    path = root / relative
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        samples[relative] = digest.hexdigest()
json.dump({"sizes": sizes, "samples": samples}, sys.stdout, sort_keys=True)
"""


def remote_inventory(
    host: str, remote_root: str, relative_paths: list[str], samples: list[str]
) -> dict[str, Any]:
    command = f"python3 -c {shlex.quote(REMOTE_INVENTORY_SCRIPT)} {shlex.quote(remote_root)}"
    result = subprocess.run(
        ["ssh", host, command],
        input=json.dumps({"relative_paths": relative_paths, "samples": samples}),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def transfer(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    require_checkpoint(output_root, "materialize")
    rows = read_jsonl(output_root / "candidate_views.materialized.jsonl")
    if len(rows) != EXPECTED_VIEWS:
        raise RuntimeError("transfer requires 1056 materialized rows")
    relative_paths = ["candidate_views.materialized.jsonl"]
    relative_paths.extend(path for row in rows for path in (row["full_path"], row["crop_path"]))
    if any("\n" in path or "\r" in path or Path(path).is_absolute() for path in relative_paths):
        raise RuntimeError("unsafe transfer relative path")
    local_sizes = {path: (output_root / path).stat().st_size for path in relative_paths}
    sample_indices = sorted({0, len(relative_paths) // 2, len(relative_paths) - 1})
    samples = [relative_paths[index] for index in sample_indices]
    local_sample_sha = {path: sha256_file(output_root / path) for path in samples}
    before = remote_inventory(args.ssh_host, args.remote_root, relative_paths, samples)
    pending = [path for path in relative_paths if before["sizes"].get(path) != local_sizes[path]]
    stage_started = time.perf_counter()
    if pending:
        subprocess.run(["ssh", args.ssh_host, f"mkdir -p {shlex.quote(args.remote_root)}"], check=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", delete=False) as handle:
            file_list = Path(handle.name)
            for path in pending:
                handle.write(path + "\n")
        try:
            tar_process = subprocess.Popen(
                ["tar", "-C", str(output_root), "-cf", "-", "-T", str(file_list)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if tar_process.stdout is None:
                raise RuntimeError("tar stdout pipe was not created")
            ssh_process = subprocess.Popen(
                ["ssh", args.ssh_host, f"tar -xf - -C {shlex.quote(args.remote_root)}"],
                stdin=tar_process.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            tar_process.stdout.close()
            _, ssh_stderr = ssh_process.communicate()
            tar_stderr = tar_process.stderr.read() if tar_process.stderr is not None else b""
            tar_returncode = tar_process.wait()
            if tar_returncode != 0 or ssh_process.returncode != 0:
                raise RuntimeError(
                    f"tar|ssh failed: tar={tar_returncode} ssh={ssh_process.returncode} "
                    f"tar_stderr={tar_stderr.decode(errors='replace')} "
                    f"ssh_stderr={ssh_stderr.decode(errors='replace')}"
                )
        finally:
            file_list.unlink(missing_ok=True)
    after = remote_inventory(args.ssh_host, args.remote_root, relative_paths, samples)
    if after["sizes"] != local_sizes:
        raise RuntimeError("remote transfer count or byte-size verification failed")
    if after["samples"] != local_sample_sha:
        raise RuntimeError("remote transfer sample SHA verification failed")
    payload = {
        "ssh_host": args.ssh_host,
        "remote_root": args.remote_root,
        "frame_files": EXPECTED_VIEWS * 2,
        "transferred_files_this_run": len(pending),
        "verified_file_count": len(relative_paths),
        "verified_bytes": sum(local_sizes.values()),
        "sample_sha256": local_sample_sha,
        "identical_remote_copy_skipped": not pending,
        "wall_seconds": time.perf_counter() - stage_started,
    }
    checkpoint = write_checkpoint(output_root, "transfer", payload)
    print(json.dumps({"status": "PASS", "checkpoint": str(checkpoint), **payload}, indent=2))
    return 0


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError(f"cannot hash empty artifact tree: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def runtime_rows(output_root: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(output_root / "candidate_views.materialized.jsonl")
    if len(rows) != EXPECTED_VIEWS:
        raise RuntimeError("runtime materialized row count drift")
    result = []
    for row in rows:
        full_path = output_root / row["full_path"]
        crop_path = output_root / row["crop_path"]
        if not full_path.is_file() or not crop_path.is_file():
            raise FileNotFoundError(f"runtime media is absent: {row['row_id']}")
        result.append(row | {"full_path": str(full_path), "crop_path": str(crop_path)})
    return result


def validate_prefix(rows: list[dict[str, Any]], completed: list[dict[str, Any]], label: str) -> None:
    expected = [row["row_id"] for row in rows[: len(completed)]]
    observed = [row["row_id"] for row in completed]
    if observed != expected:
        raise RuntimeError(f"{label} checkpoint is not an ordered row prefix")


def load_feature_batches(
    features_root: Path, rows: list[dict[str, Any]], batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    images = np.empty((len(rows), 768), dtype=np.float32)
    dinov2 = np.empty((len(rows), 768), dtype=np.float32)
    for batch_index, start in enumerate(range(0, len(rows), batch_size)):
        stop = min(start + batch_size, len(rows))
        path = features_root / "siglip2_dinov2" / f"batch_{batch_index:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        cache = np.load(path, allow_pickle=True)
        expected_ids = [row["row_id"] for row in rows[start:stop]]
        if cache["row_ids"].tolist() != expected_ids:
            raise RuntimeError(f"feature batch row drift: {path}")
        batch_images = np.asarray(cache["image"], dtype=np.float32)
        batch_dinov2 = np.asarray(cache["dinov2_cls"], dtype=np.float32)
        if batch_images.shape != (stop - start, 768) or batch_dinov2.shape != (stop - start, 768):
            raise RuntimeError(f"feature batch shape drift: {path}")
        images[start:stop] = batch_images
        dinov2[start:stop] = batch_dinov2
    return images, dinov2


def features(args: argparse.Namespace) -> int:
    dev.require_host_conda_environment()
    output_root = args.output_root.resolve()
    preflight_checkpoint = require_checkpoint(output_root, "preflight")
    if preflight_checkpoint["head"] != git_output("rev-parse", "HEAD"):
        raise RuntimeError("preflight HEAD no longer matches checkout")
    if preflight_checkpoint["runner_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("preflight runner SHA no longer matches")
    features_root = output_root / "features_host"
    write_probe(features_root)
    rows = runtime_rows(output_root)
    plan_path = features_root / "feature_plan.json"
    plan = {
        "schema_version": "meva_w4_oneshot_feature_plan_v1",
        "rows": len(rows),
        "row_ids_sha256": hashlib.sha256(canonical_bytes([row["row_id"] for row in rows])).hexdigest(),
        "batch_size": args.batch_size,
        "models": {
            "grounding_dino": dev.DINO_MODEL_ID,
            "siglip2": dev.SIGLIP_MODEL_ID,
            "dinov2": dev.DINO_V2_MODEL_ID,
        },
        "dtype": "float32",
        "text_preprocessing": {"padding": "max_length", "max_length": 64, "truncation": True},
    }
    if plan_path.exists() and load_json(plan_path) != plan:
        raise RuntimeError("feature plan drift on resume")
    if not plan_path.exists():
        atomic_write_json(plan_path, plan)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for W4 feature extraction")
    dino_path = features_root / "dino.jsonl"
    completed_dino = read_jsonl(dino_path) if dino_path.exists() else []
    validate_prefix(rows, completed_dino, "Grounding DINO")
    if len(completed_dino) < len(rows):
        from mvevirag.detection.grounding_dino import GroundingDINODetector

        detector = GroundingDINODetector(model_id=dev.DINO_MODEL_ID, device=args.device)
        prompt = "person. helmet. motorcycle. scooter. bicycle. boundary line. fence. road."
        try:
            for index, row in enumerate(rows[len(completed_dino) :], len(completed_dino) + 1):
                wall_started = time.perf_counter()
                timer = dev.cuda_timer_start(torch)
                detections = detector.detect(row["crop_path"], prompt, threshold=0.3, text_threshold=0.25)
                gpu_seconds = dev.cuda_timer_stop(torch, timer)
                item = dev.dino_feature_row(row, detections) | {
                    "schema_version": "meva_w4_oneshot_grounding_dino_v1",
                    "gpu_seconds": gpu_seconds,
                    "wall_seconds": time.perf_counter() - wall_started,
                }
                append_jsonl_fsync(dino_path, item)
                if index % 25 == 0 or index == len(rows):
                    print(f"grounding-dino {index}/{len(rows)}", flush=True)
        finally:
            del detector
            torch.cuda.empty_cache()
    completed_dino = read_jsonl(dino_path)
    validate_prefix(rows, completed_dino, "Grounding DINO")
    if len(completed_dino) != len(rows):
        raise RuntimeError("Grounding DINO stage incomplete")

    from transformers import AutoImageProcessor, AutoModel

    from mvevirag.embedding.siglip2 import SigLIP2Encoder

    encoder = SigLIP2Encoder(model_id=dev.SIGLIP_MODEL_ID, device=args.device)
    dino_processor = AutoImageProcessor.from_pretrained(dev.DINO_V2_MODEL_ID, local_files_only=True)
    dino_model = (
        AutoModel.from_pretrained(dev.DINO_V2_MODEL_ID, local_files_only=True)
        .eval()
        .to(device=encoder.device, dtype=torch.float32)
    )
    text_path = features_root / "text_features.npz"
    questions = sorted({row["question"] for row in rows})
    try:
        if not text_path.exists():
            timer = dev.cuda_timer_start(torch)
            query = encoder.encode_batch_texts(questions).astype(np.float32, copy=False)
            text_gpu_seconds = dev.cuda_timer_stop(torch, timer)
            if query.shape != (len(questions), 768):
                raise RuntimeError("SigLIP2 text feature shape drift")
            atomic_write_npz(
                text_path,
                questions=np.asarray(questions, dtype=object),
                query=query,
                gpu_seconds=np.asarray(text_gpu_seconds, dtype=np.float64),
                padding=np.asarray("max_length"),
                max_length=np.asarray(64),
                truncation=np.asarray(True),
            )
        else:
            text_cache = np.load(text_path, allow_pickle=True)
            if text_cache["questions"].tolist() != questions or int(text_cache["max_length"]) != 64:
                raise RuntimeError("SigLIP2 text checkpoint drift")
        batch_root = features_root / "siglip2_dinov2"
        batch_root.mkdir(parents=True, exist_ok=True)
        for batch_index, start in enumerate(range(0, len(rows), args.batch_size)):
            stop = min(start + args.batch_size, len(rows))
            batch_path = batch_root / f"batch_{batch_index:04d}.npz"
            expected_ids = [row["row_id"] for row in rows[start:stop]]
            if batch_path.exists():
                cache = np.load(batch_path, allow_pickle=True)
                if cache["row_ids"].tolist() != expected_ids:
                    raise RuntimeError(f"feature batch checkpoint drift: {batch_path}")
                continue
            paths = [Path(row["crop_path"]) for row in rows[start:stop]]
            wall_started = time.perf_counter()
            timer = dev.cuda_timer_start(torch)
            image = encoder.encode_batch_images(paths, batch_size=args.batch_size).astype(
                np.float32, copy=False
            )
            siglip_gpu_seconds = dev.cuda_timer_stop(torch, timer)
            timer = dev.cuda_timer_start(torch)
            dinov2_cls = dev.encode_dinov2_images(
                paths,
                processor=dino_processor,
                model=dino_model,
                device=encoder.device,
                torch=torch,
                batch_size=args.batch_size,
            )
            dinov2_gpu_seconds = dev.cuda_timer_stop(torch, timer)
            if image.shape != (stop - start, 768) or dinov2_cls.shape != (stop - start, 768):
                raise RuntimeError("SigLIP2/DINOv2 feature shape drift")
            atomic_write_npz(
                batch_path,
                row_ids=np.asarray(expected_ids, dtype=object),
                image=image,
                dinov2_cls=dinov2_cls,
                siglip_gpu_seconds=np.asarray(siglip_gpu_seconds, dtype=np.float64),
                dinov2_gpu_seconds=np.asarray(dinov2_gpu_seconds, dtype=np.float64),
                wall_seconds=np.asarray(time.perf_counter() - wall_started, dtype=np.float64),
            )
            print(f"siglip2-dinov2 {stop}/{len(rows)}", flush=True)
    finally:
        del dino_model
        del encoder
        torch.cuda.empty_cache()

    image, dinov2_cls = load_feature_batches(features_root, rows, args.batch_size)
    scaler, classifier = dev.load_frozen_availability_model(args.availability_model.resolve())
    m4_path = features_root / "m4_availability_scores.jsonl"
    completed_m4 = read_jsonl(m4_path) if m4_path.exists() else []
    validate_prefix(rows, completed_m4, "M4")
    for start in range(len(completed_m4), len(rows), args.batch_size):
        stop = min(start + args.batch_size, len(rows))
        matrix = np.concatenate([image[start:stop], dinov2_cls[start:stop]], axis=1).astype(
            np.float32, copy=False
        )
        scores = classifier.predict_proba(scaler.transform(matrix))[:, 1]
        for row, score in zip(rows[start:stop], scores, strict=True):
            append_jsonl_fsync(
                m4_path,
                {
                    "schema_version": "meva_w4_oneshot_m4_availability_v1",
                    "row_id": row["row_id"],
                    "availability_score": float(score),
                },
            )
        print(f"m4 {stop}/{len(rows)}", flush=True)
    completed_m4 = read_jsonl(m4_path)
    validate_prefix(rows, completed_m4, "M4")
    if len(completed_m4) != len(rows):
        raise RuntimeError("M4 stage incomplete")
    batch_paths = sorted((features_root / "siglip2_dinov2").glob("batch_*.npz"))
    payload = {
        "candidate_views": len(rows),
        "grounding_dino_rows": len(completed_dino),
        "m4_rows": len(completed_m4),
        "grounding_dino_sha256": sha256_file(dino_path),
        "text_features_sha256": sha256_file(text_path),
        "siglip2_dinov2_tree_sha256": sha256_tree(features_root / "siglip2_dinov2"),
        "m4_sha256": sha256_file(m4_path),
        "batch_files": len(batch_paths),
        "availability_model_sha256": sha256_file(args.availability_model),
        "grounding_dino_gpu_seconds": sum(float(row["gpu_seconds"]) for row in completed_dino),
        "siglip2_image_gpu_seconds": sum(float(np.load(path)["siglip_gpu_seconds"]) for path in batch_paths),
        "dinov2_gpu_seconds": sum(float(np.load(path)["dinov2_gpu_seconds"]) for path in batch_paths),
        "siglip2_text_gpu_seconds": float(np.load(text_path, allow_pickle=True)["gpu_seconds"]),
    }
    checkpoint = write_checkpoint(output_root, "features", payload)
    print(json.dumps({"status": "PASS", "checkpoint": str(checkpoint), **payload}, indent=2))
    return 0


def answerer(args: argparse.Namespace) -> int:
    from scripts.operator.w3w5a_common import MODEL_ID
    from scripts.operator.w5a_fair_baselines_runner import EXPLICIT_SYSTEM_PROMPT

    output_root = args.output_root.resolve()
    require_checkpoint(output_root, "preflight")
    rows = runtime_rows(output_root)
    if args.arm not in ARMS:
        raise ValueError(f"unsupported W4 answerer arm: {args.arm}")
    if args.physical_gpu != EXPECTED_GPU[args.arm]:
        raise RuntimeError(
            f"frozen GPU assignment drift for {args.arm}: {args.physical_gpu} != {EXPECTED_GPU[args.arm]}"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.physical_gpu):
        raise RuntimeError(
            f"set CUDA_VISIBLE_DEVICES={args.physical_gpu} for {args.arm}; observed {visible!r}"
        )
    adapter: Path | None = None
    if args.arm == PRIMARY_ARM:
        if args.answer_abstain_adapter is None:
            raise RuntimeError("answer_abstain_sft requires --answer-abstain-adapter")
        adapter = args.answer_abstain_adapter.resolve()
        if not adapter.is_dir():
            raise FileNotFoundError(adapter)
    elif args.answer_abstain_adapter is not None:
        raise RuntimeError("explicit_abstention_prompt must not receive a LoRA adapter")
    output_path = output_root / "answers" / f"{args.arm}.jsonl"
    completed = read_jsonl(output_path) if output_path.exists() else []
    validate_prefix(rows, completed, args.arm)
    if any(row.get("arm") != args.arm for row in completed):
        raise RuntimeError(f"answerer checkpoint contains another arm: {args.arm}")
    plan = {
        "schema_version": "meva_w4_oneshot_answerer_plan_v1",
        "arm": args.arm,
        "physical_gpu": args.physical_gpu,
        "candidate_views": len(rows),
        "model": MODEL_ID,
        "adapter_tree_sha256": sha256_tree(adapter) if adapter is not None else None,
        "system_prompt_sha256": hashlib.sha256(EXPLICIT_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "release_threshold": RELEASE_THRESHOLDS[args.arm],
        "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 32, "n": 1, "seed": SEED},
    }
    plan_path = output_root / "answers" / f"{args.arm}.plan.json"
    if plan_path.exists() and load_json(plan_path) != plan:
        raise RuntimeError(f"answerer plan drift on resume: {args.arm}")
    if completed and not plan_path.exists():
        raise RuntimeError(f"answerer rows exist without their frozen plan: {args.arm}")
    if len(completed) == len(rows):
        payload = {
            "arm": args.arm,
            "physical_gpu": args.physical_gpu,
            "candidate_views": len(completed),
            "release_threshold": RELEASE_THRESHOLDS[args.arm],
            "answers_sha256": sha256_file(output_path),
            "vlm_gpu_seconds": sum(float(row["vlm_gpu_seconds"]) for row in completed),
            "vlm_wall_seconds": sum(float(row["vlm_wall_seconds"]) for row in completed),
            "adapter_tree_sha256": sha256_tree(adapter) if adapter is not None else None,
        }
        checkpoint = write_checkpoint(output_root, f"answerer.{args.arm}", payload)
        print(json.dumps({"status": "PASS", "checkpoint": str(checkpoint), **payload}, indent=2))
        return 0

    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from scripts.operator import answerer_alignment_lora as base

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("answerer arm requires exactly one visible CUDA device")
    if not plan_path.exists():
        atomic_write_json(plan_path, plan)
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
        seed=SEED,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32, n=1, seed=SEED, logprobs=1)
    request = LoRARequest(PRIMARY_ARM, 1, lora_path=str(adapter)) if adapter is not None else None
    pending = rows[len(completed) :]
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        prompts: list[dict[str, Any]] = []
        opened: list[Any] = []
        for row in batch:
            images = base.row_images(row)
            opened.extend(images)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": EXPLICIT_SYSTEM_PROMPT}]},
                {"role": "user", "content": base.user_content(row)},
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append({"prompt": prompt, "multi_modal_data": {"image": images}})
        try:
            wall_started = time.perf_counter()
            timer = dev.cuda_timer_start(torch)
            generated = llm.generate(prompts, sampling_params=sampling, lora_request=request, use_tqdm=False)
            gpu_seconds = dev.cuda_timer_stop(torch, timer)
            wall_seconds = time.perf_counter() - wall_started
        finally:
            for image in opened:
                image.close()
        per_row_gpu = gpu_seconds / len(batch)
        per_row_wall = wall_seconds / len(batch)
        for row, output in zip(batch, generated, strict=True):
            candidate = output.outputs[0]
            if not candidate.token_ids:
                raise RuntimeError(f"empty generation: {args.arm}/{row['row_id']}")
            decision, answer, abstained = base.parse_answer(candidate.text.strip())
            correct = (
                base.positive_correct(answer, decision, abstained, row["target_answer"])
                if row["role"] == "positive"
                else abstained
            )
            append_jsonl_fsync(
                output_path,
                {
                    "schema_version": "meva_w4_oneshot_answer_v1",
                    "row_id": row["row_id"],
                    "event_id": row["event_id"],
                    "camera_id": row["camera_id"],
                    "role": row["role"],
                    "arm": args.arm,
                    "decision": decision,
                    "answer": answer,
                    "was_abstained": bool(abstained),
                    "target_correct": bool(correct),
                    "confidence": math.exp(candidate.cumulative_logprob / len(candidate.token_ids)),
                    "raw_response": candidate.text.strip(),
                    "vlm_gpu_seconds": per_row_gpu,
                    "vlm_wall_seconds": per_row_wall,
                },
            )
        completed_count = len(completed) + min(offset + len(batch), len(pending))
        print(f"{args.arm} {completed_count}/{len(rows)}", flush=True)
    completed = read_jsonl(output_path)
    validate_prefix(rows, completed, args.arm)
    if len(completed) != len(rows):
        raise RuntimeError(f"answerer arm incomplete: {args.arm}")
    payload = {
        "arm": args.arm,
        "physical_gpu": args.physical_gpu,
        "candidate_views": len(completed),
        "release_threshold": RELEASE_THRESHOLDS[args.arm],
        "answers_sha256": sha256_file(output_path),
        "vlm_gpu_seconds": sum(float(row["vlm_gpu_seconds"]) for row in completed),
        "vlm_wall_seconds": sum(float(row["vlm_wall_seconds"]) for row in completed),
        "adapter_tree_sha256": sha256_tree(adapter) if adapter is not None else None,
    }
    checkpoint = write_checkpoint(output_root, f"answerer.{args.arm}", payload)
    print(json.dumps({"status": "PASS", "checkpoint": str(checkpoint), **payload}, indent=2))
    return 0


def load_replay_rows(output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(output_root / "candidate_views.materialized.jsonl")
    if len(rows) != EXPECTED_VIEWS:
        raise RuntimeError("replay materialized row count drift")
    features_checkpoint = require_checkpoint(output_root, "features")
    feature_plan = load_json(output_root / "features_host" / "feature_plan.json")
    batch_size = int(feature_plan["batch_size"])
    image, dinov2_cls = load_feature_batches(output_root / "features_host", rows, batch_size)
    text_cache = np.load(output_root / "features_host" / "text_features.npz", allow_pickle=True)
    questions = text_cache["questions"].tolist()
    query_map = {
        question: np.asarray(text_cache["query"][index], dtype=np.float32)
        for index, question in enumerate(questions)
    }
    query = np.stack([query_map[row["question"]] for row in rows])
    dino = {row["row_id"]: row for row in read_jsonl(output_root / "features_host" / "dino.jsonl")}
    m4 = {
        row["row_id"]: float(row["availability_score"])
        for row in read_jsonl(output_root / "features_host" / "m4_availability_scores.jsonl")
    }
    answers = {
        arm: {row["row_id"]: row for row in read_jsonl(output_root / "answers" / f"{arm}.jsonl")}
        for arm in ARMS
    }
    row_id_set = {row["row_id"] for row in rows}
    if (
        set(dino) != row_id_set
        or set(m4) != row_id_set
        or any(set(table) != row_id_set for table in answers.values())
    ):
        raise RuntimeError("replay input row coverage drift")
    dino_matrix = np.asarray(
        [[float(dino[row["row_id"]][key]) for key in dev.DINO_KEYS] for row in rows], dtype=np.float64
    )
    combined = np.concatenate(
        [
            image.astype(np.float64),
            query.astype(np.float64),
            image * query,
            np.abs(image - query),
            dino_matrix,
        ],
        axis=1,
    )
    dev_selection = load_json(DEFAULT_DEV_SELECTION)
    task = logistic_scores(combined, dev_selection["task_selector"])
    semantic = dev.semantic_scores(
        image,
        query,
        np.asarray([row["semantic_valid"] is True for row in rows]),
    )
    enriched: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        arm_metrics: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            answer = answers[arm][row["row_id"]]
            released = (
                not bool(answer["was_abstained"]) and float(answer["confidence"]) >= RELEASE_THRESHOLDS[arm]
            )
            arm_metrics[arm] = {
                "released": bool(released),
                "correct_release": float(
                    row["role"] == "positive" and released and bool(answer["target_correct"])
                ),
                "violation_release": float(row["role"] != "positive" and released),
                "vlm_gpu_seconds": float(answer["vlm_gpu_seconds"]),
                "vlm_wall_seconds": float(answer["vlm_wall_seconds"]),
            }
        enriched.append(
            row
            | {
                "person_count": float(dino[row["row_id"]]["fresh_dino_person_regions"]),
                "person_area": float(dino[row["row_id"]]["person_area_sum"]),
                "semantic_score": float(semantic[index]),
                "task_matched_score": float(task[index]),
                "availability_score": m4[row["row_id"]],
                "image_embedding": image[index],
                "dinov2_embedding": dinov2_cls[index],
                "arm_metrics": arm_metrics,
            }
        )
    return enriched, features_checkpoint


def policy_replay(
    rows: list[dict[str, Any]], features_checkpoint: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, tuple[float, float, float]]]]:
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_event[row["event_id"]].append(row)
    if len(by_event) != EXPECTED_EVENTS:
        raise RuntimeError("replay event count drift")
    router_cost = {
        "deterministic_random_exact_expectation": 0.0,
        "person_count_area": float(features_checkpoint["grounding_dino_gpu_seconds"]),
        "siglip_semantic_relevance": float(features_checkpoint["siglip2_image_gpu_seconds"])
        + float(features_checkpoint["siglip2_text_gpu_seconds"]),
        "task_matched_selector": float(features_checkpoint["grounding_dino_gpu_seconds"])
        + float(features_checkpoint["siglip2_image_gpu_seconds"])
        + float(features_checkpoint["siglip2_text_gpu_seconds"]),
        "availability": float(features_checkpoint["siglip2_image_gpu_seconds"])
        + float(features_checkpoint["dinov2_gpu_seconds"]),
        "availability_plus_diversity": float(features_checkpoint["siglip2_image_gpu_seconds"])
        + float(features_checkpoint["dinov2_gpu_seconds"]),
    }
    event_k1: dict[str, dict[str, tuple[float, float, float]]] = {arm: {} for arm in ARMS}
    results: dict[str, Any] = {}
    for policy in POLICY_ORDER:
        curves: dict[str, Any] = {}
        for k in range(1, 7):
            eligible = [
                (event_id, values) for event_id, values in sorted(by_event.items()) if len(values) >= k
            ]
            if not eligible:
                continue
            by_arm: dict[str, Any] = {}
            for arm in ARMS:
                outcomes: list[tuple[float, float, float]] = []
                for event_id, values in eligible:
                    if policy == "deterministic_random_exact_expectation":
                        outcome = exact_random_event(values, k, arm)
                    else:
                        outcome = subset_outcome(deterministic_policy_order(policy, values)[:k], arm)
                    outcomes.append(outcome)
                    if k == 1:
                        event_k1[arm].setdefault(policy, {})[event_id] = outcome
                selected_vlm = float(sum(outcome[2] for outcome in outcomes))
                by_arm[arm] = {
                    "events": len(outcomes),
                    "U": float(np.mean([outcome[0] for outcome in outcomes])),
                    "V": float(np.mean([outcome[1] for outcome in outcomes])),
                    "router_gpu_seconds_all_1056_views": router_cost[policy],
                    "selected_vlm_gpu_seconds": selected_vlm,
                    "total_gpu_seconds": router_cost[policy] + selected_vlm,
                }
            curves[str(k)] = by_arm
        results[policy] = {"curves": curves}
    return results, event_k1


def role_release_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for arm in ARMS:
        arm_table: dict[str, Any] = {}
        for role in ("positive", "negative_geometric"):
            selected = [row for row in rows if row["role"] == role]
            released = sum(row["arm_metrics"][arm]["released"] for row in selected)
            correct = sum(row["arm_metrics"][arm]["correct_release"] for row in selected)
            violations = sum(row["arm_metrics"][arm]["violation_release"] for row in selected)
            arm_table[role] = {
                "views": len(selected),
                "released": int(released),
                "release_rate": float(released / len(selected)),
                "correct_released": int(correct),
                "correct_released_yield": float(correct / len(selected)),
                "violation_released": int(violations),
                "violation_rate": float(violations / len(selected)),
            }
        table[arm] = arm_table
    return table


def replay(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    for arm in ARMS:
        require_checkpoint(output_root, f"answerer.{arm}")
    rows, features_checkpoint = load_replay_rows(output_root)
    policy_results, event_k1 = policy_replay(rows, features_checkpoint)
    event_ids = sorted({row["event_id"] for row in rows})
    focal_u = [event_k1[PRIMARY_ARM][FOCAL_POLICY][event_id][0] for event_id in event_ids]
    comparator_u = [event_k1[PRIMARY_ARM][COMPARATOR_POLICY][event_id][0] for event_id in event_ids]
    focal_v = [event_k1[PRIMARY_ARM][FOCAL_POLICY][event_id][1] for event_id in event_ids]
    comparator_v = [event_k1[PRIMARY_ARM][COMPARATOR_POLICY][event_id][1] for event_id in event_ids]
    h1_u = paired_bootstrap(focal_u, comparator_u)
    h1_v = paired_bootstrap(focal_v, comparator_v)
    focal_cost = policy_results[FOCAL_POLICY]["curves"]["1"][PRIMARY_ARM]["total_gpu_seconds"]
    all_camera_cost = sum(row["arm_metrics"][PRIMARY_ARM]["vlm_gpu_seconds"] for row in rows)
    h1 = {
        "estimand": f"{FOCAL_POLICY} minus {COMPARATOR_POLICY}",
        "events": len(event_ids),
        "delta_u": h1_u["delta"],
        "delta_u_two_sided_95_ci": h1_u["two_sided_95_ci"],
        "delta_v": h1_v["delta"],
        "delta_v_one_sided_95_upper": h1_v["one_sided_95_upper"],
        "effect_pass": h1_u["delta"] >= 0.10,
        "ci_pass": h1_u["two_sided_95_ci"][0] > 0.0,
        "safety_pass": h1_v["one_sided_95_upper"] < 0.05,
        "focal_total_gpu_seconds": focal_cost,
        "all_camera_measured_vlm_gpu_seconds": all_camera_cost,
        "cost_pass": focal_cost < all_camera_cost,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": SEED,
    }
    h1["pass"] = h1["effect_pass"] and h1["ci_pass"] and h1["safety_pass"] and h1["cost_pass"]
    primary_yield = [event_k1[PRIMARY_ARM][FOCAL_POLICY][event_id][0] for event_id in event_ids]
    control_yield = [event_k1[CONTROL_ARM][FOCAL_POLICY][event_id][0] for event_id in event_ids]
    primary_v = [event_k1[PRIMARY_ARM][FOCAL_POLICY][event_id][1] for event_id in event_ids]
    control_v = [event_k1[CONTROL_ARM][FOCAL_POLICY][event_id][1] for event_id in event_ids]
    h2_yield = paired_bootstrap(primary_yield, control_yield)
    h2_v = paired_bootstrap(primary_v, control_v)
    h2 = {
        "estimand": f"{PRIMARY_ARM} minus {CONTROL_ARM} under {FOCAL_POLICY} U(1)",
        "events": len(event_ids),
        "delta_yield": h2_yield["delta"],
        "delta_yield_two_sided_95_ci": h2_yield["two_sided_95_ci"],
        "delta_v": h2_v["delta"],
        "delta_v_one_sided_95_upper": h2_v["one_sided_95_upper"],
        "point_pass": h2_yield["delta"] > 0.0,
        "ci_pass": h2_yield["two_sided_95_ci"][0] > 0.0,
        "safety_pass": h2_v["one_sided_95_upper"] < 0.05,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": SEED,
    }
    h2["pass"] = h2["point_pass"] and h2["ci_pass"] and h2["safety_pass"]
    result = {
        "schema_version": "meva_w4_oneshot_result_v1",
        "generated_at_utc": utc_now(),
        "boundary": "single sealed new cohort; rule-supported correct released yield only",
        "cohort": {
            "events": len(event_ids),
            "candidate_views": len(rows),
            "n_cand_ge_3": len({row["event_id"] for row in rows if row["n_cand"] >= 3}),
        },
        "policy_order": list(POLICY_ORDER),
        "policy_results": policy_results,
        "H1": h1,
        "H2": h2,
        "role_release_table": role_release_table(rows),
        "branch_color": branch_color(h1, h2),
        "claim_boundary": {
            "U_k_gt_1": "secondary cumulative recovery-versus-cost curve; not a fusion estimand",
            "human_semantic_sufficiency": "not_claimed",
            "rerun_after_result": "prohibited",
        },
    }
    result_path = output_root / "w4_results.json"
    result_sha = atomic_write_json(result_path, result)
    payload = {
        "result_path": str(result_path),
        "result_sha256": result_sha,
        "branch_color": result["branch_color"],
    }
    checkpoint = write_checkpoint(output_root, "replay", payload)
    print(json.dumps({"status": "PASS", "checkpoint": str(checkpoint), **payload}, indent=2))
    return 0


def seal(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    replay_checkpoint = require_checkpoint(output_root, "replay")
    result_path = output_root / "w4_results.json"
    if sha256_file(result_path) != replay_checkpoint["result_sha256"]:
        raise RuntimeError("result SHA drift before seal")
    seal_root = output_root / "seal"
    data_files = sorted(
        path for path in output_root.rglob("*") if path.is_file() and seal_root not in path.parents
    )
    artifact_manifest = {
        "schema_version": "meva_w4_oneshot_artifact_sha_manifest_v1",
        "generated_at_utc": utc_now(),
        "files": [
            {
                "path": path.relative_to(output_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in data_files
        ],
    }
    manifest_path = seal_root / "artifact_sha_manifest.json"
    manifest_sha = atomic_write_json(manifest_path, artifact_manifest)
    registration_sha = verify_sidecar(args.registration)
    result_diff = {
        "schema_version": "meva_w4_registration_result_hash_diff_v1",
        "registration_path": str(args.registration.relative_to(ROOT)).replace("\\", "/"),
        "registration_sha256_before_result_link": registration_sha,
        "result_path": str(result_path.relative_to(ROOT)).replace("\\", "/"),
        "result_sha256": sha256_file(result_path),
        "artifact_manifest_path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "artifact_manifest_sha256": manifest_sha,
        "prepared_json_patch": [
            {
                "op": "add",
                "path": "/sealed_result",
                "value": {
                    "path": str(result_path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256_file(result_path),
                    "artifact_manifest_sha256": manifest_sha,
                },
            }
        ],
        "registration_modified": False,
    }
    diff_path = seal_root / "registration_result_hash_diff.json"
    diff_sha = atomic_write_json(diff_path, result_diff)
    payload = {
        "artifact_files": len(data_files),
        "artifact_manifest_sha256": manifest_sha,
        "result_sha256": sha256_file(result_path),
        "registration_sha256_before_result_link": registration_sha,
        "registration_result_hash_diff_sha256": diff_sha,
    }
    checkpoint = write_checkpoint(output_root, "seal", payload)
    print(json.dumps({"status": "PASS", "checkpoint": str(checkpoint), **payload}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def sealed_options(item: argparse.ArgumentParser) -> None:
        item.add_argument("--cohort-manifest", type=Path, default=DEFAULT_COHORT_MANIFEST)
        item.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
        item.add_argument("--runtime-contract", type=Path, default=DEFAULT_CONTRACT)
        item.add_argument("--dev-selection", type=Path, default=DEFAULT_DEV_SELECTION)
        item.add_argument("--w5a-selection", type=Path, default=DEFAULT_W5A_SELECTION)

    dry = subparsers.add_parser("dry-run")
    sealed_options(dry)
    dry.add_argument("--dev-root", type=Path, default=DEFAULT_DEV_ROOT)
    dry.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    preflight_parser = subparsers.add_parser("preflight")
    sealed_options(preflight_parser)
    preflight_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    preflight_parser.add_argument("--availability-model", type=Path, default=DEFAULT_AVAILABILITY_MODEL)
    preflight_parser.add_argument("--device", default="cpu")

    materialize_parser = subparsers.add_parser("materialize")
    sealed_options(materialize_parser)
    materialize_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    materialize_parser.add_argument("--video-root", type=Path, required=True)
    materialize_parser.add_argument(
        "--meva-repo",
        type=Path,
        default=Path.home()
        / "AppData"
        / "Local"
        / "Temp"
        / "mvevirag_meva_pilot_20260713"
        / "meva-data-repo",
    )

    transfer_parser = subparsers.add_parser("transfer")
    transfer_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    transfer_parser.add_argument("--ssh-host", required=True)
    transfer_parser.add_argument("--remote-root", required=True)

    features_parser = subparsers.add_parser("features")
    features_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    features_parser.add_argument("--availability-model", type=Path, default=DEFAULT_AVAILABILITY_MODEL)
    features_parser.add_argument("--device", default="cuda:2")
    features_parser.add_argument("--batch-size", type=int, default=16)

    answerer_parser = subparsers.add_parser("answerer")
    answerer_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    answerer_parser.add_argument("--arm", choices=ARMS, required=True)
    answerer_parser.add_argument("--physical-gpu", type=int, choices=(0, 1), required=True)
    answerer_parser.add_argument("--answer-abstain-adapter", type=Path)
    answerer_parser.add_argument("--batch-size", type=int, default=8)

    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    seal_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    commands = {
        "dry-run": dry_run,
        "preflight": preflight,
        "materialize": materialize,
        "transfer": transfer,
        "features": features,
        "answerer": answerer,
        "replay": replay,
        "seal": seal,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
