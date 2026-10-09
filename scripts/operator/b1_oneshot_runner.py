"""Execute the registered B1 cohort without aggregating opened outcomes.

Only ``dry-run`` is permitted before a clean, synced execution commit receives
separate user approval.  Live stages reuse the frozen W4 oneshot materializer,
feature extractor, and vLLM answerer implementation through a narrow
compatibility layer.  This file intentionally has no aggregate/replay command:
configuration sealing and raw-output sealing are kept separate from post-open
analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPECTED_EVENTS = 260
EXPECTED_VIEWS = 1096
EXPECTED_PRIOR_CLIPS = 641
EXPECTED_RESERVE_EVENTS = 454
DECODING_SEED = 20260713
MAX_STAGE_ATTEMPTS = 3
MIN_FREE_BYTES = 80 * 1024**3
MODEL_MODULES = ("torch", "transformers", "vllm", "peft")

DEFAULT_ROOT = ROOT / "outputs" / "b1_prereg_20260805"
DEFAULT_MANIFEST = DEFAULT_ROOT / "b1_cohort_manifest.json"
DEFAULT_RESERVE = DEFAULT_ROOT / "b1_reserve.json"
DEFAULT_EXCLUSION = DEFAULT_ROOT / "exclusion_clips.txt"
DEFAULT_THRESHOLDS = DEFAULT_ROOT / "b1_thresholds.json"
DEFAULT_REGISTRATION = DEFAULT_ROOT / "b1_registration.json"
DEFAULT_PINS = ROOT / "outputs" / "b1_prereg_pins_20260805" / "pins.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "b1_oneshot_execution"
W4_RUNNER = ROOT / "scripts" / "w4_oneshot_runner.py"

GENERATION_PASSES = (
    "answer_abstain_seed_20260713",
    "answer_abstain_seed_20260714",
    "answer_abstain_seed_20260715",
    "answer_only_seed_20260713",
    "answer_only_seed_20260714",
    "answer_only_seed_20260715",
    "explicit_abstention_prompt",
    "zero_shot",
)
HEAD_PASSES = ("correctness_selector", "action_classification_head")
GPU_WAVES = (
    {"wave": 1, "gpu0": GENERATION_PASSES[0], "gpu1": GENERATION_PASSES[1], "gpu2": GENERATION_PASSES[2]},
    {"wave": 2, "gpu0": GENERATION_PASSES[3], "gpu1": GENERATION_PASSES[4], "gpu2": GENERATION_PASSES[5]},
    {"wave": 3, "gpu0": "explicit_abstention_prompt", "gpu1": "zero_shot", "gpu2": "idle"},
    {"wave": 4, "gpu0": "correctness_selector", "gpu1": "action_classification_head", "gpu2": "idle"},
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    if not files:
        raise RuntimeError(f"cannot hash empty tree: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(item).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write(path: Path, payload: bytes) -> str:
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
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> str:
    return atomic_write(path, canonical_bytes(value))


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def require_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{label} SHA-256 drift: {observed} != {expected}")


def normalized_clip(value: str) -> str:
    result = Path(value).name
    return result[: -len(".r13.avi")] if result.endswith(".r13.avi") else result


def site_slot(clip: str) -> str:
    fields = normalized_clip(clip).split(".")
    date, start, site = fields[0], fields[1], fields[-2]
    hour, minute, _ = map(int, start.split("-"))
    return f"{site}|{date}.{hour:02d}-{(minute // 5) * 5:02d}-00"


def manifest_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    from scripts.operator.w3w5a_common import ABSTAIN, ACTIVITY_TEXT

    if manifest.get("schema_version") != "meva_b1_clip_slot_cohort_manifest_v1":
        raise RuntimeError("B1 manifest schema drift")
    events = manifest.get("events")
    if not isinstance(events, list) or len(events) != EXPECTED_EVENTS:
        raise RuntimeError("B1 manifest event count drift")
    rows: list[dict[str, Any]] = []
    for event in events:
        views = event["views"]
        positives = [view for view in views if view["role"] == "positive"]
        if len(positives) != 1 or len(views) != int(event["n_cand"]):
            raise RuntimeError(f"B1 event structure drift: {event['event_id']}")
        positive = positives[0]
        question, affirmative = ACTIVITY_TEXT[event["activity"]]
        common = {
            "event_id": event["event_id"],
            "group_id": event["event_id"],
            "split": "b1_sealed_cohort",
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
            "proxy_strata": event["proxy_strata"],
        }
        for view in views:
            role = view["role"]
            row = common | {
                "view_id": view["view_id"],
                "row_id": view["view_id"],
                "sample_id": view["view_id"],
                "camera_id": view["camera_id"],
                "role": role,
                "evaluation_role": "pplus" if role == "positive" else "eminus",
                "label": int(role == "positive"),
                "clip": view["clip"],
                "native_frame": int(view["native_frame"]),
                "model_file": view["model_file"],
                "target_answer": affirmative if role == "positive" else ABSTAIN,
                "target_type": "answer" if role == "positive" else "abstain",
                "sealed_view_spec_sha256": view["view_spec_sha256"],
                "source_video": view["s3"],
                "content_pin": view["content_pin"],
            }
            if role != "positive":
                row["minimum_outside_clearance_deg_beyond_buffer"] = float(
                    view["minimum_outside_clearance_deg_beyond_buffer"]
                )
            rows.append(row)
    rows.sort(key=lambda row: (row["event_id"], row["camera_id"]))
    if len(rows) != EXPECTED_VIEWS or len({row["row_id"] for row in rows}) != EXPECTED_VIEWS:
        raise RuntimeError("B1 row coverage drift")
    return rows


def load_inputs(args: argparse.Namespace, registration_required: bool) -> dict[str, Any]:
    inputs = {
        "manifest": load_json(args.manifest),
        "reserve": load_json(args.reserve),
        "thresholds": load_json(args.thresholds),
        "pins": load_json(args.pins),
    }
    inputs["rows"] = manifest_rows(inputs["manifest"])
    if args.registration.is_file():
        inputs["registration"] = load_json(args.registration)
    elif registration_required:
        raise FileNotFoundError(args.registration)
    else:
        inputs["registration"] = None
    return inputs


def file_sha_map(args: argparse.Namespace) -> dict[str, str]:
    paths = (args.manifest, args.reserve, args.exclusion, args.thresholds, args.pins, Path(__file__))
    return {
        str(path.resolve().relative_to(ROOT)).replace("\\", "/"): sha256_file(path.resolve())
        for path in paths
    }


def generation_contract(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checkpoints = inputs["pins"]["P1_seed_adapter_reuse"]["checkpoints"]
    by_key = {(int(row["seed"]), row["arm"]): row for row in checkpoints}
    thresholds = inputs["thresholds"]["thresholds"]
    fixed_thresholds = inputs["thresholds"]["fixed_policy_thresholds"]
    result: dict[str, dict[str, Any]] = {}
    for b1_arm, source_arm in (
        ("answer_abstain", "standard_alignment"),
        ("answer_only", "answer_only_sft"),
    ):
        for seed in (20260713, 20260714, 20260715):
            pass_id = f"{b1_arm}_seed_{seed}"
            checkpoint = by_key[(seed, source_arm)]
            result[pass_id] = {
                "kind": "adapter_generation",
                "seed": seed,
                "source_arm": source_arm,
                "checkpoint_path": checkpoint["remote_checkpoint_path"],
                "checkpoint_tree_sha256": checkpoint["tree_sha256"],
                "adapter_model_sha256": checkpoint["adapter_model_sha256"],
                "tau": thresholds[str(seed)][b1_arm]["tau"],
                "release_enabled": thresholds[str(seed)][b1_arm]["release_enabled"],
                "system_prompt": "explicit_evidence_abstention_prompt",
                "decoding_seed": DECODING_SEED,
            }
    result["explicit_abstention_prompt"] = {
        "kind": "base_generation",
        "checkpoint_path": None,
        "tau": fixed_thresholds["explicit_abstention_prompt"]["tau"],
        "release_enabled": fixed_thresholds["explicit_abstention_prompt"]["release_enabled"],
        "system_prompt": "explicit_evidence_abstention_prompt",
        "decoding_seed": DECODING_SEED,
    }
    result["zero_shot"] = {
        "kind": "base_generation",
        "checkpoint_path": None,
        "record_only": True,
        "system_prompt": "neutral_evidence_only_prompt",
        "decoding_seed": DECODING_SEED,
    }
    return result


def head_contract(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return inputs["pins"]["P3_same_supervision_heads"]["arms"]


def execution_plan() -> dict[str, Any]:
    return {
        "stages": [
            "preflight",
            "seal-config",
            "materialize",
            "features",
            "generation waves 1-3",
            "head wave 4",
            "seal-raw",
        ],
        "gpu_waves": GPU_WAVES,
        "retry": {
            "maximum_attempts_per_fixed_stage_or_pass": MAX_STAGE_ATTEMPTS,
            "semantics": "same command, config, prompt, seed, checkpoint, and ordered row prefix only",
            "exhausted_or_unresolved": "record failure and force non-release; no substitute inference",
        },
        "aggregation": "not implemented as a command in this runner; post-open only",
    }


def dry_run(args: argparse.Namespace) -> int:
    before_modules = {name for name in MODEL_MODULES if name in sys.modules}
    inputs = load_inputs(args, registration_required=False)
    manifest, reserve, thresholds = inputs["manifest"], inputs["reserve"], inputs["thresholds"]
    rows = inputs["rows"]
    exclusion = [line for line in args.exclusion.read_text(encoding="utf-8").splitlines() if line]
    if exclusion != sorted(set(exclusion)) or len(exclusion) != EXPECTED_PRIOR_CLIPS:
        raise RuntimeError("exclusion list order/count drift")
    prior = set(exclusion)
    prior_slots = {site_slot(clip) for clip in exclusion}
    cohort_clips = {normalized_clip(row["clip"]) for row in rows}
    cohort_slots = {site_slot(clip) for clip in cohort_clips}
    generation = generation_contract(inputs)
    heads = head_contract(inputs)
    checks = {
        "events_260": len(manifest["events"]) == EXPECTED_EVENTS,
        "views_1096": len(rows) == EXPECTED_VIEWS,
        "exclusion_641": len(exclusion) == EXPECTED_PRIOR_CLIPS,
        "clip_intersection_empty": not bool(cohort_clips & prior),
        "site_slot_intersection_empty": not bool(cohort_slots & prior_slots),
        "reserve_454": len(reserve["entries"]) == EXPECTED_RESERVE_EVENTS,
        "reserve_manifest_link": reserve["cohort_manifest_sha256"] == sha256_file(args.manifest),
        "threshold_six_seed_arms": sum(len(arms) for arms in thresholds["thresholds"].values()) == 6,
        "explicit_prompt_tau_pinned": generation["explicit_abstention_prompt"].get("tau")
        == 0.9491225851331859,
        "generation_passes_8": tuple(generation) == GENERATION_PASSES,
        "head_passes_2": set(heads) == set(HEAD_PASSES),
        "content_pins_complete": all(
            row["content_pin"]["clip_id"] == normalized_clip(row["clip"])
            and int(row["content_pin"]["expected_bytes"]) > 0
            and bool(row["content_pin"]["etag"])
            and row["content_pin"]["downloaded_sha256"] is None
            for row in rows
        ),
        "exact_frames_integer": all(isinstance(row["native_frame"], int) for row in rows),
        "proxy_labels_present": all(isinstance(row["proxy_strata"], dict) for row in rows),
        "three_gpu_wave_plan": all(set(wave) == {"wave", "gpu0", "gpu1", "gpu2"} for wave in GPU_WAVES),
        "no_aggregate_subcommand": "aggregate" not in COMMANDS,
    }
    registration = inputs["registration"]
    if registration is not None:
        registered = registration["artifacts"]
        checks |= {
            "registration_manifest_sha": registered["b1_cohort_manifest.json"] == sha256_file(args.manifest),
            "registration_reserve_sha": registered["b1_reserve.json"] == sha256_file(args.reserve),
            "registration_exclusion_sha": registered["exclusion_clips.txt"] == sha256_file(args.exclusion),
            "registration_threshold_sha": registered["b1_thresholds.json"] == sha256_file(args.thresholds),
            "registration_runner_sha": registration["execution"]["runner"]["sha256"]
            == sha256_file(Path(__file__)),
            "registration_stop_not_run": registration["status"] == "FROZEN_NOT_EXECUTED",
        }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"B1 dry-run checks failed: {failed}")
    after_modules = {name for name in MODEL_MODULES if name in sys.modules}
    newly_loaded = sorted(after_modules - before_modules)
    if newly_loaded:
        raise RuntimeError(f"dry-run loaded model runtimes: {newly_loaded}")
    report = {
        "schema_version": "meva_b1_oneshot_dry_run_v1",
        "status": "PASS",
        "registration_state": "verified" if registration is not None else "pre_registration",
        "cohort": {"events": EXPECTED_EVENTS, "candidate_views": EXPECTED_VIEWS},
        "file_sha256": file_sha_map(args),
        "generation_contract": generation,
        "head_contract": heads,
        "execution_plan": execution_plan(),
        "checks": {name: "PASS" for name in sorted(checks)},
        "model_modules_loaded_by_dry_run": newly_loaded,
        "media_or_pixels_opened": 0,
        "model_outputs_read": 0,
        "model_inferences": 0,
        "gpu_jobs_started": 0,
        "writes_performed": 0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def require_registered(args: argparse.Namespace) -> dict[str, Any]:
    inputs = load_inputs(args, registration_required=True)
    registration = inputs["registration"]
    if registration["status"] != "FROZEN_NOT_EXECUTED":
        raise RuntimeError("registration is not in FROZEN_NOT_EXECUTED state")
    if registration["execution"]["runner"]["sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("registered runner SHA drift")
    return inputs


def preflight(args: argparse.Namespace) -> int:
    require_registered(args)
    if git("status", "--porcelain"):
        raise RuntimeError("execution requires a clean worktree")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/main")
    if head != remote:
        raise RuntimeError(f"execution commit is not synced: HEAD={head}, origin/main={remote}")
    free = shutil.disk_usage(args.output_root.resolve().anchor).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"free disk below 80 GiB: {free}")
    payload = {
        "schema_version": "meva_b1_preflight_v1",
        "stage": "preflight",
        "status": "completed",
        "head": head,
        "origin_main": remote,
        "runner_sha256": sha256_file(Path(__file__)),
        "w4_runner_sha256": sha256_file(W4_RUNNER),
        "registration_sha256": sha256_file(args.registration),
        "free_bytes": free,
        "attempt_limit": MAX_STAGE_ATTEMPTS,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "checkpoints" / "b1_preflight.json", payload)
    # W4 compatibility checkpoint used only by the imported feature stage.
    atomic_json(
        args.output_root / "checkpoints" / "preflight.json",
        {
            "schema_version": "meva_w4_oneshot_checkpoint_v1",
            "stage": "preflight",
            "status": "completed",
            "head": head,
            "runner_sha256": sha256_file(W4_RUNNER),
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def require_b1_preflight(output_root: Path) -> dict[str, Any]:
    checkpoint = load_json(output_root / "checkpoints" / "b1_preflight.json")
    if checkpoint.get("stage") != "preflight" or checkpoint.get("status") != "completed":
        raise RuntimeError("invalid B1 preflight checkpoint")
    if checkpoint["head"] != git("rev-parse", "HEAD"):
        raise RuntimeError("checkout changed after B1 preflight")
    return checkpoint


def seal_config(args: argparse.Namespace) -> int:
    require_b1_preflight(args.output_root)
    inputs = require_registered(args)
    config = {
        "schema_version": "meva_b1_config_seal_v1",
        "status": "SEALED_BEFORE_B1_MEDIA_ACCESS",
        "registration_sha256": sha256_file(args.registration),
        "files": file_sha_map(args),
        "generation_contract": generation_contract(inputs),
        "head_contract": head_contract(inputs),
        "execution_plan": execution_plan(),
    }
    digest = atomic_json(args.output_root / "seal" / "config_seal.json", config)
    atomic_json(
        args.output_root / "checkpoints" / "seal.config.json",
        {"stage": "seal.config", "status": "completed", "config_seal_sha256": digest},
    )
    print(json.dumps({"status": "PASS", "config_seal_sha256": digest}, indent=2))
    return 0


def require_config_seal(output_root: Path) -> None:
    checkpoint = load_json(output_root / "checkpoints" / "seal.config.json")
    path = output_root / "seal" / "config_seal.json"
    require_hash(path, checkpoint["config_seal_sha256"], "config seal")


def patch_w4(expected_views: int = EXPECTED_VIEWS) -> Any:
    from scripts import w4_oneshot_runner as w4

    w4.EXPECTED_EVENTS = EXPECTED_EVENTS
    w4.EXPECTED_VIEWS = expected_views
    w4.SEED = DECODING_SEED
    return w4


def materialize(args: argparse.Namespace) -> int:
    require_config_seal(args.output_root)
    inputs = require_registered(args)
    rows = inputs["rows"]
    w4 = patch_w4()
    from scripts import w4_dev_comparator_runner as dev

    videos = dev.video_index(args.video_root.resolve())
    pins_by_clip: dict[str, dict[str, Any]] = {}
    for row in rows:
        pin = row["content_pin"]
        clip = normalized_clip(row["clip"])
        if clip in pins_by_clip and pins_by_clip[clip] != pin:
            raise RuntimeError(f"content pin drift within clip: {clip}")
        pins_by_clip[clip] = pin
    missing = sorted(set(pins_by_clip) - set(videos))
    if missing:
        raise FileNotFoundError(f"manifest-consumed source clips missing: {len(missing)}")
    source_ledger = []
    for clip in sorted(pins_by_clip):
        path = videos[clip]
        pin = pins_by_clip[clip]
        if path.stat().st_size != int(pin["expected_bytes"]):
            raise RuntimeError(f"source size drift: {clip}")
        observed_sha = sha256_file(path)
        expected_sha = pin.get("downloaded_sha256")
        if expected_sha is not None and observed_sha != expected_sha:
            raise RuntimeError(f"source SHA drift: {clip}")
        source_ledger.append(
            {
                "clip_id": clip,
                "expected_bytes": int(pin["expected_bytes"]),
                "etag": pin["etag"],
                "downloaded_sha256": observed_sha,
            }
        )
    atomic_json(args.output_root / "source_content_sha256.json", source_ledger)
    manifest, registration = inputs["manifest"], inputs["registration"]
    w4.cohort_rows = lambda _: rows
    w4.load_sealed_inputs = lambda _: (
        manifest,
        registration | {"_verified_sha256": sha256_file(args.registration)},
    )
    return w4.materialize(args)


def features(args: argparse.Namespace) -> int:
    require_config_seal(args.output_root)
    require_b1_preflight(args.output_root)
    patch_w4()
    from scripts import w4_oneshot_runner as w4

    return w4.features(args)


def generation_entry(inputs: dict[str, Any], pass_id: str) -> dict[str, Any]:
    contract = generation_contract(inputs)
    if pass_id not in contract:
        raise ValueError(pass_id)
    return contract[pass_id]


def generate(args: argparse.Namespace) -> int:
    require_config_seal(args.output_root)
    inputs = require_registered(args)
    entry = generation_entry(inputs, args.pass_id)
    expected_gpu = {
        GENERATION_PASSES[0]: 0,
        GENERATION_PASSES[1]: 1,
        GENERATION_PASSES[2]: 2,
        GENERATION_PASSES[3]: 0,
        GENERATION_PASSES[4]: 1,
        GENERATION_PASSES[5]: 2,
        "explicit_abstention_prompt": 0,
        "zero_shot": 1,
    }[args.pass_id]
    if args.physical_gpu != expected_gpu:
        raise RuntimeError(f"GPU shard drift: {args.pass_id} -> GPU{expected_gpu}")
    adapter: Path | None = None
    if entry["kind"] == "adapter_generation":
        if args.adapter is None:
            raise RuntimeError("adapter generation requires --adapter")
        adapter = args.adapter.resolve()
        if sha256_tree(adapter) != entry["checkpoint_tree_sha256"]:
            raise RuntimeError(f"adapter tree SHA drift: {args.pass_id}")
        require_hash(
            adapter / "adapter_model.safetensors",
            entry["adapter_model_sha256"],
            f"{args.pass_id} adapter model",
        )
    elif args.adapter is not None:
        raise RuntimeError("base generation must not receive --adapter")

    w4 = patch_w4()
    from scripts.operator import w5a_fair_baselines_runner as w5a

    original_prompt = w5a.EXPLICIT_SYSTEM_PROMPT
    if args.pass_id == "zero_shot":
        w5a.EXPLICIT_SYSTEM_PROMPT = w5a.NEUTRAL_SYSTEM_PROMPT
    try:
        w4.ARMS = (args.pass_id,)
        w4.PRIMARY_ARM = args.pass_id if adapter is not None else "__adapter_none__"
        w4.CONTROL_ARM = args.pass_id if adapter is None else "__control_none__"
        w4.EXPECTED_GPU = {args.pass_id: expected_gpu}
        w4.RELEASE_THRESHOLDS = {args.pass_id: float(entry.get("tau", 0.0))}
        delegated = argparse.Namespace(
            output_root=args.output_root,
            arm=args.pass_id,
            physical_gpu=args.physical_gpu,
            answer_abstain_adapter=adapter,
            batch_size=args.batch_size,
        )
        return w4.answerer(delegated)
    finally:
        w5a.EXPLICIT_SYSTEM_PROMPT = original_prompt


def validate_prefix(rows: list[dict[str, Any]], completed: list[dict[str, Any]], label: str) -> None:
    if [row["row_id"] for row in completed] != [row["row_id"] for row in rows[: len(completed)]]:
        raise RuntimeError(f"{label} output is not an ordered prefix")


def head(args: argparse.Namespace) -> int:
    require_config_seal(args.output_root)
    inputs = require_registered(args)
    if args.arm not in HEAD_PASSES:
        raise ValueError(args.arm)
    expected_gpu = 0 if args.arm == "correctness_selector" else 1
    if args.physical_gpu != expected_gpu or os.environ.get("CUDA_VISIBLE_DEVICES") != str(expected_gpu):
        raise RuntimeError(f"{args.arm} must run alone on physical GPU{expected_gpu}")
    pin = head_contract(inputs)[args.arm]
    checkpoint_path = args.checkpoint.resolve()
    require_hash(checkpoint_path, pin["remote_checkpoint_sha256"], f"{args.arm} checkpoint")

    import numpy as np
    import torch
    from torch import nn

    from scripts.operator import answerer_alignment_lora as base
    from scripts.operator import w5a_fair_baselines_runner as w5a

    w4 = patch_w4()
    rows = w4.runtime_rows(args.output_root.resolve())
    output_path = args.output_root / "heads" / f"{args.arm}.jsonl"
    completed = read_jsonl(output_path)
    validate_prefix(rows, completed, args.arm)
    if len(completed) == len(rows):
        print(json.dumps({"status": "PASS", "arm": args.arm, "rows": len(rows)}, indent=2))
        return 0
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model, processor = base.load_model_and_processor(adapter=None, trainable=False)
    model.eval()
    classifier = nn.Linear(
        int(checkpoint["hidden_size"]), len(checkpoint["classes"])
    ).to("cuda:0", dtype=torch.float32)
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.eval()
    with torch.inference_mode():
        for index, row in enumerate(rows[len(completed) :], len(completed) + 1):
            batch = {
                key: value.to("cuda:0")
                for key, value in w5a._encode_head(processor, [row], args.arm).items()
            }
            output = model(**batch, output_hidden_states=True, return_dict=True)
            position = int(batch["attention_mask"].sum()) - 1
            hidden = output.hidden_states[-1][:, position, :].float()
            probabilities = torch.softmax(classifier(hidden), dim=1)[0].cpu().numpy()
            class_index = int(np.argmax(probabilities))
            append_jsonl(
                output_path,
                {
                    "schema_version": "meva_b1_head_raw_v1",
                    "row_id": row["row_id"],
                    "event_id": row["event_id"],
                    "camera_id": row["camera_id"],
                    "role": row["role"],
                    "arm": args.arm,
                    "prediction": checkpoint["classes"][class_index],
                    "confidence": float(probabilities[class_index]),
                    "supported_probability": (
                        float(probabilities[1]) if args.arm == "correctness_selector" else None
                    ),
                },
            )
            if index % 25 == 0 or index == len(rows):
                print(f"{args.arm} {index}/{len(rows)}", flush=True)
    completed = read_jsonl(output_path)
    validate_prefix(rows, completed, args.arm)
    if len(completed) != EXPECTED_VIEWS:
        raise RuntimeError(f"incomplete head output: {args.arm}")
    print(
        json.dumps(
            {"status": "PASS", "arm": args.arm, "rows": len(completed), "sha256": sha256_file(output_path)},
            indent=2,
        )
    )
    return 0


def seal_raw(args: argparse.Namespace) -> int:
    require_config_seal(args.output_root)
    expected = [
        args.output_root / "candidate_views.materialized.jsonl",
        args.output_root / "source_content_sha256.json",
        args.output_root / "features_host" / "m4_availability_scores.jsonl",
        *[args.output_root / "answers" / f"{pass_id}.jsonl" for pass_id in GENERATION_PASSES],
        *[args.output_root / "heads" / f"{arm}.jsonl" for arm in HEAD_PASSES],
    ]
    row_ids: set[str] | None = None
    records = []
    for path in expected:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == ".jsonl":
            rows = read_jsonl(path)
            if len(rows) != EXPECTED_VIEWS:
                raise RuntimeError(f"raw row count drift: {path} -> {len(rows)}")
            ids = {row["row_id"] for row in rows}
            if row_ids is None:
                row_ids = ids
            elif ids != row_ids:
                raise RuntimeError(f"raw row identity drift: {path}")
        records.append(
            {
                "path": str(path.relative_to(args.output_root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "meva_b1_raw_sha_manifest_v1",
        "status": "RAW_SEALED_NOT_AGGREGATED",
        "files": records,
        "candidate_views": EXPECTED_VIEWS,
        "generation_passes": len(GENERATION_PASSES),
        "head_passes": len(HEAD_PASSES),
        "aggregation_performed": False,
    }
    manifest_sha = atomic_json(args.output_root / "seal" / "raw_sha_manifest.json", manifest)
    raw_seal = {
        "schema_version": "meva_b1_raw_seal_v1",
        "config_seal_sha256": sha256_file(args.output_root / "seal" / "config_seal.json"),
        "raw_sha_manifest_sha256": manifest_sha,
        "opened_definition": "only a later explicit aggregate script invocation after this seal",
    }
    raw_seal_sha = atomic_json(args.output_root / "seal" / "raw_seal.json", raw_seal)
    print(
        json.dumps(
            {"status": "PASS", "raw_manifest_sha256": manifest_sha, "raw_seal_sha256": raw_seal_sha},
            indent=2,
        )
    )
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reserve", type=Path, default=DEFAULT_RESERVE)
    parser.add_argument("--exclusion", type=Path, default=DEFAULT_EXCLUSION)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "preflight", "seal-config", "seal-raw"):
        child = sub.add_parser(name)
        add_common(child)
    materialize_parser = sub.add_parser("materialize")
    add_common(materialize_parser)
    materialize_parser.add_argument("--video-root", type=Path, required=True)
    materialize_parser.add_argument("--meva-repo", type=Path, required=True)
    materialize_parser.add_argument("--semantic-audit", type=Path)
    feature_parser = sub.add_parser("features")
    add_common(feature_parser)
    from scripts import w4_dev_comparator_runner as dev

    feature_parser.add_argument("--availability-model", type=Path, default=dev.DEFAULT_AVAILABILITY_MODEL)
    feature_parser.add_argument("--device", default="cuda:2")
    feature_parser.add_argument("--batch-size", type=int, default=16)
    generate_parser = sub.add_parser("generate")
    add_common(generate_parser)
    generate_parser.add_argument("--pass-id", choices=GENERATION_PASSES, required=True)
    generate_parser.add_argument("--adapter", type=Path)
    generate_parser.add_argument("--physical-gpu", type=int, choices=(0, 1, 2), required=True)
    generate_parser.add_argument("--batch-size", type=int, default=8)
    head_parser = sub.add_parser("head")
    add_common(head_parser)
    head_parser.add_argument("--arm", choices=HEAD_PASSES, required=True)
    head_parser.add_argument("--checkpoint", type=Path, required=True)
    head_parser.add_argument("--physical-gpu", type=int, choices=(0, 1), required=True)
    return parser.parse_args()


COMMANDS = {
    "dry-run": dry_run,
    "preflight": preflight,
    "seal-config": seal_config,
    "materialize": materialize,
    "features": features,
    "generate": generate,
    "head": head,
    "seal-raw": seal_raw,
}


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(COMMANDS[arguments.command](arguments))
