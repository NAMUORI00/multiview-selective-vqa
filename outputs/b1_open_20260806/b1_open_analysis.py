"""Open B1 with only the frozen registration's confirmatory and auxiliary analyses."""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Sequence

import joblib
import numpy as np


REPO = Path(__file__).resolve().parents[2]
PREREG = REPO / "outputs" / "b1_prereg_20260805"
RUN = REPO / "outputs" / "b1_run_20260806"
OUT = Path(__file__).resolve().parent
REGISTRATION = PREREG / "b1_registration.json"
THRESHOLDS = PREREG / "b1_thresholds.json"
COHORT = PREREG / "b1_cohort_manifest.json"
RAW_MANIFEST = RUN / "seal" / "raw_sha_manifest.json"
RAW_SEAL = RUN / "seal" / "raw_seal.json"
CONFIG_SEAL = RUN / "seal" / "config_seal.json"
TASK_SELECTION = (
    REPO
    / "outputs"
    / "meva_full_20260713"
    / "w4_dev_comparator_20260714"
    / "features_host_v2"
    / "w4_dev_comparator_selection.json"
)
AVAILABILITY_MODEL = (
    REPO
    / "outputs"
    / "meva_full_20260713_m3"
    / "judge"
    / "siglip2_dinov2_logreg.joblib"
)

EXPECTED_REGISTRATION_SHA256 = "8ac9ab2e42ed9db80fa10db20b2c679f59b9052af94c1fc23385f17bd1c21dad"
EXPECTED_EVENTS = 260
EXPECTED_ROWS = 1096
BOOTSTRAP_REPLICATES = 4_000
BOOTSTRAP_SEED = 20260805
SAFETY_MARGIN = 0.05
ABSTAIN = "The visual evidence is insufficient to determine this."
DINO_KEYS = (
    "fresh_dino_total_regions",
    "fresh_dino_person_regions",
    "fresh_dino_max_conf",
    "fresh_dino_max_person_conf",
    "fresh_dino_region_area_sum",
)
ANSWER_ABSTAIN_ARMS = tuple(f"answer_abstain_seed_{seed}" for seed in (20260713, 20260714, 20260715))
ANSWER_ONLY_ARMS = tuple(f"answer_only_seed_{seed}" for seed in (20260713, 20260714, 20260715))
GENERATION_ARMS = (
    *ANSWER_ABSTAIN_ARMS,
    *ANSWER_ONLY_ARMS,
    "explicit_abstention_prompt",
    "zero_shot",
)
HEAD_ARMS = ("correctness_selector", "action_classification_head")
PRIMARY = "answer_abstain_seed_20260713"
H1_COMPARATOR = "explicit_abstention_prompt"
H2_COMPARATOR = "correctness_selector"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Match scripts/w4_oneshot_runner.py's frozen feature-tree hash."""
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError(f"empty tree: {path}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rate(numerator: int | float, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("rate denominator must be positive")
    return float(numerator / denominator)


def atom(value: float) -> str:
    return repr(float(value))


def verify_raw_integrity() -> dict[str, Any]:
    registration_sha = sha256_file(REGISTRATION)
    if registration_sha != EXPECTED_REGISTRATION_SHA256:
        raise RuntimeError(f"registration SHA mismatch: {registration_sha}")
    raw_seal = read_json(RAW_SEAL)
    manifest_sha = sha256_file(RAW_MANIFEST)
    config_sha = sha256_file(CONFIG_SEAL)
    if manifest_sha != raw_seal["raw_sha_manifest_sha256"]:
        raise RuntimeError("raw manifest SHA does not match raw seal")
    if config_sha != raw_seal["config_seal_sha256"]:
        raise RuntimeError("config seal SHA does not match raw seal")

    manifest = read_json(RAW_MANIFEST)
    checks: list[dict[str, Any]] = []
    for entry in manifest["files"]:
        path = RUN / entry["path"]
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_sha = sha256_file(path) if exists else None
        check = {
            "path": entry["path"],
            "expected_bytes": int(entry["bytes"]),
            "actual_bytes": actual_bytes,
            "bytes_match": actual_bytes == int(entry["bytes"]),
            "expected_sha256": entry["sha256"],
            "actual_sha256": actual_sha,
            "sha256_match": actual_sha == entry["sha256"],
        }
        checks.append(check)
    if len(checks) != 13 or not all(item["bytes_match"] and item["sha256_match"] for item in checks):
        raise RuntimeError("one or more of the 13 registered raw files failed integrity verification")

    registration = read_json(REGISTRATION)
    reference_checks = []
    for relative in (
        "outputs/b1_prereg_20260805/b1_cohort_manifest.json",
        "outputs/b1_prereg_20260805/b1_thresholds.json",
        "outputs/b1_prereg_20260805/b1_reserve.json",
        "outputs/b1_prereg_20260805/exclusion_clips.txt",
    ):
        expected = registration["reference_file_sha256"][relative]
        actual = sha256_file(REPO / relative)
        reference_checks.append({"path": relative, "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected})
    if not all(item["match"] for item in reference_checks):
        raise RuntimeError("registered preregistration reference SHA mismatch")
    return {
        "status": "PASS",
        "registration": {"path": str(REGISTRATION.relative_to(REPO)), "sha256": registration_sha, "match": True},
        "raw_seal": {"path": str(RAW_SEAL.relative_to(REPO)), "sha256": sha256_file(RAW_SEAL)},
        "raw_manifest": {"path": str(RAW_MANIFEST.relative_to(REPO)), "sha256": manifest_sha, "files": checks},
        "config_seal": {"path": str(CONFIG_SEAL.relative_to(REPO)), "sha256": config_sha},
        "reference_files": reference_checks,
    }


def index_rows(rows: list[dict[str, Any]], label: str, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"{label} row count drift: {len(rows)}")
    indexed = {row["row_id"]: row for row in rows}
    if len(indexed) != EXPECTED_ROWS or set(indexed) != expected_ids:
        raise RuntimeError(f"{label} row identity drift")
    return indexed


def strict_parse(raw: str) -> tuple[bool, str | None, str | None, bool | None]:
    match = re.search(r"\{.*\}", raw.strip(), flags=re.DOTALL)
    if not match:
        return False, None, None, None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False, None, None, None
    if not isinstance(value, dict):
        return False, None, None, None
    decision = value.get("decision")
    answer = value.get("answer")
    if not isinstance(decision, str) or decision not in {"answer", "abstain"} or not isinstance(answer, str):
        return False, None, None, None
    normalized = re.sub(r"\s+", "", answer.lower())
    abstained = decision == "abstain" or re.sub(r"\s+", "", ABSTAIN.lower()) in normalized
    return True, decision, answer, abstained


def generation_metrics(
    arm: str,
    answers: dict[str, dict[str, Any]],
    tau: float | None,
    candidates: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    unparsable = 0
    for row_id, answer in answers.items():
        parsed, decision, parsed_answer, abstained = strict_parse(str(answer["raw_response"]))
        if parsed:
            if (
                decision != answer["decision"]
                or parsed_answer != answer["answer"]
                or bool(abstained) != bool(answer["was_abstained"])
            ):
                raise RuntimeError(f"strict parser disagrees with frozen parser: {arm}/{row_id}")
        else:
            unparsable += 1

        if arm == "zero_shot":
            # The registered zero-shot pass is record-only. Its frozen parser state is
            # retained for the correctness-selector answer-source contract.
            answered = not bool(answer["was_abstained"])
            correct = bool(answer["target_correct"]) if answered else False
            released = False
            below_tau = False
        else:
            answered = parsed and not bool(abstained)
            correct = bool(answer["target_correct"]) if answered else False
            released = bool(answered and tau is not None and float(answer["confidence"]) >= tau)
            below_tau = bool(answered and not released)
        metrics[row_id] = {
            "answered": bool(answered),
            "correct": bool(correct),
            "below_tau": bool(below_tau),
            "released": bool(released),
            "unparsable": not parsed,
            "confidence": float(answer["confidence"]),
            "gpu_seconds": float(answer["vlm_gpu_seconds"]),
            "role": candidates[row_id]["role"],
        }
    return metrics, {"rows": EXPECTED_ROWS, "unparsable": unparsable, "rate": rate(unparsable, EXPECTED_ROWS)}


def head_metrics(
    arm: str,
    head: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    zero_shot: dict[str, dict[str, Any]],
    tau: float,
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for row_id, raw in head.items():
        candidate = candidates[row_id]
        if arm == "correctness_selector":
            score = float(raw["supported_probability"])
            answered = bool(score >= 0.5 and not bool(zero_shot[row_id]["was_abstained"]))
            correct = bool(zero_shot[row_id]["target_correct"]) if answered else False
        elif arm == "action_classification_head":
            score = float(raw["confidence"])
            answered = raw["prediction"] == candidate["activity"]
            correct = bool(candidate["role"] == "positive") if answered else False
        else:
            raise ValueError(arm)
        released = bool(answered and score >= tau)
        metrics[row_id] = {
            "answered": bool(answered),
            "correct": bool(correct),
            "below_tau": bool(answered and not released),
            "released": released,
            "unparsable": False,
            "confidence": score,
            "gpu_seconds": None,
            "role": candidate["role"],
        }
    return metrics


def stage_summary(metrics: dict[str, dict[str, Any]], candidates: dict[str, dict[str, Any]], tau: float | None) -> dict[str, Any]:
    def summarize(ids: Iterable[str]) -> dict[str, Any]:
        selected = [metrics[row_id] for row_id in ids]
        n = len(selected)
        answered = [item for item in selected if item["answered"]]
        below = [item for item in selected if item["below_tau"]]
        released = [item for item in selected if item["released"]]
        return {
            "denominator_rows": n,
            "unparsable": sum(item["unparsable"] for item in selected),
            "stage2_answered": len(answered),
            "stage2_answered_correct": sum(item["correct"] for item in answered),
            "stage2_answered_wrong": sum(not item["correct"] for item in answered),
            "stage3_below_tau": len(below) if tau is not None else None,
            "stage3_below_tau_correct": sum(item["correct"] for item in below) if tau is not None else None,
            "stage3_below_tau_wrong": sum(not item["correct"] for item in below) if tau is not None else None,
            "stage3_released": len(released) if tau is not None else None,
            "stage3_released_correct": sum(item["correct"] for item in released) if tau is not None else None,
            "stage3_released_wrong": sum(not item["correct"] for item in released) if tau is not None else None,
        }

    all_ids = list(candidates)
    result = {"tau": tau, "all_rows": summarize(all_ids), "by_role": {}}
    for role in ("positive", "negative_geometric"):
        ids = [row_id for row_id, row in candidates.items() if row["role"] == role]
        result["by_role"][role] = summarize(ids)
    return result


def event_outcomes(
    metrics: dict[str, dict[str, Any]],
    selection: dict[str, str],
    rows_by_event: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    outcomes: dict[str, dict[str, int]] = {}
    for event_id, event_rows in rows_by_event.items():
        selected_id = selection[event_id]
        selected = metrics[selected_id]
        outcomes[event_id] = {
            "correct_released_yield": int(selected["released"] and selected["correct"]),
            "wrong_release_risk": int(selected["released"] and not selected["correct"]),
            "any_negative_violation_risk": int(
                any(metrics[row["row_id"]]["released"] for row in event_rows if row["role"] != "positive")
            ),
        }
    return outcomes


def outcome_summary(outcomes: dict[str, dict[str, int | float]], event_ids: Sequence[str]) -> dict[str, Any]:
    result = {"denominator_events": len(event_ids)}
    for metric in ("correct_released_yield", "wrong_release_risk", "any_negative_violation_risk"):
        numerator = float(sum(float(outcomes[event_id][metric]) for event_id in event_ids))
        result[metric] = {"numerator": numerator, "rate": rate(numerator, len(event_ids))}
    return result


def paired_bootstrap(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.ndim != 1 or left_array.size == 0:
        raise ValueError("paired bootstrap requires non-empty equal vectors")
    difference = left_array - right_array
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, difference.size, size=(BOOTSTRAP_REPLICATES, difference.size))
    draws = difference[indices].mean(axis=1)
    return {
        "delta": float(difference.mean()),
        "two_sided_95_ci": [
            float(np.quantile(draws, 0.025, method="linear")),
            float(np.quantile(draws, 0.975, method="linear")),
        ],
        "one_sided_95_upper": float(np.quantile(draws, 0.95, method="linear")),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "resampling_unit": "event",
        "percentile_method": "linear",
    }


def exact_mcnemar(n10: int, n01: int) -> float:
    discordant = n10 + n01
    if discordant == 0:
        return 1.0
    lower = min(n10, n01)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1)) * 0.5**discordant
    return min(1.0, 2.0 * tail)


def wilson(successes: int, n: int, z: float) -> tuple[float, float]:
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return centre - half, centre + half


def newcombe_method10(n11: int, n10: int, n01: int, n00: int) -> list[float]:
    n = n11 + n10 + n01 + n00
    z = NormalDist().inv_cdf(0.975)
    p_left = (n11 + n10) / n
    p_right = (n11 + n01) / n
    l_left, u_left = wilson(n11 + n10, n, z)
    l_right, u_right = wilson(n11 + n01, n, z)
    dl_left, du_left = p_left - l_left, u_left - p_left
    dl_right, du_right = p_right - l_right, u_right - p_right
    denominator = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    numerator = n11 * n00 - n10 * n01
    if denominator == 0.0:
        phi = 0.0
    elif numerator > 0:
        phi = max(numerator - n / 2.0, 0.0) / denominator
    else:
        phi = numerator / denominator
    delta = (n10 - n01) / n
    lower_distance = math.sqrt(max(0.0, dl_left**2 - 2.0 * phi * dl_left * du_right + du_right**2))
    upper_distance = math.sqrt(max(0.0, du_left**2 - 2.0 * phi * du_left * dl_right + dl_right**2))
    return [max(-1.0, delta - lower_distance), min(1.0, delta + upper_distance)]


def paired_binary(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    pairs = Counter((int(a), int(b)) for a, b in zip(left, right, strict=True))
    counts = {
        "n11": pairs[(1, 1)],
        "n10": pairs[(1, 0)],
        "n01": pairs[(0, 1)],
        "n00": pairs[(0, 0)],
    }
    return {
        "counts": counts,
        "denominator_events": sum(counts.values()),
        "delta": (counts["n10"] - counts["n01"]) / sum(counts.values()),
        "newcombe_method10_two_sided_95_ci": newcombe_method10(**counts),
        "exact_two_sided_mcnemar_p": exact_mcnemar(counts["n10"], counts["n01"]),
        "role": "descriptive companion; not an additional gate",
    }


def compare_outcomes(
    left: dict[str, dict[str, int | float]],
    right: dict[str, dict[str, int | float]],
    event_ids: Sequence[str],
    *,
    confirmatory: bool,
) -> dict[str, Any]:
    vectors: dict[str, tuple[list[float], list[float]]] = {}
    bootstraps: dict[str, Any] = {}
    for metric in ("correct_released_yield", "wrong_release_risk", "any_negative_violation_risk"):
        left_values = [float(left[event_id][metric]) for event_id in event_ids]
        right_values = [float(right[event_id][metric]) for event_id in event_ids]
        vectors[metric] = (left_values, right_values)
        bootstraps[metric] = paired_bootstrap(left_values, right_values)
    criteria = {
        "correct_released_yield_ci_lower_gt_0": bool(
            bootstraps["correct_released_yield"]["two_sided_95_ci"][0] > 0.0
        ),
        "wrong_release_risk_upper_le_0_05": bool(
            bootstraps["wrong_release_risk"]["one_sided_95_upper"] <= SAFETY_MARGIN
        ),
        "any_negative_violation_risk_upper_le_0_05": bool(
            bootstraps["any_negative_violation_risk"]["one_sided_95_upper"] <= SAFETY_MARGIN
        ),
    }
    all_pass = all(criteria.values())
    return {
        "analysis_status": "confirmatory" if confirmatory else "nonconfirmatory",
        "denominator_events": len(event_ids),
        "left": outcome_summary(left, event_ids),
        "right": outcome_summary(right, event_ids),
        "differences": bootstraps,
        "criteria": criteria,
        "verdict": ("PASS" if all_pass else "FAIL") if confirmatory else "NOT_A_GATE",
        "paired_binary_correct_released_yield": paired_binary(*vectors["correct_released_yield"]),
    }


def mean_seed_outcomes(seed_outcomes: Sequence[dict[str, dict[str, int]]], event_ids: Sequence[str]) -> dict[str, dict[str, float]]:
    return {
        event_id: {
            metric: float(np.mean([outcome[event_id][metric] for outcome in seed_outcomes]))
            for metric in ("correct_released_yield", "wrong_release_risk", "any_negative_violation_risk")
        }
        for event_id in event_ids
    }


def holm_adjust(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    m = len(ordered)
    running = 0.0
    result: dict[str, dict[str, Any]] = {}
    for rank, (name, pvalue) in enumerate(ordered, start=1):
        adjusted = min(1.0, max(running, (m - rank + 1) * pvalue))
        running = adjusted
        result[name] = {
            "raw_exact_two_sided_mcnemar_p": pvalue,
            "holm_adjusted_p": adjusted,
            "reject_at_0_05": adjusted <= alpha,
            "holm_rank": rank,
            "family_size": m,
        }
    return result


def load_feature_batches(rows: list[dict[str, Any]], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    feature_root = RUN / "features_host" / "siglip2_dinov2"
    images = np.empty((len(rows), 768), dtype=np.float32)
    dinov2 = np.empty((len(rows), 768), dtype=np.float32)
    for batch_index, start in enumerate(range(0, len(rows), batch_size)):
        stop = min(start + batch_size, len(rows))
        path = feature_root / f"batch_{batch_index:04d}.npz"
        with np.load(path, allow_pickle=True) as cache:
            expected_ids = [row["row_id"] for row in rows[start:stop]]
            if cache["row_ids"].tolist() != expected_ids:
                raise RuntimeError(f"feature row order drift: {path}")
            images[start:stop] = np.asarray(cache["image"], dtype=np.float32)
            dinov2[start:stop] = np.asarray(cache["dinov2_cls"], dtype=np.float32)
    return images, dinov2


def logistic_scores(matrix: np.ndarray, selector: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(selector["scaler_mean"], dtype=np.float64)
    scale = np.asarray(selector["scaler_scale"], dtype=np.float64)
    coefficient = np.asarray(selector["coef"], dtype=np.float64)
    if matrix.shape[1] != mean.size or mean.size != scale.size or scale.size != coefficient.size:
        raise RuntimeError("task selector feature dimension drift")
    standardized = (matrix.astype(np.float64) - mean) / scale
    logits = standardized @ coefficient + float(selector["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))


def feature_and_task_scores(
    rows: list[dict[str, Any]],
    rows_by_event: dict[str, list[dict[str, Any]]],
    sealed_availability: dict[str, float],
) -> tuple[dict[str, float], dict[str, Any]]:
    checkpoint = read_json(RUN / "checkpoints" / "features.json")
    feature_plan = read_json(RUN / "features_host" / "feature_plan.json")
    image, dinov2 = load_feature_batches(rows, int(feature_plan["batch_size"]))

    text_path = RUN / "features_host" / "text_features.npz"
    with np.load(text_path, allow_pickle=True) as cache:
        questions = cache["questions"].tolist()
        query_map = {question: np.asarray(cache["query"][index], dtype=np.float32) for index, question in enumerate(questions)}
    query = np.stack([query_map[row["question"]] for row in rows])
    dino_rows = {row["row_id"]: row for row in read_jsonl(RUN / "features_host" / "dino.jsonl")}
    if set(dino_rows) != {row["row_id"] for row in rows}:
        raise RuntimeError("DINO auxiliary row coverage drift")
    dino_matrix = np.asarray([[float(dino_rows[row["row_id"]][key]) for key in DINO_KEYS] for row in rows])
    combined = np.concatenate(
        [image.astype(np.float64), query.astype(np.float64), image * query, np.abs(image - query), dino_matrix], axis=1
    )

    selection = read_json(TASK_SELECTION)
    task = logistic_scores(combined, selection["task_selector"])
    task_scores = {row["row_id"]: float(task[index]) for index, row in enumerate(rows)}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frozen = joblib.load(AVAILABILITY_MODEL)
    availability_matrix = np.concatenate([image, dinov2], axis=1).astype(np.float32, copy=False)
    scaler = frozen["scaler"]
    classifier = frozen["classifier"]
    standardized = (availability_matrix - np.asarray(scaler.mean_)) / np.asarray(scaler.scale_)
    logits = standardized @ np.asarray(classifier.coef_)[0] + float(np.asarray(classifier.intercept_)[0])
    recomputed = 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))
    observed = np.asarray([sealed_availability[row["row_id"]] for row in rows])
    recomputed_map = {row["row_id"]: float(recomputed[index]) for index, row in enumerate(rows)}
    sealed_top1 = {
        event_id: min(event_rows, key=lambda row: (-sealed_availability[row["row_id"]], row["camera_id"]))["row_id"]
        for event_id, event_rows in rows_by_event.items()
    }
    recomputed_top1 = {
        event_id: min(event_rows, key=lambda row: (-recomputed_map[row["row_id"]], row["camera_id"]))["row_id"]
        for event_id, event_rows in rows_by_event.items()
    }

    tree_expected = checkpoint["siglip2_dinov2_tree_sha256"]
    tree_actual = sha256_tree(RUN / "features_host" / "siglip2_dinov2")
    integrity = {
        "status": "MISMATCH_NONSEALED_AUXILIARY" if tree_actual != tree_expected else "PASS",
        "scope_note": "These auxiliary feature files are outside the registered 13-file raw manifest and are used only for record-only task-matched allocation.",
        "siglip2_dinov2_tree": {"expected_sha256": tree_expected, "actual_sha256": tree_actual, "match": tree_actual == tree_expected},
        "dino_jsonl": {
            "expected_sha256": checkpoint["grounding_dino_sha256"],
            "actual_sha256": sha256_file(RUN / "features_host" / "dino.jsonl"),
            "match": sha256_file(RUN / "features_host" / "dino.jsonl") == checkpoint["grounding_dino_sha256"],
        },
        "text_features": {
            "expected_sha256": checkpoint["text_features_sha256"],
            "actual_sha256": sha256_file(text_path),
            "match": sha256_file(text_path) == checkpoint["text_features_sha256"],
        },
        "task_selection": {
            "expected_sha256": "eb235723964355585e974b5f669a2e5eeb225794f6a2307999db5ee6cb89d042",
            "actual_sha256": sha256_file(TASK_SELECTION),
            "match": sha256_file(TASK_SELECTION) == "eb235723964355585e974b5f669a2e5eeb225794f6a2307999db5ee6cb89d042",
        },
        "availability_recomputation_diagnostic": {
            "max_absolute_score_difference": float(np.max(np.abs(recomputed - observed))),
            "top1_identical_events": sum(sealed_top1[event_id] == recomputed_top1[event_id] for event_id in sealed_top1),
            "denominator_events": EXPECTED_EVENTS,
        },
    }
    return task_scores, integrity


def allocation_record(
    policies: dict[str, dict[str, str]],
    candidates: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    answers: dict[str, dict[str, Any]],
    features_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    router = {
        "availability": {
            "siglip2_image_gpu_seconds": float(features_checkpoint["siglip2_image_gpu_seconds"]),
            "dinov2_gpu_seconds": float(features_checkpoint["dinov2_gpu_seconds"]),
        },
        "task_matched_selector": {
            "grounding_dino_gpu_seconds": float(features_checkpoint["grounding_dino_gpu_seconds"]),
            "siglip2_image_gpu_seconds": float(features_checkpoint["siglip2_image_gpu_seconds"]),
            "siglip2_text_gpu_seconds": float(features_checkpoint["siglip2_text_gpu_seconds"]),
        },
    }
    result: dict[str, Any] = {}
    for policy, selection in policies.items():
        ids = list(selection.values())
        allocation_hits = sum(candidates[row_id]["role"] == "positive" for row_id in ids)
        correct_release = sum(metrics[row_id]["released"] and metrics[row_id]["correct"] for row_id in ids)
        vlm_gpu = sum(float(answers[row_id]["vlm_gpu_seconds"]) for row_id in ids)
        router_gpu = sum(router[policy].values())
        result[policy] = {
            "denominator_events": EXPECTED_EVENTS,
            "A_1_numerator_selected_positive": allocation_hits,
            "A_1": rate(allocation_hits, EXPECTED_EVENTS),
            "U_1_numerator_correct_released": correct_release,
            "U_1": rate(correct_release, EXPECTED_EVENTS),
            "gpu_cost": {
                "router_components": router[policy],
                "router_gpu_seconds_all_1096_views": router_gpu,
                "selected_primary_vlm_gpu_seconds_260_rows": vlm_gpu,
                "total_gpu_seconds": router_gpu + vlm_gpu,
            },
        }
    result["selected_row_overlap"] = {
        "same_row_events": sum(
            policies["availability"][event_id] == policies["task_matched_selector"][event_id]
            for event_id in policies["availability"]
        ),
        "denominator_events": EXPECTED_EVENTS,
    }
    return {
        "analysis_status": "record_only_no_test_no_gate",
        "not_a_deployment_claim": True,
        "policies": result,
    }


def connected_components(cohort_events: list[dict[str, Any]]) -> list[list[str]]:
    parent = {event["event_id"]: event["event_id"] for event in cohort_events}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    clip_owner: dict[str, str] = {}
    for event in cohort_events:
        event_id = event["event_id"]
        for clip in {view["clip"] for view in event["views"]}:
            if clip in clip_owner:
                union(event_id, clip_owner[clip])
            else:
                clip_owner[clip] = event_id
    groups: dict[str, list[str]] = defaultdict(list)
    for event_id in parent:
        groups[find(event_id)].append(event_id)
    return sorted((sorted(values) for values in groups.values()), key=lambda values: values[0])


def cluster_bootstrap(differences: dict[str, float], clusters: list[list[str]], label: str) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        values = [differences[event_id] for cluster_index in sampled for event_id in clusters[int(cluster_index)]]
        draws[index] = float(np.mean(values))
    return {
        "delta": float(np.mean(list(differences.values()))),
        "two_sided_95_ci": [
            float(np.quantile(draws, 0.025, method="linear")),
            float(np.quantile(draws, 0.975, method="linear")),
        ],
        "one_sided_95_upper": float(np.quantile(draws, 0.95, method="linear")),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "resampling_unit": label,
        "clusters": len(clusters),
        "cluster_event_counts": sorted((len(cluster) for cluster in clusters), reverse=True),
        "analysis_status": "descriptive_nonconfirmatory",
    }


def clustered_comparison(
    left: dict[str, dict[str, int]],
    right: dict[str, dict[str, int]],
    clusters: list[list[str]],
    label: str,
) -> dict[str, Any]:
    result = {"analysis_status": "descriptive_nonconfirmatory"}
    for metric in ("correct_released_yield", "wrong_release_risk", "any_negative_violation_risk"):
        differences = {event_id: left[event_id][metric] - right[event_id][metric] for cluster in clusters for event_id in cluster}
        result[metric] = cluster_bootstrap(differences, clusters, label)
    return result


def activity_descriptives(
    cohort_events: list[dict[str, Any]],
    outcomes: dict[str, dict[str, dict[str, int]]],
) -> dict[str, Any]:
    by_activity: dict[str, list[str]] = defaultdict(list)
    for event in cohort_events:
        by_activity[event["activity"]].append(event["event_id"])
    result = {}
    for activity, event_ids in sorted(by_activity.items()):
        result[activity] = {
            "denominator_events": len(event_ids),
            "arms": {arm: outcome_summary(table, event_ids) for arm, table in outcomes.items()},
        }
    return {"analysis_status": "descriptive_no_test", "activities": result}


def proxy_strata(
    cohort_events: list[dict[str, Any]],
    comparisons: dict[str, tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]],
) -> dict[str, Any]:
    hard: list[str] = []
    reasons: dict[str, list[str]] = {"height": [], "fov": []}
    details = {}
    for event in cohort_events:
        proxy = event["proxy_strata"]
        height = float(proxy["minimum_positive_person_height_px"])
        fov = float(proxy["minimum_negative_fov_clearance_deg_beyond_buffer"])
        is_hard = height <= 64.0 or fov <= 10.0
        if not is_hard:
            continue
        event_id = event["event_id"]
        hard.append(event_id)
        height_severity = float(np.clip((64.0 - height) / 64.0, 0.0, 1.0))
        fov_severity = float(np.clip((10.0 - fov) / 10.0, 0.0, 1.0))
        reason = "height" if height_severity >= fov_severity else "fov"
        reasons[reason].append(event_id)
        details[event_id] = {"height_severity": height_severity, "fov_severity": fov_severity, "argmax_reason": reason}
    result = {
        "analysis_status": "prespecified_descriptive_nonconfirmatory_cannot_rescue_H1_or_H2",
        "definition": "minimum positive height <=64 px OR minimum negative FOV clearance <=10 degrees",
        "denominator_events": len(hard),
        "argmax_counts": {reason: len(values) for reason, values in reasons.items()},
        "comparisons": {},
        "reason_strata": {},
        "event_assignment": details,
    }
    for name, (left, right) in comparisons.items():
        result["comparisons"][name] = compare_outcomes(left, right, sorted(hard), confirmatory=False)
        result["reason_strata"][name] = {
            reason: compare_outcomes(left, right, sorted(event_ids), confirmatory=False)
            for reason, event_ids in reasons.items()
        }
    return result


def amendment_and_deviation_summary() -> dict[str, Any]:
    amendments = {}
    for number in range(1, 5):
        stem = f"amendment_{number:03d}"
        path = RUN / "status" / f"{stem}.json"
        sidecar = RUN / "status" / f"{stem}.sha256"
        expected = sidecar.read_text(encoding="utf-8").split()[0].lower()
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{stem} sidecar SHA mismatch")
        value = read_json(path)
        amendments[stem] = {
            "sha256": actual,
            "status": value["status"],
            "recorded_at": value["recorded_at"],
            "opening_state": value["opening_state"],
        }
    deviation_initial = read_json(RUN / "status" / "deviation.json")
    deviation_head = read_json(RUN / "status" / "deviation_resume_004.json")
    amendment_2 = read_json(RUN / "status" / "amendment_002.json")
    amendment_3 = read_json(RUN / "status" / "amendment_003.json")
    amendment_4 = read_json(RUN / "status" / "amendment_004.json")
    receipt = read_json(RUN / "status" / "resume_2_completion_receipt.json")
    return {
        "analysis_status": "registered_deviation_and_retry_record",
        "amendments": amendments,
        "attempts_and_deviations": {
            "initial_materialization": {
                "attempts_started": deviation_initial["materialize_attempts_started"],
                "status": deviation_initial["status"],
                "reason": deviation_initial["reason"],
            },
            "resume_002_host_environment": amendment_3["resume_002_operational_failure"],
            "resume_003_gpu_shard_environment": amendment_4["resume_003_operational_failure"],
            "head_fixed_setting_failure": deviation_head["failure"],
            "head_repair": amendment_2["repair"],
            "prefix_identity_gate": amendment_2["prefix_identity_gate"],
            "final_raw_seal_status": receipt["status"],
        },
        "unresolved_execution_failures_in_final_raw_files": 0,
    }


def build_report(results: dict[str, Any]) -> str:
    integrity = results["integrity"]
    h1 = results["confirmatory"]["H1"]
    h2 = results["confirmatory"]["H2"]
    lines = [
        "# B1 registered opening results",
        "",
        f"- Registration SHA-256: `{integrity['registration']['sha256']}`",
        f"- Raw integrity: **{integrity['status']}** (13/13 SHA-256 and byte counts; manifest/config seal SHA values matched)",
        f"- Bootstrap: {BOOTSTRAP_REPLICATES} paired event replicates, seed {BOOTSTRAP_SEED}, percentile linear quantiles",
        "- Confirmatory orientation: `answer_abstain_seed_20260713 - comparator`; verdicts use unrounded atomic values below.",
        "- Availability remains a pre-generation allocation rule, never a release gate.",
        "",
        "## Confirmatory verdicts",
        "",
        f"### H1: **{h1['verdict']}**",
        "",
    ]
    if h1["verdict"] == "FAIL" and h1["differences"]["correct_released_yield"]["delta"] > 0:
        lines.append("방향은 유지되었으나 재확증 기준은 충족하지 못했다.")
    elif h1["verdict"] == "PASS":
        lines.append("등록된 세 IUT 조건을 모두 충족했다.")
    else:
        lines.append("등록된 재확증 기준을 충족하지 못했다.")
    lines.extend(["", comparison_markdown(h1)])
    lines.extend(["", f"### H2: **{h2['verdict']}**", ""])
    if h2["verdict"] == "NOT_EVALUATED_CONFIRMATORILY":
        lines.append("H1이 세 조건을 모두 통과하지 않아 등록된 고정 순서에 따라 H2를 확인적으로 판정하지 않았다.")
    else:
        if h2["verdict"] == "FAIL" and h2["differences"]["correct_released_yield"]["delta"] > 0:
            lines.append("방향은 유지되었으나 재확증 기준은 충족하지 못했다.")
        elif h2["verdict"] == "PASS":
            lines.append("등록된 세 IUT 조건을 모두 충족했다.")
        else:
            lines.append("등록된 재확증 기준을 충족하지 못했다.")
        lines.extend(["", comparison_markdown(h2)])

    lines.extend(["", "## Three-seed robustness (secondary; not a gate)", ""])
    for comparison_name, comparison in results["seed_robustness"].items():
        lines.extend([f"### {comparison_name}", "", "| Layer | N | Yield delta (95% CI) | Wrong-risk delta (one-sided 95% upper) | Any-negative delta (one-sided 95% upper) |", "|---|---:|---:|---:|---:|"])
        for seed, item in comparison["per_seed"].items():
            lines.append(robustness_row(seed, item))
        lines.append(robustness_row("event mean over 3 seeds", comparison["event_seed_mean"]))
        worst = comparison["worst_seed_safety_upper"]
        lines.extend([
            "",
            f"Worst-seed upper bounds: wrong-release `{atom(worst['wrong_release_risk']['upper'])}` ({worst['wrong_release_risk']['seed']}), any-negative `{atom(worst['any_negative_violation_risk']['upper'])}` ({worst['any_negative_violation_risk']['seed']}).",
            "",
        ])

    lines.extend([
        "## Auxiliary arms (secondary/descriptive; not a gate)",
        "",
        "The Holm family contains the three answer-only release policies and the activity head, each compared with the primary exact policy on event correct-released yield using exact two-sided McNemar p-values.",
        "",
        "| Auxiliary arm | Correct-released / 260 | Wrong-release / 260 | Any-negative / 260 | Raw p | Holm-adjusted p |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm, item in results["auxiliary_arms"]["comparisons"].items():
        right = item["right"]
        holm = item["holm_correct_released_yield_family"]
        lines.append(
            f"| `{arm}` | {right['correct_released_yield']['numerator']}/260 | "
            f"{right['wrong_release_risk']['numerator']}/260 | {right['any_negative_violation_risk']['numerator']}/260 | "
            f"{atom(holm['raw_exact_two_sided_mcnemar_p'])} | {atom(holm['holm_adjusted_p'])} |"
        )
    zero = results["auxiliary_arms"]["zero_shot_record_only"]
    lines.extend([
        "",
        f"Zero-shot is record-only with no tau or test: frozen-parser answered {zero['selected_frozen_parser_answered']}/260 selected rows, of which {zero['selected_frozen_parser_answered_correct']}/260 were correct.",
        "",
    ])

    lines.extend(["## Stage 2/3 decomposition (all registered arms)", "", "All counts use 1,096 candidate-view rows; zero-shot has no registered release threshold, so Stage 3 is N/A.", "", "| Arm | tau | Unparsable | Answered correct/wrong | Below-tau correct/wrong | Released correct/wrong |", "|---|---:|---:|---:|---:|---:|"])
    for arm, summary in results["stage_decomposition"].items():
        row = summary["all_rows"]
        tau = "N/A" if summary["tau"] is None else atom(summary["tau"])
        below = "N/A" if row["stage3_below_tau"] is None else f"{row['stage3_below_tau']} ({row['stage3_below_tau_correct']}/{row['stage3_below_tau_wrong']})"
        released = "N/A" if row["stage3_released"] is None else f"{row['stage3_released']} ({row['stage3_released_correct']}/{row['stage3_released_wrong']})"
        lines.append(f"| `{arm}` | {tau} | {row['unparsable']}/1096 | {row['stage2_answered']} ({row['stage2_answered_correct']}/{row['stage2_answered_wrong']}) | {below} | {released} |")

    allocation = results["allocation_record"]["policies"]
    lines.extend(["", "## Allocation and measured cost (record-only; no test)", "", "| Policy | A(1) | U(1), primary exact policy | Router GPU s | Selected VLM GPU s | Total GPU s |", "|---|---:|---:|---:|---:|---:|"])
    for policy in ("availability", "task_matched_selector"):
        item = allocation[policy]
        cost = item["gpu_cost"]
        lines.append(f"| `{policy}` | {item['A_1_numerator_selected_positive']}/260 = {atom(item['A_1'])} | {item['U_1_numerator_correct_released']}/260 = {atom(item['U_1'])} | {atom(cost['router_gpu_seconds_all_1096_views'])} | {atom(cost['selected_primary_vlm_gpu_seconds_260_rows'])} | {atom(cost['total_gpu_seconds'])} |")
    gpu = results["gpu_cost_record"]
    feature = gpu["feature_cuda_timer_components"]
    head = gpu["successful_head_active_gpu_wall_accounting"]
    lines.extend([
        "",
        f"Measured feature CUDA timers (1,096 rows): Grounding-DINO `{atom(feature['grounding_dino_gpu_seconds'])}` s, SigLIP2 image `{atom(feature['siglip2_image_gpu_seconds'])}` s, SigLIP2 text `{atom(feature['siglip2_text_gpu_seconds'])}` s, DINOv2 `{atom(feature['dinov2_gpu_seconds'])}` s.",
        f"All eight full generation passes sum to `{atom(gpu['full_generation_cuda_timer_sum_seconds'])}` CUDA-timer seconds over 8 x 1,096 rows; per-pass values and checkpoint equality are in `b1_results.json`.",
        f"Successful resume-2 head accounting is `{head['successful_resume2_active_gpu_wall_seconds']}` active-GPU wall-seconds (2 x ({head['head_completion_wall_seconds']} + {head['prefix_identity_gate_wall_seconds']})); this is not a CUDA timer. Earlier failed attempts lack an exact additive GPU-seconds log and are not fabricated.",
    ])
    aux = results["integrity"]["task_matched_auxiliary_features"]
    if aux["status"] == "PASS":
        auxiliary_integrity_sentence = (
            f"Task-matched auxiliary feature integrity: **PASS**. SigLIP2/DINOv2 tree, DINO JSONL, "
            f"text-feature, and frozen selector SHA values all match their checkpoints; the auxiliary "
            f"embeddings also reproduce the sealed availability top-1 for "
            f"{aux['availability_recomputation_diagnostic']['top1_identical_events']}/"
            f"{aux['availability_recomputation_diagnostic']['denominator_events']} events. These feature "
            "files remain outside the 13-file raw manifest and are used only for the registered record-only allocation comparison."
        )
    else:
        auxiliary_integrity_sentence = (
            f"Task-matched auxiliary feature integrity: **{aux['status']}**. Its SigLIP2/DINOv2 tree "
            f"observed SHA `{aux['siglip2_dinov2_tree']['actual_sha256']}` differs from checkpoint "
            f"`{aux['siglip2_dinov2_tree']['expected_sha256']}`; these files were not among the 13 raw-sealed "
            f"files. DINO JSONL and text-feature hashes match, and the auxiliary embeddings reproduce the "
            f"sealed availability top-1 for {aux['availability_recomputation_diagnostic']['top1_identical_events']}/"
            f"{aux['availability_recomputation_diagnostic']['denominator_events']} events. Therefore task-matched "
            "figures remain record-only with this integrity limitation."
        )
    lines.extend([
        "",
        auxiliary_integrity_sentence,
        "",
        "## Prespecified descriptive layers",
        "",
        f"- Proxy-hard union: {results['stratified']['proxy_hard']['denominator_events']}/260 events; argmax reason height={results['stratified']['proxy_hard']['argmax_counts']['height']}, FOV={results['stratified']['proxy_hard']['argmax_counts']['fov']}. Exact ties were assigned to height.",
        f"- Source-clip connected-component bootstrap: {results['stratified']['cluster_bootstrap']['source_clip']['components']} components; 4,000 component replicates, seed 20260805.",
        f"- Site bootstrap: {results['stratified']['cluster_bootstrap']['site']['components']} sites; 4,000 site-cluster replicates, seed 20260805; descriptive because there are only three sites.",
        f"- Activity strata: {len(results['stratified']['activity']['activities'])} registered activities; descriptive rates and denominators are in `b1_results.json`.",
        "",
        "### Proxy-hard event bootstrap",
        "",
        "| Comparison | N | Yield delta (95% CI) | Wrong-risk delta (one-sided upper) | Any-negative delta (one-sided upper) |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, item in results["stratified"]["proxy_hard"]["comparisons"].items():
        lines.append(descriptive_difference_row(name, item))
    lines.extend([
        "",
        "### Cluster bootstrap",
        "",
        "| Resampling unit / comparison | Clusters | Yield delta (95% CI) | Wrong-risk delta (one-sided upper) | Any-negative delta (one-sided upper) |",
        "|---|---:|---:|---:|---:|",
    ])
    for unit in ("source_clip", "site"):
        cluster = results["stratified"]["cluster_bootstrap"][unit]
        for name in ("H1", "H2"):
            item = cluster[name]
            y = item["correct_released_yield"]
            w = item["wrong_release_risk"]
            n = item["any_negative_violation_risk"]
            lines.append(
                f"| {unit} / {name} | {y['clusters']} | {atom(y['delta'])} "
                f"[{atom(y['two_sided_95_ci'][0])}, {atom(y['two_sided_95_ci'][1])}] | "
                f"{atom(w['delta'])} ({atom(w['one_sided_95_upper'])}) | "
                f"{atom(n['delta'])} ({atom(n['one_sided_95_upper'])}) |"
            )
    lines.extend([
        "",
        "### Activity-stratified descriptive counts",
        "",
        "Cells are correct-released / wrong-released / any-negative counts; the row N is the denominator for every cell.",
        "",
        "| Activity | N | Primary | Explicit prompt | Correctness selector |",
        "|---|---:|---:|---:|---:|",
    ])
    for activity, item in results["stratified"]["activity"]["activities"].items():
        cells = []
        for arm in (PRIMARY, H1_COMPARATOR, H2_COMPARATOR):
            summary = item["arms"][arm]
            cells.append(
                f"{int(summary['correct_released_yield']['numerator'])}/"
                f"{int(summary['wrong_release_risk']['numerator'])}/"
                f"{int(summary['any_negative_violation_risk']['numerator'])}"
            )
        lines.append(f"| `{activity}` | {item['denominator_events']} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.extend([
        "",
        "## Transport sensitivity (nonconfirmatory)",
        "",
        "The primary tau `0.981128697326919` is copied without reselection to the registered stored DEV responses:",
        "",
        "| Target seed | Events/rows | P+ correct-released yield | E- release | P- release | Released/errors |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for seed, item in results["transport_sensitivity"]["metrics_by_target_seed"].items():
        lines.append(f"| {seed} | {item['events']}/{item['rows']} | {atom(item['pplus_correct_released_yield'])} | {atom(item['eminus_release_rate'])} | {atom(item['pminus_release_rate'])} | {item['released']}/{item['released_errors']} |")

    lines.extend(["", "## Strict unparsable frequency", "", "| Generation pass | Unparsable / 1,096 | Rate |", "|---|---:|---:|"])
    for arm, item in results["unparsable"].items():
        lines.append(f"| `{arm}` | {item['unparsable']}/1096 | {atom(item['rate'])} |")
    lines.extend([
        "",
        "The neutral zero-shot pass uses the frozen parser's stored non-abstention state only for the registered correctness-selector answer-source semantics. Its strict-wrapper unparsable count does not redefine H2.",
        "",
        "## Retry and deviation disclosure",
        "",
        "- Amendment 001 records the restored orientation-audit dependency and operator exit-code wrapper fix before opening.",
        "- Amendments 003/004 record rejected host/container launch environments before valid generation started.",
        "- The fixed head setting exhausted three attempts after 227 action-head and 221 correctness-head prefix rows.",
        "- Amendment 002 records the B1-only no-truncation repair, 448-row prefix identity gate with 0 mismatches, completed heads, and final raw seal.",
        "- No unresolved row is present in the final 13 raw files; the final files were sealed before this aggregation.",
        "",
        "## Claim boundary",
        "",
        "This is an offline sealed-cohort result. It does not claim human semantic sufficiency, new-site/camera transfer, formal risk guarantees, deployment safety, online latency/throughput, or availability-based release gating.",
        "",
        f"Analysis code SHA-256: `{results['analysis_code']['sha256']}`",
        "",
    ])
    return "\n".join(lines)


def comparison_markdown(result: dict[str, Any]) -> str:
    left = result["left"]
    right = result["right"]
    diff = result["differences"]
    criteria = result["criteria"]
    names = (
        ("Correct-released yield", "correct_released_yield", "two-sided 95% CI lower > 0"),
        ("Selected-row wrong-release risk", "wrong_release_risk", "one-sided 95% upper <= 0.05"),
        ("Event any-negative-violation risk", "any_negative_violation_risk", "one-sided 95% upper <= 0.05"),
    )
    lines = [
        "| Condition | Primary numerator/N (rate) | Comparator numerator/N (rate) | Delta and registered interval/bound | Pass |",
        "|---|---:|---:|---:|---:|",
    ]
    criterion_keys = list(criteria)
    for index, (label, key, rule) in enumerate(names):
        left_item, right_item, difference = left[key], right[key], diff[key]
        interval = (
            f"95% CI [{atom(difference['two_sided_95_ci'][0])}, {atom(difference['two_sided_95_ci'][1])}]"
            if key == "correct_released_yield"
            else f"one-sided 95% upper {atom(difference['one_sided_95_upper'])}"
        )
        lines.append(
            f"| {label} ({rule}) | {left_item['numerator']}/260 ({atom(left_item['rate'])}) | "
            f"{right_item['numerator']}/260 ({atom(right_item['rate'])}) | {atom(difference['delta'])}; {interval} | "
            f"{'PASS' if criteria[criterion_keys[index]] else 'FAIL'} |"
        )
    pair = result["paired_binary_correct_released_yield"]
    counts = pair["counts"]
    lines.extend([
        "",
        f"Descriptive paired companion: n11/n10/n01/n00 = {counts['n11']}/{counts['n10']}/{counts['n01']}/{counts['n00']}; Newcombe method-10 95% CI [{atom(pair['newcombe_method10_two_sided_95_ci'][0])}, {atom(pair['newcombe_method10_two_sided_95_ci'][1])}]; exact two-sided McNemar p={atom(pair['exact_two_sided_mcnemar_p'])}. This is not an additional condition.",
    ])
    return "\n".join(lines)


def robustness_row(label: str, result: dict[str, Any]) -> str:
    diff = result["differences"]
    y = diff["correct_released_yield"]
    w = diff["wrong_release_risk"]
    n = diff["any_negative_violation_risk"]
    return f"| {label} | {result['denominator_events']} | {atom(y['delta'])} [{atom(y['two_sided_95_ci'][0])}, {atom(y['two_sided_95_ci'][1])}] | {atom(w['delta'])} ({atom(w['one_sided_95_upper'])}) | {atom(n['delta'])} ({atom(n['one_sided_95_upper'])}) |"


def descriptive_difference_row(label: str, result: dict[str, Any]) -> str:
    diff = result["differences"]
    y = diff["correct_released_yield"]
    w = diff["wrong_release_risk"]
    n = diff["any_negative_violation_risk"]
    return (
        f"| {label} | {result['denominator_events']} | {atom(y['delta'])} "
        f"[{atom(y['two_sided_95_ci'][0])}, {atom(y['two_sided_95_ci'][1])}] | "
        f"{atom(w['delta'])} ({atom(w['one_sided_95_upper'])}) | "
        f"{atom(n['delta'])} ({atom(n['one_sided_95_upper'])}) |"
    )


def main() -> None:
    integrity = verify_raw_integrity()
    registration = read_json(REGISTRATION)
    thresholds = read_json(THRESHOLDS)
    cohort = read_json(COHORT)
    cohort_events = cohort["events"]
    if len(cohort_events) != EXPECTED_EVENTS:
        raise RuntimeError("cohort event count drift")

    rows = read_jsonl(RUN / "candidate_views.materialized.jsonl")
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError("candidate row count drift")
    row_ids = {row["row_id"] for row in rows}
    if len(row_ids) != EXPECTED_ROWS:
        raise RuntimeError("candidate row identity drift")
    candidates = {row["row_id"]: row for row in rows}
    rows_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_event[row["event_id"]].append(row)
    if len(rows_by_event) != EXPECTED_EVENTS or any(sum(row["role"] == "positive" for row in values) != 1 for values in rows_by_event.values()):
        raise RuntimeError("event structure drift")
    event_ids = sorted(rows_by_event)

    availability_rows = read_jsonl(RUN / "features_host" / "m4_availability_scores.jsonl")
    availability = {row["row_id"]: float(row["availability_score"]) for row in availability_rows}
    if len(availability_rows) != EXPECTED_ROWS or set(availability) != row_ids:
        raise RuntimeError("availability row coverage drift")
    availability_selection = {
        event_id: min(event_rows, key=lambda row: (-availability[row["row_id"]], row["camera_id"]))["row_id"]
        for event_id, event_rows in rows_by_event.items()
    }

    generation_raw = {
        arm: index_rows(read_jsonl(RUN / "answers" / f"{arm}.jsonl"), arm, row_ids) for arm in GENERATION_ARMS
    }
    head_raw = {arm: index_rows(read_jsonl(RUN / "heads" / f"{arm}.jsonl"), arm, row_ids) for arm in HEAD_ARMS}

    tau = {
        "answer_abstain_seed_20260713": 0.981128697326919,
        "answer_abstain_seed_20260714": 0.9722987314076553,
        "answer_abstain_seed_20260715": 0.9662443411375032,
        "answer_only_seed_20260713": 0.9999968725353879,
        "answer_only_seed_20260714": 0.9999948179767415,
        "answer_only_seed_20260715": 0.9999959819939936,
        "explicit_abstention_prompt": 0.9491225851331859,
        "zero_shot": None,
        "correctness_selector": 0.9999998807907104,
        "action_classification_head": 0.9994823932647705,
    }
    generation = {}
    unparsable = {}
    for arm in GENERATION_ARMS:
        generation[arm], unparsable[arm] = generation_metrics(arm, generation_raw[arm], tau[arm], candidates)
    heads = {
        arm: head_metrics(arm, head_raw[arm], candidates, generation_raw["zero_shot"], float(tau[arm]))
        for arm in HEAD_ARMS
    }
    all_metrics = generation | heads
    stage = {arm: stage_summary(metrics, candidates, tau[arm]) for arm, metrics in all_metrics.items()}
    outcomes = {arm: event_outcomes(metrics, availability_selection, rows_by_event) for arm, metrics in all_metrics.items() if tau[arm] is not None}

    h1 = compare_outcomes(outcomes[PRIMARY], outcomes[H1_COMPARATOR], event_ids, confirmatory=True)
    h1["comparison"] = f"{PRIMARY} minus {H1_COMPARATOR}"
    if h1["verdict"] == "PASS":
        h2 = compare_outcomes(outcomes[PRIMARY], outcomes[H2_COMPARATOR], event_ids, confirmatory=True)
        h2["comparison"] = f"{PRIMARY} minus {H2_COMPARATOR}"
    else:
        h2 = {
            "comparison": f"{PRIMARY} minus {H2_COMPARATOR}",
            "analysis_status": "not_evaluated_confirmatorily_due_fixed_sequence",
            "verdict": "NOT_EVALUATED_CONFIRMATORILY",
            "gate_reason": "H1 did not pass all three registered IUT conditions",
        }

    seed_robustness = {}
    for name, comparator in (("H1 comparator", H1_COMPARATOR), ("H2 comparator", H2_COMPARATOR)):
        per_seed = {
            str(seed): compare_outcomes(outcomes[f"answer_abstain_seed_{seed}"], outcomes[comparator], event_ids, confirmatory=False)
            for seed in (20260713, 20260714, 20260715)
        }
        seed_mean = mean_seed_outcomes([outcomes[arm] for arm in ANSWER_ABSTAIN_ARMS], event_ids)
        mean_result = compare_outcomes(seed_mean, outcomes[comparator], event_ids, confirmatory=False)
        worst = {}
        for metric in ("wrong_release_risk", "any_negative_violation_risk"):
            seed, item = max(
                ((seed, result["differences"][metric]["one_sided_95_upper"]) for seed, result in per_seed.items()),
                key=lambda pair: (pair[1], pair[0]),
            )
            worst[metric] = {"seed": seed, "upper": item}
        seed_robustness[name] = {
            "analysis_status": "secondary_robustness_not_confirmatory",
            "comparator": comparator,
            "per_seed": per_seed,
            "event_seed_mean": mean_result,
            "worst_seed_safety_upper": worst,
        }

    auxiliary_comparisons = {}
    pvalues = {}
    for arm in (*ANSWER_ONLY_ARMS, "action_classification_head"):
        comparison = compare_outcomes(outcomes[PRIMARY], outcomes[arm], event_ids, confirmatory=False)
        auxiliary_comparisons[arm] = comparison
        pvalues[arm] = comparison["paired_binary_correct_released_yield"]["exact_two_sided_mcnemar_p"]
    holm = holm_adjust(pvalues)
    for arm, correction in holm.items():
        auxiliary_comparisons[arm]["holm_correct_released_yield_family"] = correction
    zero_selected = [generation["zero_shot"][availability_selection[event_id]] for event_id in event_ids]
    auxiliary = {
        "analysis_status": "secondary_or_descriptive_no_recycled_alpha",
        "holm_family_definition": "Primary exact-policy correct-released outcome versus three answer_only release policies and the activity head; four exact two-sided McNemar p-values.",
        "comparisons": auxiliary_comparisons,
        "zero_shot_record_only": {
            "denominator_events": EXPECTED_EVENTS,
            "selected_frozen_parser_answered": sum(item["answered"] for item in zero_selected),
            "selected_frozen_parser_answered_correct": sum(item["answered"] and item["correct"] for item in zero_selected),
            "registered_release_threshold": None,
            "hypothesis_test": None,
        },
    }

    task_scores, task_integrity = feature_and_task_scores(rows, rows_by_event, availability)
    integrity["task_matched_auxiliary_features"] = task_integrity
    task_selection = {
        event_id: min(event_rows, key=lambda row: (-task_scores[row["row_id"]], row["camera_id"]))["row_id"]
        for event_id, event_rows in rows_by_event.items()
    }
    features_checkpoint = read_json(RUN / "checkpoints" / "features.json")
    allocation = allocation_record(
        {"availability": availability_selection, "task_matched_selector": task_selection},
        candidates,
        generation[PRIMARY],
        generation_raw[PRIMARY],
        features_checkpoint,
    )

    full_generation_cost = {}
    for arm in GENERATION_ARMS:
        raw_sum = sum(float(row["vlm_gpu_seconds"]) for row in generation_raw[arm].values())
        checkpoint = read_json(RUN / "checkpoints" / f"answerer.{arm}.json")
        full_generation_cost[arm] = {
            "rows": EXPECTED_ROWS,
            "raw_sum_vlm_gpu_seconds": raw_sum,
            "checkpoint_vlm_gpu_seconds": float(checkpoint["vlm_gpu_seconds"]),
            "difference": raw_sum - float(checkpoint["vlm_gpu_seconds"]),
        }
    receipt = read_json(RUN / "status" / "resume_2_completion_receipt.json")
    gpu_cost = {
        "analysis_status": "record_only_no_test_no_deployment_claim",
        "feature_cuda_timer_components": {
            key: float(features_checkpoint[key])
            for key in ("grounding_dino_gpu_seconds", "siglip2_image_gpu_seconds", "siglip2_text_gpu_seconds", "dinov2_gpu_seconds")
        },
        "full_generation_cuda_timer_components": full_generation_cost,
        "full_generation_cuda_timer_sum_seconds": sum(item["raw_sum_vlm_gpu_seconds"] for item in full_generation_cost.values()),
        "successful_head_active_gpu_wall_accounting": {
            "head_completion_wall_seconds": int(receipt["stages"]["head_completion"]["wall_seconds"]),
            "prefix_identity_gate_wall_seconds": int(receipt["stages"]["prefix_identity_gate"]["wall_seconds"]),
            "active_gpus": 2,
            "successful_resume2_active_gpu_wall_seconds": 2
            * (
                int(receipt["stages"]["head_completion"]["wall_seconds"])
                + int(receipt["stages"]["prefix_identity_gate"]["wall_seconds"])
            ),
            "note": "wall-time x active GPUs, not a CUDA device timer; earlier failed attempts lack an exact additive GPU-seconds log and are not fabricated",
        },
    }

    clip_components = connected_components(cohort_events)
    sites: dict[str, list[str]] = defaultdict(list)
    for event in cohort_events:
        sites[event["site"]].append(event["event_id"])
    site_clusters = [sorted(values) for _, values in sorted(sites.items())]
    clustered = {
        "source_clip": {
            "components": len(clip_components),
            "component_event_counts": sorted((len(component) for component in clip_components), reverse=True),
            "H1": clustered_comparison(outcomes[PRIMARY], outcomes[H1_COMPARATOR], clip_components, "source_clip_connected_component"),
            "H2": clustered_comparison(outcomes[PRIMARY], outcomes[H2_COMPARATOR], clip_components, "source_clip_connected_component"),
        },
        "site": {
            "components": len(site_clusters),
            "site_event_counts": {site: len(values) for site, values in sorted(sites.items())},
            "H1": clustered_comparison(outcomes[PRIMARY], outcomes[H1_COMPARATOR], site_clusters, "site_cluster"),
            "H2": clustered_comparison(outcomes[PRIMARY], outcomes[H2_COMPARATOR], site_clusters, "site_cluster"),
        },
    }
    activity = activity_descriptives(
        cohort_events,
        {PRIMARY: outcomes[PRIMARY], H1_COMPARATOR: outcomes[H1_COMPARATOR], H2_COMPARATOR: outcomes[H2_COMPARATOR]},
    )
    proxy = proxy_strata(
        cohort_events,
        {
            "H1": (outcomes[PRIMARY], outcomes[H1_COMPARATOR]),
            "H2": (outcomes[PRIMARY], outcomes[H2_COMPARATOR]),
        },
    )

    transport = thresholds["primary_tau_transport_to_new_seeds"] | {
        "analysis_status": "nonconfirmatory_transport_sensitivity",
        "source": "registered threshold artifact; stored 246-row DEV responses, not B1 outcome reselection",
    }
    deviations = amendment_and_deviation_summary()

    results = {
        "schema_version": "meva_b1_registered_open_results_v1",
        "status": "OPENED_REGISTERED_ANALYSIS_COMPLETE",
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrity": integrity,
        "analysis_contract": {
            "registration_sha256": EXPECTED_REGISTRATION_SHA256,
            "confirmatory_allocation": "availability top-1; descending availability_score, camera_id ascending tie-break",
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "primary_unit": "event",
                "two_sided_percentiles": [0.025, 0.975],
                "one_sided_upper_percentile": 0.95,
                "quantile_method": "linear",
            },
            "safety_margin": SAFETY_MARGIN,
            "H1_to_H2_fixed_sequence": True,
            "availability_is_not_a_release_gate": True,
            "exploratory_analysis_added": False,
            "registration_claim_contract": registration["claim_contract"],
        },
        "confirmatory": {"H1": h1, "H2": h2},
        "seed_robustness": seed_robustness,
        "auxiliary_arms": auxiliary,
        "allocation_record": allocation,
        "gpu_cost_record": gpu_cost,
        "stage_decomposition": stage,
        "stratified": {"activity": activity, "proxy_hard": proxy, "cluster_bootstrap": clustered},
        "transport_sensitivity": transport,
        "unparsable": unparsable,
        "retry_and_deviation": deviations,
        "analysis_code": {"path": str(Path(__file__).resolve().relative_to(REPO)), "sha256": sha256_file(Path(__file__).resolve())},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "b1_results.json"
    report_path = OUT / "b1_results_report.md"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(results), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": results["status"],
                "H1": h1["verdict"],
                "H2": h2["verdict"],
                "results": str(result_path),
                "report": str(report_path),
                "analysis_code_sha256": results["analysis_code"]["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
