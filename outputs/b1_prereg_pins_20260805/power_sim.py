"""Reproduce the B0 paired table and simulate B1 paired-test power on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


N_EVENTS = 260
ALPHA = 0.05
MONTE_CARLO_TRIALS = 10_000
BOOTSTRAP_REPLICATES = 4_000
SIMULATION_SEED = 20260805
TARGET_DELTAS = (0.30, 0.15, 0.10)
RELEASE_THRESHOLDS = {
    "answer_abstain_sft": 0.981128697326919,
    "explicit_abstention_prompt": 0.9491225851331859,
}
EXPECTED_PAIR_COUNTS = {"n11": 162, "n10": 83, "n01": 0, "n00": 15}
EXPECTED_SOURCE_SHA256 = {
    "candidate_views": "3c62245a37b536af52dd0c0b1a3dd5eb3181101182d17bc263b02fa4350a6170",
    "availability_scores": "8872d1b01ef873c67900ae5b664ad5be9520ec8a74a2a6ae66f6cf591b585ef7",
    "answer_abstain_sft": "9943fd38d217d141b25f090d0b4e92f45b25828aa94ad1b8a56bf4e8d052b40a",
    "explicit_abstention_prompt": "1f58d11a96588837bc5e7accf7d763d7a8b123da80b21e41cf6ad9ad15b23039",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_mcnemar_pvalue(n10: int, n01: int) -> float:
    discordant = n10 + n01
    if discordant == 0:
        return 1.0
    lower = min(n10, n01)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) * 0.5**discordant
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(
    counts: tuple[int, int, int, int], rng: np.random.Generator
) -> tuple[float, float]:
    n11, n10, n01, n00 = counts
    probabilities = np.asarray([n01, n11 + n00, n10], dtype=np.float64) / sum(counts)
    bootstrap_counts = rng.multinomial(
        sum(counts), probabilities, size=BOOTSTRAP_REPLICATES
    )
    estimates = (bootstrap_counts[:, 2] - bootstrap_counts[:, 0]) / sum(counts)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def build_b0_pair_table(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, str]]:
    root = repo_root / "outputs" / "meva_full_20260713" / "w4_oneshot_20260714"
    paths = {
        "candidate_views": root / "candidate_views.materialized.jsonl",
        "availability_scores": root / "features_host" / "m4_availability_scores.jsonl",
        "answer_abstain_sft": root / "answers" / "answer_abstain_sft.jsonl",
        "explicit_abstention_prompt": root / "answers" / "explicit_abstention_prompt.jsonl",
    }
    observed_sha256 = {name: sha256_file(path) for name, path in paths.items()}
    if observed_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"B0 source hash drift: expected={EXPECTED_SOURCE_SHA256} observed={observed_sha256}"
        )

    candidates = read_jsonl(paths["candidate_views"])
    availability = {
        row["row_id"]: float(row["availability_score"])
        for row in read_jsonl(paths["availability_scores"])
    }
    answers = {
        arm: {row["row_id"]: row for row in read_jsonl(paths[arm])}
        for arm in RELEASE_THRESHOLDS
    }
    row_ids = {row["row_id"] for row in candidates}
    if set(availability) != row_ids or any(set(rows) != row_ids for rows in answers.values()):
        raise RuntimeError("B0 paired-table input coverage drift")

    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_event[row["event_id"]].append(row)
    if len(by_event) != N_EVENTS:
        raise RuntimeError(f"expected {N_EVENTS} B0 events, observed {len(by_event)}")

    table: list[dict[str, Any]] = []
    pair_counts: Counter[tuple[int, int]] = Counter()
    for event_id, event_rows in sorted(by_event.items()):
        selected = min(
            event_rows,
            key=lambda row: (-availability[row["row_id"]], row["camera_id"]),
        )
        outcomes: dict[str, int] = {}
        details: dict[str, Any] = {}
        for arm, threshold in RELEASE_THRESHOLDS.items():
            answer = answers[arm][selected["row_id"]]
            released = (
                not bool(answer["was_abstained"])
                and float(answer["confidence"]) >= threshold
            )
            correct_release = int(
                selected["role"] == "positive"
                and released
                and bool(answer["target_correct"])
            )
            outcomes[arm] = correct_release
            details[f"{arm}_released"] = bool(released)
            details[f"{arm}_confidence"] = float(answer["confidence"])
        pair_counts[(outcomes["answer_abstain_sft"], outcomes["explicit_abstention_prompt"])] += 1
        table.append(
            {
                "event_id": event_id,
                "selected_row_id": selected["row_id"],
                "selected_role": selected["role"],
                "availability_score": availability[selected["row_id"]],
                "answer_abstain_sft_correct_release": outcomes["answer_abstain_sft"],
                "explicit_abstention_prompt_correct_release": outcomes[
                    "explicit_abstention_prompt"
                ],
                **details,
            }
        )

    counts = {
        "n11": pair_counts[(1, 1)],
        "n10": pair_counts[(1, 0)],
        "n01": pair_counts[(0, 1)],
        "n00": pair_counts[(0, 0)],
    }
    if counts != EXPECTED_PAIR_COUNTS or sum(counts.values()) != N_EVENTS:
        raise RuntimeError(f"B0 paired-table invariant failed: {counts}")
    return table, counts, observed_sha256


def rejection_summary(rejections: int) -> dict[str, float | int]:
    power = rejections / MONTE_CARLO_TRIALS
    standard_error = math.sqrt(power * (1.0 - power) / MONTE_CARLO_TRIALS)
    return {
        "rejections": rejections,
        "trials": MONTE_CARLO_TRIALS,
        "power": power,
        "monte_carlo_standard_error": standard_error,
        "approx_95pct_half_width": 1.96 * standard_error,
    }


def simulate_scenario(
    target_delta: float, seed_sequence: np.random.SeedSequence, observed_counts: dict[str, int]
) -> dict[str, Any]:
    n = sum(observed_counts.values())
    total_discordance = (observed_counts["n10"] + observed_counts["n01"]) / n
    if not 0.0 <= target_delta <= total_discordance:
        raise ValueError("target delta must lie within the preserved discordant mass")

    # Preserve B0 concordant mass and total discordance, then rebalance the two
    # discordant cells so p10 - p01 equals the attenuated target difference.
    probabilities = np.asarray(
        [
            observed_counts["n11"] / n,
            (total_discordance + target_delta) / 2.0,
            (total_discordance - target_delta) / 2.0,
            observed_counts["n00"] / n,
        ],
        dtype=np.float64,
    )
    if not np.isclose(probabilities.sum(), 1.0):
        raise RuntimeError("scenario probabilities do not sum to one")

    simulation_seed, bootstrap_seed = seed_sequence.spawn(2)
    simulation_rng = np.random.default_rng(simulation_seed)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    simulated = simulation_rng.multinomial(n, probabilities, size=MONTE_CARLO_TRIALS)

    mcnemar_rejections = 0
    bootstrap_rejections = 0
    for n11, n10, n01, n00 in simulated:
        n11_i, n10_i, n01_i, n00_i = map(int, (n11, n10, n01, n00))
        if n10_i > n01_i and exact_mcnemar_pvalue(n10_i, n01_i) < ALPHA:
            mcnemar_rejections += 1
        ci_low, _ = paired_bootstrap_ci(
            (n11_i, n10_i, n01_i, n00_i), bootstrap_rng
        )
        if ci_low > 0.0:
            bootstrap_rejections += 1

    return {
        "target_delta_probability_points": target_delta,
        "joint_probabilities": {
            "p11": float(probabilities[0]),
            "p10": float(probabilities[1]),
            "p01": float(probabilities[2]),
            "p00": float(probabilities[3]),
        },
        "implied_marginals": {
            "answer_abstain_sft": float(probabilities[0] + probabilities[1]),
            "explicit_abstention_prompt": float(probabilities[0] + probabilities[2]),
        },
        "exact_two_sided_mcnemar": rejection_summary(mcnemar_rejections),
        "paired_bootstrap_two_sided_95pct_ci_lower_gt_zero": rejection_summary(
            bootstrap_rejections
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("power_sim_results.json"),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    pair_table, counts, source_sha256 = build_b0_pair_table(repo_root)

    master_seed = np.random.SeedSequence(SIMULATION_SEED)
    observed_seed, *scenario_seeds = master_seed.spawn(1 + len(TARGET_DELTAS))
    observed_ci = paired_bootstrap_ci(
        (counts["n11"], counts["n10"], counts["n01"], counts["n00"]),
        np.random.default_rng(observed_seed),
    )
    results = {
        "schema_version": "b1_prereg_power_sim_v1",
        "cpu_only": True,
        "source": {
            "campaign": "B0 W4 availability U(1)",
            "release_thresholds": RELEASE_THRESHOLDS,
            "sha256": source_sha256,
        },
        "b0_paired_table": {
            "events": len(pair_table),
            "counts": counts,
            "marginals": {
                "answer_abstain_sft": counts["n11"] + counts["n10"],
                "explicit_abstention_prompt": counts["n11"] + counts["n01"],
            },
            "observed_delta": (counts["n10"] - counts["n01"]) / N_EVENTS,
            "exact_two_sided_mcnemar_pvalue": exact_mcnemar_pvalue(
                counts["n10"], counts["n01"]
            ),
            "paired_bootstrap_two_sided_95pct_ci": list(observed_ci),
            "event_pairs": pair_table,
        },
        "simulation": {
            "sample_size_events": N_EVENTS,
            "alpha": ALPHA,
            "monte_carlo_trials": MONTE_CARLO_TRIALS,
            "paired_bootstrap_replicates_per_trial": BOOTSTRAP_REPLICATES,
            "seed": SIMULATION_SEED,
            "numpy_version": np.__version__,
            "attenuation_rule": (
                "Preserve B0 p11, p00, and total discordance p10+p01; "
                "rebalance p10 and p01 so p10-p01 equals the target delta."
            ),
            "scenarios": [
                simulate_scenario(delta, seed, counts)
                for delta, seed in zip(TARGET_DELTAS, scenario_seeds, strict=True)
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"output": str(args.output), "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
