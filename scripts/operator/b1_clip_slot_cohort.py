"""Build the metadata-only B1 clip/slot-disjoint cohort and reserve.

This script deliberately imports the frozen W4 metadata gate instead of
reimplementing annotation, synchronization, KRTD, incident matching, frame,
or geometric-negative rules.  It never opens video, image, model-output, or
sealed B1 input files.

The reserve order is frozen here.  Candidates are ordered by (1) whether a
single loss in their activity would reach the eight-event floor, (2) smaller
activity slack above that floor, (3) SHA-256 of the B1 reserve seed and event
ID, then (4) event ID.  At execution time, remove the failed active event and
scan this static list from the beginning; accept the first unused candidate
for which ``cohort_constraints_hold`` is true.  An unresolved failure is not
silently repaired and must become non-release under the registered protocol.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import w4_metadata_gate as w4

B1_SELECTION_SEED = w4.SELECTION_SEED
B1_RESERVE_SEED = "meva-b1-clip-slot-reserve-20260805-v1"
EXPECTED_MEVA_COMMIT = "421841a75577b697c314e952e585aecbb1b99e17"
EXPECTED_PRIOR_CLIPS = 641
EXPECTED_PRIOR_SITE_SLOTS = 160
EXPECTED_ELIGIBLE_EVENTS = 714
EXPECTED_ELIGIBLE_VIEWS = 3354
TARGET_EVENTS = 260
MIN_EVENTS_PER_ACTIVITY = 8
MAX_ACTIVITY_EVENTS = 70  # 70/260 = 26.9230769%, reported as 26.92%.
MAX_CAMERA_EVENTS = 104  # 104/260 = 40%.
MIN_N_CAND_GE_3 = 200
MIN_N_CAND_GE_4 = 70
MIN_TWO_ROLE_CAMERAS = 6
MIN_LOCATION_DATE_BLOCKS = 3
LOW_HEIGHT_PROXY_PX = 64.0
BOUNDARY_PROXY_DEG = 10.0


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


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
    return sha256_bytes(payload)


def normalize_clip(value: str) -> str:
    clip = Path(value.strip()).name
    if clip.endswith(".r13.avi"):
        clip = clip[: -len(".r13.avi")]
    if not clip:
        raise ValueError("empty clip identifier")
    return clip


def site_slot(clip_value: str) -> str:
    """Return site|YYYY-MM-DD.HH-(floor(MM/5)*5)-00; camera is excluded."""
    clip = normalize_clip(clip_value)
    fields = clip.split(".")
    if len(fields) < 5:
        raise ValueError(f"malformed MEVA clip: {clip}")
    date, start, _, site = fields[0], fields[1], fields[2], fields[-2]
    time_fields = start.split("-")
    if len(time_fields) != 3:
        raise ValueError(f"malformed MEVA clip start time: {clip}")
    hour, minute, _ = map(int, time_fields)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"out-of-range MEVA clip start time: {clip}")
    floored_minute = (minute // 5) * 5
    return f"{site}|{date}.{hour:02d}-{floored_minute:02d}-00"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_sha(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"SHA-256 drift for {path}: {observed} != {expected}")


def csv_clips(path: Path, fields: list[str]) -> set[str]:
    clips: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in fields if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing clip fields in {path}: {missing}")
        for row in reader:
            for field in fields:
                if row[field].strip():
                    clips.add(normalize_clip(row[field]))
    return clips


def jsonl_clips(path: Path, fields: list[str]) -> set[str]:
    clips: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            row = json.loads(line)
            for field in fields:
                if field not in row:
                    raise ValueError(f"missing {field} at {path}:{lineno}")
                if row[field]:
                    clips.add(normalize_clip(str(row[field])))
    return clips


def manifest_clips(path: Path) -> set[str]:
    payload = load_json(path)
    return {
        normalize_clip(view["clip"])
        for event in payload["events"]
        for view in event["views"]
    }


def conservative_exclusion(pins_path: Path) -> tuple[list[str], list[Path], list[dict[str, Any]]]:
    pins = load_json(pins_path)
    entries = pins["P4_cohort"]["conservative_exclusion_clip_list"]["reproduction_inputs"]
    by_name = {entry["name"]: entry for entry in entries}
    required = {
        "pilot",
        "superseded_v2",
        "superseded_v2_pairs",
        "canonical_v4_train_dev_test",
        "w4_train_dev_all_candidate_comparator",
        "w4_sealed_260",
    }
    if set(by_name) != required:
        raise RuntimeError(f"unexpected exclusion input set: {sorted(by_name)}")

    resolved: dict[str, Path] = {}
    for name, entry in by_name.items():
        path = (ROOT / entry["path"]).resolve()
        verify_sha(path, entry["sha256"])
        resolved[name] = path

    pilot = csv_clips(resolved["pilot"], by_name["pilot"]["fields"])
    v2 = csv_clips(resolved["superseded_v2"], by_name["superseded_v2"]["fields"])
    v2 |= csv_clips(resolved["superseded_v2_pairs"], by_name["superseded_v2_pairs"]["fields"])
    v4 = csv_clips(
        resolved["canonical_v4_train_dev_test"],
        by_name["canonical_v4_train_dev_test"]["fields"],
    )
    comparator = jsonl_clips(
        resolved["w4_train_dev_all_candidate_comparator"],
        by_name["w4_train_dev_all_candidate_comparator"]["fields"],
    )
    sealed = manifest_clips(resolved["w4_sealed_260"])

    cumulative: set[str] = set()
    audit: list[dict[str, Any]] = []
    for history, values, expected in (
        ("pilot", pilot, 12),
        ("superseded_v2", v2, 266),
        ("canonical_v4_train_dev_test", v4, 305),
        ("w4_train_dev_all_candidate_comparator", comparator, 352),
        ("w4_sealed_260", sealed, 641),
    ):
        cumulative.update(values)
        audit.append(
            {
                "history": history,
                "history_unique_clips": len(values),
                "cumulative_unique_clips": len(cumulative),
            }
        )
        if len(cumulative) != expected:
            raise RuntimeError(
                f"exclusion union drift after {history}: {len(cumulative)} != {expected}"
            )
    return sorted(cumulative), [resolved[name] for name in sorted(resolved)], audit


def add_constraint(
    rows: list[dict[int, float]],
    lower: list[float],
    upper: list[float],
    coefficients: dict[int, float],
    lb: float,
    ub: float,
) -> None:
    rows.append(coefficients)
    lower.append(lb)
    upper.append(ub)


def solve_selection(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        events,
        key=lambda event: (
            hashlib.sha256(
                f"{B1_SELECTION_SEED}|event|{event['selection_source_event_id']}".encode()
            ).hexdigest(),
            event["selection_source_event_id"],
        ),
    )
    n_events = len(ordered)
    activities = list(w4.ACTIVITIES)
    blocks = sorted({event["location_date_block"] for event in ordered})
    cameras = sorted({view["camera_id"] for event in ordered for view in event["views"]})
    block_var = {block: n_events + index for index, block in enumerate(blocks)}
    camera_offset = n_events + len(blocks)
    camera_vars = {
        camera: (
            camera_offset + 3 * index,
            camera_offset + 3 * index + 1,
            camera_offset + 3 * index + 2,
        )
        for index, camera in enumerate(cameras)
    }
    variable_count = camera_offset + 3 * len(cameras)
    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []
    inf = np.inf

    add_constraint(rows, lower, upper, {i: 1.0 for i in range(n_events)}, TARGET_EVENTS, TARGET_EVENTS)
    add_constraint(
        rows,
        lower,
        upper,
        {i: 1.0 for i, event in enumerate(ordered) if event["n_cand"] >= 3},
        MIN_N_CAND_GE_3,
        inf,
    )
    add_constraint(
        rows,
        lower,
        upper,
        {i: 1.0 for i, event in enumerate(ordered) if event["n_cand"] >= 4},
        MIN_N_CAND_GE_4,
        inf,
    )
    for activity in activities:
        coefficients = {
            i: 1.0 for i, event in enumerate(ordered) if event["activity"] == activity
        }
        if len(coefficients) < MIN_EVENTS_PER_ACTIVITY:
            raise RuntimeError(f"activity pool below frozen floor: {activity}")
        add_constraint(
            rows,
            lower,
            upper,
            coefficients,
            MIN_EVENTS_PER_ACTIVITY,
            MAX_ACTIVITY_EVENTS,
        )

    for block in blocks:
        coefficients = {
            i: 1.0 for i, event in enumerate(ordered) if event["location_date_block"] == block
        }
        y = block_var[block]
        add_constraint(rows, lower, upper, coefficients | {y: -1.0}, 0.0, inf)
        add_constraint(rows, lower, upper, coefficients | {y: -float(TARGET_EVENTS)}, -inf, 0.0)
    add_constraint(
        rows,
        lower,
        upper,
        {block_var[block]: 1.0 for block in blocks},
        MIN_LOCATION_DATE_BLOCKS,
        inf,
    )

    for camera in cameras:
        incidence = {
            i: 1.0
            for i, event in enumerate(ordered)
            if camera in {view["camera_id"] for view in event["views"]}
        }
        add_constraint(rows, lower, upper, incidence, -inf, MAX_CAMERA_EVENTS)
        positive = {
            i: 1.0
            for i, event in enumerate(ordered)
            if any(
                view["camera_id"] == camera and view["role"] == "positive"
                for view in event["views"]
            )
        }
        negative = {
            i: 1.0
            for i, event in enumerate(ordered)
            if any(
                view["camera_id"] == camera and view["role"] == "negative_geometric"
                for view in event["views"]
            )
        }
        p_var, n_var, both_var = camera_vars[camera]
        add_constraint(rows, lower, upper, positive | {p_var: -1.0}, 0.0, inf)
        add_constraint(rows, lower, upper, positive | {p_var: -float(TARGET_EVENTS)}, -inf, 0.0)
        add_constraint(rows, lower, upper, negative | {n_var: -1.0}, 0.0, inf)
        add_constraint(rows, lower, upper, negative | {n_var: -float(TARGET_EVENTS)}, -inf, 0.0)
        add_constraint(rows, lower, upper, {both_var: 1.0, p_var: -1.0}, -inf, 0.0)
        add_constraint(rows, lower, upper, {both_var: 1.0, n_var: -1.0}, -inf, 0.0)
        add_constraint(
            rows,
            lower,
            upper,
            {both_var: 1.0, p_var: -1.0, n_var: -1.0},
            -1.0,
            inf,
        )
    add_constraint(
        rows,
        lower,
        upper,
        {camera_vars[camera][2]: 1.0 for camera in cameras},
        MIN_TWO_ROLE_CAMERAS,
        inf,
    )

    matrix_rows: list[int] = []
    matrix_cols: list[int] = []
    matrix_data: list[float] = []
    for row_index, coefficients in enumerate(rows):
        for column, value in coefficients.items():
            matrix_rows.append(row_index)
            matrix_cols.append(column)
            matrix_data.append(value)
    matrix = csr_matrix(
        (matrix_data, (matrix_rows, matrix_cols)), shape=(len(rows), variable_count)
    )
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[:n_events] = np.arange(1, n_events + 1, dtype=np.float64)
    result = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 300.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"B1 MILP failed: {result.message}")
    selected = [event for index, event in enumerate(ordered) if result.x[index] >= 0.5]
    if len(selected) != TARGET_EVENTS:
        raise RuntimeError(f"B1 MILP selected {len(selected)} events")
    return sorted(selected, key=lambda event: event["event_id"])


def cohort_constraints_hold(events: list[dict[str, Any]]) -> bool:
    if len(events) != TARGET_EVENTS:
        return False
    metrics = w4.incident_metrics(events)
    activity_counts = Counter(event["activity"] for event in events)
    return (
        set(activity_counts) == set(w4.ACTIVITIES)
        and min(activity_counts.values()) >= MIN_EVENTS_PER_ACTIVITY
        and max(activity_counts.values()) <= MAX_ACTIVITY_EVENTS
        and metrics["n_cand_ge_3"] >= MIN_N_CAND_GE_3
        and metrics["n_cand_ge_4"] >= MIN_N_CAND_GE_4
        and metrics["two_role_cameras"] >= MIN_TWO_ROLE_CAMERAS
        and metrics["location_date_blocks"] >= MIN_LOCATION_DATE_BLOCKS
        and max(metrics["camera_event_incidence"].values()) <= MAX_CAMERA_EVENTS
    )


def decorate_event(event: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(event))
    old_event_id = result["event_id"]
    result["selection_source_event_id"] = old_event_id
    result["event_id"] = f"MEVA-B1-{result['component_sha256'][:16].upper()}"
    for view in result["views"]:
        view["view_id"] = f"{result['event_id']}|{view['camera_id']}"
        clip = normalize_clip(view["clip"])
        s3 = view["s3"]
        view["content_pin"] = {
            "metadata_stage_identity": "clip_id + remote_key + expected_bytes + ETag",
            "clip_id": clip,
            "remote_key": s3["s3_key"],
            "expected_bytes": int(s3["expected_bytes"]),
            "etag": s3["etag"],
            "downloaded_sha256": None,
            "downloaded_sha256_status": "reserved_for_execution_time_verification",
        }
        view["site_slot_key"] = site_slot(clip)
        view.pop("view_spec_sha256", None)
        view["view_spec_sha256"] = sha256_bytes(w4.canonical_bytes(view))
    result["source_site_slot_keys"] = sorted({view["site_slot_key"] for view in result["views"]})
    positive_heights = [
        float(view["selection_height_px"])
        for view in result["views"]
        if view["role"] == "positive"
    ]
    negatives = [
        view["minimum_outside_clearance_deg_beyond_buffer"]
        for view in result["views"]
        if view["role"] == "negative_geometric"
    ]
    min_clearance = min(negatives)
    min_positive_height = min(positive_heights)
    low_height = min_positive_height <= LOW_HEIGHT_PROXY_PX
    boundary_near = float(min_clearance) <= BOUNDARY_PROXY_DEG
    result["proxy_strata"] = {
        "metadata_only_not_semantic_sufficiency": True,
        "minimum_positive_person_height_px": min_positive_height,
        "minimum_positive_height_le_64px": low_height,
        "minimum_negative_fov_clearance_deg_beyond_buffer": float(min_clearance),
        "minimum_negative_fov_clearance_le_10deg": boundary_near,
        "joint_label": (
            "both"
            if low_height and boundary_near
            else "height_only"
            if low_height
            else "fov_only"
            if boundary_near
            else "neither"
        ),
    }
    result["selection_sort_key_sha256"] = hashlib.sha256(
        f"{B1_SELECTION_SEED}|event|{old_event_id}".encode()
    ).hexdigest()
    result.pop("event_spec_sha256", None)
    result["event_spec_sha256"] = sha256_bytes(w4.canonical_bytes(result))
    if old_event_id == result["event_id"]:
        raise RuntimeError("B1 event namespace was not changed")
    return result


def reserve_payload(
    eligible: list[dict[str, Any]], selected: list[dict[str, Any]], manifest_sha: str
) -> dict[str, Any]:
    selected_ids = {event["event_id"] for event in selected}
    counts = Counter(event["activity"] for event in selected)
    candidates = [event for event in eligible if event["event_id"] not in selected_ids]
    ordered = sorted(
        candidates,
        key=lambda event: (
            0 if counts[event["activity"]] <= MIN_EVENTS_PER_ACTIVITY else 1,
            counts[event["activity"]] - MIN_EVENTS_PER_ACTIVITY,
            hashlib.sha256(
                f"{B1_RESERVE_SEED}|reserve|{event['event_id']}".encode()
            ).hexdigest(),
            event["event_id"],
        ),
    )
    entries: list[dict[str, Any]] = []
    for event in ordered:
        compatible = 0
        for failed in selected:
            trial = [row for row in selected if row["event_id"] != failed["event_id"]] + [event]
            if cohort_constraints_hold(trial):
                compatible += 1
        if compatible == 0:
            continue
        entries.append(
            {
                "reserve_rank": len(entries) + 261,
                "activity_post_single_loss_floor_risk": counts[event["activity"]]
                <= MIN_EVENTS_PER_ACTIVITY,
                "selected_activity_count": counts[event["activity"]],
                "activity_slack_above_floor": counts[event["activity"]]
                - MIN_EVENTS_PER_ACTIVITY,
                "reserve_sort_key_sha256": hashlib.sha256(
                    f"{B1_RESERVE_SEED}|reserve|{event['event_id']}".encode()
                ).hexdigest(),
                "base_cohort_compatible_failed_event_count": compatible,
                "event": event,
            }
        )
    if not entries:
        raise RuntimeError("no one-for-one constraint-preserving reserve candidates")
    return {
        "schema_version": "meva_b1_reserve_v1",
        "status": "metadata_only_result_unobserved",
        "seed": B1_RESERVE_SEED,
        "cohort_manifest_sha256": manifest_sha,
        "rank_origin": 261,
        "ordering_rule": [
            "activities at the eight-event floor first",
            "smaller selected-cohort activity slack above eight first",
            "ascending SHA-256(seed + '|reserve|' + event_id)",
            "ascending event_id",
        ],
        "replacement_rule": (
            "Remove the failed active event, then scan this static list from rank 261. "
            "Choose the first unused candidate for which cohort_constraints_hold is true "
            "on the current active 260-event cohort. Do not skip a feasible earlier rank."
        ),
        "unresolved_failure_rule": "record failure and treat it as non-release; do not alter constraints",
        "unselected_eligible_events": len(candidates),
        "constraint_preserving_base_reserve_events": len(entries),
        "entries": entries,
    }


def build(args: argparse.Namespace) -> dict[str, str]:
    pins_path = args.pins.resolve()
    pins_sha = sha256_file(pins_path)
    prior_clips, prior_paths, exclusion_audit = conservative_exclusion(pins_path)
    if len(prior_clips) != EXPECTED_PRIOR_CLIPS:
        raise RuntimeError(f"prior clip count drift: {len(prior_clips)}")
    prior_slots = {site_slot(clip) for clip in prior_clips}
    if len(prior_slots) != EXPECTED_PRIOR_SITE_SLOTS:
        raise RuntimeError(f"prior site-slot count drift: {len(prior_slots)}")

    stage_a_path = ROOT / "outputs" / "meva_pilot_20260713" / "stage_a_summary.json"
    stage_a = load_json(stage_a_path)
    if stage_a["meva_data_repo_commit"] != EXPECTED_MEVA_COMMIT:
        raise RuntimeError("MEVA metadata commit drift")

    meva_repo = args.meva_repo.resolve()
    data_root = args.data_root.resolve()
    pilot_root = args.pilot_root.resolve()
    annotation_root = meva_repo / "annotation" / "DIVA-phase-2" / "MEVA"
    models_dir = meva_repo / "metadata" / "camera-models" / "krtd"
    clip_table_path = meva_repo / "metadata" / "meva-clip-camera-and-time-table.txt"
    m1_path = pilot_root / "independent_multiview_events_5m.csv"
    inventory_path = pilot_root / "s3_object_inventory.csv"
    for path in (annotation_root, models_dir, clip_table_path, m1_path, inventory_path):
        if not path.exists():
            raise FileNotFoundError(path)

    clip_table = w4.parse_clip_table(clip_table_path)
    inventory = w4.build_inventory(w4.read_csv(inventory_path))
    clip_index, clip_rejections = w4.build_clip_index(clip_table, models_dir, inventory)
    events, aliases, activity_paths, event_rejections = w4.load_events(
        annotation_root, clip_table
    )
    representatives, alias_to_rep, rep_aliases = w4.choose_representatives(events, aliases)
    union = w4.UnionFind(representatives)
    m1_rows = [
        row for row in w4.read_csv(m1_path) if row["activity"] in set(w4.ACTIVITIES)
    ]
    m1_to_reps, rep_to_m1 = w4.attach_m1_components(union, m1_rows, alias_to_rep)

    # Clip/slot exclusion is intentionally restricted to the five pinned
    # consumption histories above.  Physical-incident exclusion follows the
    # frozen W4 identifier ledger, which also prevents reuse through another
    # synchronized camera/clip of the same incident.  B1's own output root is
    # excluded so a second deterministic run cannot ingest its first output.
    excluded_b1_output_roots = {
        args.output_root.resolve(),
        (ROOT / "outputs" / "b1_prereg_20260805").resolve(),
    }
    physical_prior_paths = [
        path
        for path in w4.overlap_metadata_paths(ROOT, data_root)
        if not any(root in path.parents or path == root for root in excluded_b1_output_roots)
    ]
    prior = w4.collect_prior_use(physical_prior_paths)
    used_reps: set[str] = set()
    for event_id in prior["event_ids"]:
        used_reps.update(m1_to_reps.get(event_id, set()))
    for raw_uid in prior["raw_uids"]:
        representative = alias_to_rep.get(raw_uid)
        if representative:
            used_reps.add(representative)

    print(
        json.dumps(
            {
                "stage": "metadata_prior_components",
                "prior_event_ids": len(prior["event_ids"]),
                "prior_raw_uids": len(prior["raw_uids"]),
                "used_representatives": len(used_reps),
                "used_physical_components": len({union.find(rep) for rep in used_reps}),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    profiles, annotation_profiles, profile_rejections, model_hashes = w4.scan_profiles(
        representatives, activity_paths, models_dir, clip_index, inventory
    )
    match_stats = w4.match_positive_incidents(union, annotation_profiles)
    components: dict[str, list[str]] = defaultdict(list)
    for representative in representatives:
        components[union.find(representative)].append(representative)
    used_roots = {union.find(representative) for representative in used_reps}

    prior_clip_set = set(prior_clips)

    def clip_is_b1_eligible(clip: str) -> bool:
        normalized = normalize_clip(clip)
        return normalized not in prior_clip_set and site_slot(normalized) not in prior_slots

    filtered_profiles: dict[str, w4.PositiveProfile] = {}
    for representative, profile in profiles.items():
        if not clip_is_b1_eligible(profile.event.clip):
            continue
        filtered_profiles[representative] = profile.__class__(
            **{
                **profile.__dict__,
                "negative_candidates": {
                    camera: candidate
                    for camera, candidate in profile.negative_candidates.items()
                    if clip_is_b1_eligible(candidate["clip"])
                },
            }
        )
    filtered_annotation_profiles = {
        representative: profile
        for representative, profile in annotation_profiles.items()
        if clip_is_b1_eligible(profile.event.clip)
    }

    eligible: list[dict[str, Any]] = []
    component_rejections: Counter[str] = Counter()
    for root in sorted(components):
        reps = sorted(components[root])
        if root in used_roots:
            component_rejections["prior_physical_component"] += 1
            continue
        configured = w4.select_component_configuration(
            reps, filtered_profiles, filtered_annotation_profiles, rep_aliases, rep_to_m1
        )
        if configured is None:
            component_rejections["no_eligible_candidate_configuration"] += 1
            continue
        eligible.append(decorate_event(configured))

    eligible.sort(key=lambda event: event["event_id"])
    print(
        json.dumps(
            {
                "stage": "metadata_eligible_pool",
                "eligible_events": len(eligible),
                "eligible_views": sum(event["n_cand"] for event in eligible),
                "component_rejections": dict(sorted(component_rejections.items())),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if len(eligible) != EXPECTED_ELIGIBLE_EVENTS:
        raise RuntimeError(f"eligible pool drift: {len(eligible)} != {EXPECTED_ELIGIBLE_EVENTS}")
    population_metrics = w4.incident_metrics(eligible)
    if population_metrics["total_candidate_views"] != EXPECTED_ELIGIBLE_VIEWS:
        raise RuntimeError(
            "eligible view pool drift: "
            f"{population_metrics['total_candidate_views']} != {EXPECTED_ELIGIBLE_VIEWS}"
        )

    first = solve_selection(eligible)
    second = solve_selection(eligible)
    first_ids = [event["event_id"] for event in first]
    if first_ids != [event["event_id"] for event in second]:
        raise RuntimeError("same-seed double MILP execution selected different event IDs")
    if not cohort_constraints_hold(first):
        raise RuntimeError("selected cohort violates frozen B1 constraints")
    selected_metrics = w4.incident_metrics(first)
    selected_clips = {normalize_clip(view["clip"]) for event in first for view in event["views"]}
    selected_slots = {site_slot(clip) for clip in selected_clips}
    if selected_clips & set(prior_clips) or selected_slots & prior_slots:
        raise RuntimeError("selected cohort overlaps the frozen clip/site-slot exclusion")

    exclusion_payload = ("\n".join(prior_clips) + "\n").encode("utf-8")
    exclusion_sha = sha256_bytes(exclusion_payload)
    source_hashes = {
        "pins.json": pins_sha,
        "w4_metadata_gate.py": sha256_file(ROOT / "scripts" / "w4_metadata_gate.py"),
        "stage_a_summary.json": sha256_file(stage_a_path),
        "independent_multiview_events_5m.csv": sha256_file(m1_path),
        "s3_object_inventory.csv": sha256_file(inventory_path),
        "meva_clip_camera_and_time_table.txt": sha256_file(clip_table_path),
    }
    for path in prior_paths:
        source_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(path)

    manifest = {
        "schema_version": "meva_b1_clip_slot_cohort_manifest_v1",
        "status": "metadata_only_result_unobserved_no_media_materialized",
        "seed": B1_SELECTION_SEED,
        "source": {
            "meva_metadata_commit": EXPECTED_MEVA_COMMIT,
            "source_hashes": dict(sorted(source_hashes.items())),
            "krtd_files_used": len(model_hashes),
            "per_view_annotation_and_krtd_hashes_embedded": True,
        },
        "exclusion": {
            "clip_list_path": "outputs/b1_prereg_20260805/exclusion_clips.txt",
            "clip_list_sha256": exclusion_sha,
            "prior_clip_count": len(prior_clips),
            "prior_site_slot_count": len(prior_slots),
            "site_slot_definition": (
                "site|YYYY-MM-DD.HH-(floor(MM/5)*5)-00 from source clip start; camera excluded"
            ),
            "set_constraints": [
                "B1_clip_ids intersect prior_clip_ids = empty set",
                "B1_site_slot_keys intersect prior_site_slot_keys = empty set",
            ],
            "same_camera_and_same_site_reuse_allowed": True,
            "union_audit": exclusion_audit,
            "prior_identifier_ledger_sha256": prior["ledger_sha256"],
            "prior_identifier_ledger_files": [
                {
                    **row,
                    "path": str(Path(row["path"]).relative_to(ROOT)).replace("\\", "/"),
                }
                for row in prior["files"]
            ],
        },
        "eligibility": {
            "definition": "verbatim scripts/w4_metadata_gate.py metadata/KPF/KRTD rules",
            "w4_metadata_gate_sha256": source_hashes["w4_metadata_gate.py"],
            "minimum_positive_height_px": w4.MIN_PERSON_HEIGHT,
            "minimum_positive_area_px2": w4.MIN_PERSON_AREA,
            "negative_fov_buffer_degrees_per_axis": w4.FOV_BUFFER_DEG,
            "result_observation_used": False,
            "pixel_or_frame_accessed": False,
        },
        "selection": {
            "solver": "SciPy HiGHS MILP",
            "objective": "ascending SHA-256(seed + '|event|' + event_id), then event_id",
            "constraints": {
                "events": TARGET_EVENTS,
                "all_12_activities_min_events_each": MIN_EVENTS_PER_ACTIVITY,
                "max_activity_events": MAX_ACTIVITY_EVENTS,
                "max_activity_share_exact": MAX_ACTIVITY_EVENTS / TARGET_EVENTS,
                "max_activity_share_display_percent": 26.92,
                "max_camera_events": MAX_CAMERA_EVENTS,
                "max_camera_share": MAX_CAMERA_EVENTS / TARGET_EVENTS,
                "n_cand_ge_3_min": MIN_N_CAND_GE_3,
                "n_cand_ge_4_min": MIN_N_CAND_GE_4,
                "two_role_cameras_min": MIN_TWO_ROLE_CAMERAS,
                "location_date_blocks_min": MIN_LOCATION_DATE_BLOCKS,
            },
            "same_seed_solved_twice": True,
            "event_id_sequence_sha256_run_1": sha256_bytes(("\n".join(first_ids) + "\n").encode()),
            "event_id_sequence_sha256_run_2": sha256_bytes(("\n".join(first_ids) + "\n").encode()),
        },
        "content_pin_rule": {
            "metadata_stage": "clip ID, remote object key, expected bytes, and ETag",
            "execution_stage": "download only manifest-consumed frame source and fill/verify downloaded_sha256",
        },
        "proxy_strata": {
            "labels_are_metadata_proxies_not_human_sufficiency": True,
            "anchor_positive_person_height_le_px": LOW_HEIGHT_PROXY_PX,
            "minimum_negative_fov_clearance_le_deg": BOUNDARY_PROXY_DEG,
        },
        "diagnostics": {
            "clip_index_rejections": dict(sorted(clip_rejections.items())),
            "event_loader_rejections": dict(sorted(event_rejections.items())),
            "positive_profile_rejections": dict(sorted(profile_rejections.items())),
            "component_rejections": dict(sorted(component_rejections.items())),
            "cross_camera_matching": match_stats,
        },
        "population_metrics": population_metrics,
        "cohort_metrics": selected_metrics,
        "events": first,
    }
    manifest_payload = canonical_bytes(manifest)
    manifest_sha = sha256_bytes(manifest_payload)
    reserve = reserve_payload(eligible, first, manifest_sha)
    reserve_payload_bytes = canonical_bytes(reserve)

    output_root = args.output_root.resolve()
    hashes = {
        "exclusion_clips.txt": atomic_write(output_root / "exclusion_clips.txt", exclusion_payload),
        "b1_cohort_manifest.json": atomic_write(
            output_root / "b1_cohort_manifest.json", manifest_payload
        ),
        "b1_reserve.json": atomic_write(output_root / "b1_reserve.json", reserve_payload_bytes),
    }
    if hashes["b1_cohort_manifest.json"] != manifest_sha:
        raise RuntimeError("manifest write hash drift")
    print(
        json.dumps(
            {
                "status": "PASS",
                "prior_clips": len(prior_clips),
                "prior_site_slots": len(prior_slots),
                "prior_physical_components": len(used_roots),
                "eligible_events": len(eligible),
                "eligible_views": population_metrics["total_candidate_views"],
                "selected_metrics": selected_metrics,
                "reserve_events": len(reserve["entries"]),
                "sha256": hashes,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return hashes


def parse_args() -> argparse.Namespace:
    default_meva = (
        Path(os.environ["LOCALAPPDATA"])
        / "Temp"
        / "mvevirag_meva_pilot_20260713"
        / "meva-data-repo"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pins",
        type=Path,
        default=ROOT / "outputs" / "b1_prereg_pins_20260805" / "pins.json",
    )
    parser.add_argument("--meva-repo", type=Path, default=default_meva)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "outputs" / "meva_full_20260713"
    )
    parser.add_argument(
        "--pilot-root", type=Path, default=ROOT / "outputs" / "meva_pilot_20260713"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs" / "b1_prereg_20260805"
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
