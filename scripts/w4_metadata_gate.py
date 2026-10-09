"""Build the amended W4 unused-incident metadata gate without media access.

The script consumes only MEVA KPF activity/type/geometry text, KRTD camera
models, synchronization metadata, object-inventory metadata, and prior
manifest identifiers used solely for overlap exclusion.  It writes a cohort
manifest only when every amended H1 cohort stop line is satisfied.  The
multi-positive diversity row is reported separately as a fusion-analysis
prerequisite and never changes H1 cohort selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

ACTIVITIES = (
    "person_talks_to_person",
    "person_carries_heavy_object",
    "person_texts_on_phone",
    "hand_interacts_with_person",
    "person_exits_vehicle",
    "person_enters_vehicle",
    "person_closes_vehicle_door",
    "person_opens_vehicle_door",
    "person_enters_scene_through_structure",
    "person_exits_scene_through_structure",
    "person_talks_on_phone",
    "person_opens_facility_door",
)
ACTIVITY_SUFFIX = ".activities.yml"
SOURCES = ("kitware", "kitware-meva-training")
SOURCE_PRIORITY = {"kitware-meva-training": 0, "kitware": 1}
INCOMPATIBLE_LOCAL_CAMERAS = {"G299", "G330"}
IMAGE_W = 1920
IMAGE_H = 1080
FPS = 30
MIN_PERSON_HEIGHT = 32.0
MIN_PERSON_AREA = 1024.0
FOV_BUFFER_DEG = 3.0
LOCATION_THRESHOLD_M = 5.0
POSITION_SAMPLE_STRIDE = 15
MAX_TIME_ALIGNMENT_FRAMES = 30
SELECTION_SEED = "meva-w4-unused-cohort-20260714-v1"
BOOTSTRAP_SEED = 20260714
BOOTSTRAP_REPLICATES = 4000
COHORT_TARGET_MAX = 260
COHORT_TARGET_MIN = 220

COHORT_STOP_LINES = {
    "independent_events": 220,
    "n_cand_ge_3": 200,
    "n_cand_ge_4": 70,
    "activities": 8,
    "min_events_per_activity": 8,
    "max_activity_share": 0.35,
    "two_role_cameras": 6,
    "max_camera_share": 0.40,
    "location_date_blocks": 3,
}
FUSION_AUXILIARY_PREREQUISITE = {"n_cand_ge_4_m_plus_ge_2": 50}
PRE_AMENDMENT_GATE_SHA256 = "2027a10662f0eae3025be7b57a561d8f405e9f05b9ab670a037de6ebd95239a5"
ORIGINAL_STOP_REPORT_SHA256 = "878ab797b6415c6c9eeb9a32616c1f5aa14bd767b28fc8d2ca46a6143c1f69d2"
USER_APPROVAL_KST = "2026-07-14T08:39:21+09:00"

G0_RE = re.compile(
    r"['\"]?g0['\"]?\s*:\s*['\"]?(-?\d+(?:\.\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)['\"]?"
)
ID1_RE = re.compile(r"['\"]?id1['\"]?\s*:\s*(\d+)")
TS0_RE = re.compile(r"['\"]?ts0['\"]?\s*:\s*(\d+)")
MEVA_EVENT_RE = re.compile(r"\bMEVA-MV-\d{5}\b")
RAW_UID_RE = re.compile(r"(?:kitware-meva-training|kitware)\|[^|\s,\"']+\|\d+")


@dataclass(frozen=True)
class Event:
    uid: str
    source: str
    clip: str
    camera: str
    site: str
    activity: str
    id2: int
    start: int
    end: int
    actor_ids: tuple[int, ...]
    slot: str
    offset: int
    model_file: str

    @property
    def ref_start(self) -> int:
        return self.start + self.offset

    @property
    def ref_end(self) -> int:
        return self.end + self.offset


@dataclass
class PositiveProfile:
    event: Event
    person_ids: tuple[int, ...]
    best_actor_id: int
    best_local_frame: int
    best_ref_frame: int
    best_box: tuple[float, float, float, float]
    best_height: float
    best_area: float
    sampled_positions: dict[int, tuple[float, float]]
    negative_candidates: dict[str, dict[str, Any]]
    annotation: dict[str, str]
    positive_krtd_sha256: str
    positive_s3: dict[str, Any]


@dataclass(frozen=True)
class AnnotationProfile:
    event: Event
    sampled_positions: dict[int, tuple[float, float]]


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            if root_left < root_right:
                self.parent[root_right] = root_left
            else:
                self.parent[root_left] = root_right


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")
    return sha256_bytes(payload.encode("utf-8"))


def write_sha_sidecar(path: Path, digest: str) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )


def markdown_section(path: Path, section_number: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    prefix = f"## {section_number}."
    start = next((index for index, line in enumerate(lines) if line.startswith(prefix)), None)
    if start is None:
        raise ValueError(f"missing section {section_number} in {path}")
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    return "".join(lines[start:end])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_kpf_records(path: Path, key: str) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            if not text.startswith("- "):
                raise ValueError(f"unexpected KPF line {path}:{lineno}")
            if re.match(r"^- \{\s*['\"]?meta['\"]?\s*:", text):
                continue
            try:
                record = yaml.safe_load(text)[0]
            except yaml.YAMLError as exc:
                raise ValueError(f"invalid KPF YAML {path}:{lineno}") from exc
            if "meta" in record:
                continue
            if key not in record:
                raise ValueError(f"missing {key} in {path}:{lineno}")
            yield record[key]


def camera_from_clip(clip: str) -> tuple[str, str]:
    fields = clip.split(".")
    if len(fields) < 5:
        raise ValueError(f"malformed clip: {clip}")
    return fields[-1], fields[-2]


def clip_name(path: Path, suffix: str) -> str:
    if not path.name.endswith(suffix):
        raise ValueError(path)
    return path.name[: -len(suffix)]


def parse_clip_table(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 7:
                raise ValueError(f"clip-table fields != 7 at {path}:{lineno}")
            clip, slot, camera_file, camera_set, ref_clip, offset, precision = fields
            result[clip] = {
                "slot": slot,
                "camera_file": camera_file,
                "camera_set": camera_set,
                "ref_clip": ref_clip,
                "offset": int(offset),
                "precision": int(precision),
            }
    return result


def parse_krtd(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    groups: list[list[list[float]]] = []
    current: list[list[float]] = []
    for row in rows:
        if not row:
            if current:
                groups.append(current)
                current = []
            continue
        current.append([float(value) for value in row.split()])
    if current:
        groups.append(current)
    if len(groups) < 3:
        raise ValueError(f"malformed KRTD: {path}")
    intrinsic = np.asarray(groups[0], dtype=np.float64)
    rotation = np.asarray(groups[1], dtype=np.float64)
    translation = np.asarray(groups[2], dtype=np.float64).reshape(3)
    distortion = np.asarray(groups[3][0] if len(groups) > 3 else [], dtype=np.float64)
    return intrinsic, rotation, translation, distortion


def ground_point(
    u: float, v: float, model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> tuple[float, float] | None:
    intrinsic, rotation, translation, distortion = model
    point = np.array([[[u, v]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(point, intrinsic, distortion).reshape(2)
    ray_camera = np.array([undistorted[0], undistorted[1], 1.0])
    ray_world = rotation.T @ ray_camera
    center = -rotation.T @ translation
    if abs(ray_world[2]) < 1e-9:
        return None
    scale = -center[2] / ray_world[2]
    if scale <= 0:
        return None
    world = center + scale * ray_world
    if not np.all(np.isfinite(world)) or np.linalg.norm(world[:2]) > 1000:
        return None
    return float(world[0]), float(world[1])


def angular_envelope(
    model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float, float, float]:
    intrinsic, _, _, distortion = model
    samples: list[tuple[float, float]] = []
    for value in np.linspace(0.0, float(IMAGE_W), 257):
        samples.extend(((value, 0.0), (value, float(IMAGE_H))))
    for value in np.linspace(0.0, float(IMAGE_H), 145):
        samples.extend(((0.0, value), (float(IMAGE_W), value)))
    points = np.asarray(samples, dtype=np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(points, intrinsic, distortion).reshape(-1, 2)
    horizontal = np.arctan2(normalized[:, 0], 1.0)
    vertical = np.arctan2(normalized[:, 1], np.sqrt(normalized[:, 0] ** 2 + 1.0))
    buffer_rad = math.radians(FOV_BUFFER_DEG)
    return (
        float(horizontal.min() - buffer_rad),
        float(horizontal.max() + buffer_rad),
        float(vertical.min() - buffer_rad),
        float(vertical.max() + buffer_rad),
    )


def outside_clearance_deg(
    world_xy: tuple[float, float],
    model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    envelope: tuple[float, float, float, float],
) -> float | None:
    _, rotation, translation, _ = model
    camera = rotation @ np.asarray([world_xy[0], world_xy[1], 0.0], dtype=np.float64) + translation
    x, y, z = map(float, camera)
    if not np.all(np.isfinite(camera)):
        return None
    if z <= 0.0:
        return 90.0
    horizontal = math.atan2(x, z)
    vertical = math.atan2(y, math.hypot(x, z))
    h_min, h_max, v_min, v_max = envelope
    clearance = max(
        h_min - horizontal,
        horizontal - h_max,
        v_min - vertical,
        vertical - v_max,
    )
    return math.degrees(clearance) if clearance > 0.0 else None


def clip_duration_frames(clip: str) -> int:
    fields = clip.split(".")
    if len(fields) < 5:
        raise ValueError(f"malformed clip name: {clip}")
    start = datetime.strptime(f"{fields[0]}.{fields[1]}", "%Y-%m-%d.%H-%M-%S")
    end = datetime.strptime(f"{fields[0]}.{fields[2]}", "%Y-%m-%d.%H-%M-%S")
    if end < start:
        end += timedelta(days=1)
    return int(round((end - start).total_seconds() * FPS))


def build_inventory(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["key"]
        name = Path(key).name
        if not name.endswith(".r13.avi"):
            continue
        clip = name[: -len(".r13.avi")]
        candidate = {
            "s3_key": key,
            "expected_bytes": int(row["bytes"]),
            "etag": row["etag"],
        }
        current = result.get(clip)
        if current is None or (not key.startswith("updates-r13/"), key) < (
            not current["s3_key"].startswith("updates-r13/"),
            current["s3_key"],
        ):
            result[clip] = candidate
    return result


def build_clip_index(
    clip_table: dict[str, dict[str, Any]], models_dir: Path, inventory: dict[str, dict[str, Any]]
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], Counter[str]]:
    by_slot_site: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    for clip, meta in sorted(clip_table.items()):
        camera, site = camera_from_clip(clip)
        if camera in INCOMPATIBLE_LOCAL_CAMERAS:
            rejected["incompatible_local_camera"] += 1
            continue
        if meta["precision"] < 0:
            rejected["unsynchronized"] += 1
            continue
        model_file = meta["camera_file"]
        if model_file == "no-camera-model":
            rejected["missing_krtd"] += 1
            continue
        if clip not in inventory:
            rejected["missing_public_video_object"] += 1
            continue
        if not (models_dir / model_file).is_file():
            raise FileNotFoundError(models_dir / model_file)
        duration_frames = clip_duration_frames(clip)
        if duration_frames <= 0:
            rejected["zero_duration_clip"] += 1
            continue
        by_slot_site[(meta["slot"], site)].append(
            {
                "clip": clip,
                "camera": camera,
                "site": site,
                "slot": meta["slot"],
                "offset": int(meta["offset"]),
                "model_file": model_file,
                "duration_frames": duration_frames,
                "s3": inventory[clip],
            }
        )
    return by_slot_site, rejected


def load_events(
    annotation_root: Path, clip_table: dict[str, dict[str, Any]]
) -> tuple[dict[str, Event], dict[tuple[Any, ...], list[str]], dict[tuple[str, str], Path], Counter[str]]:
    events: dict[str, Event] = {}
    aliases: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    activity_paths: dict[tuple[str, str], Path] = {}
    rejected: Counter[str] = Counter()
    activity_scope = set(ACTIVITIES)
    for source in SOURCES:
        for path in sorted((annotation_root / source).rglob(f"*{ACTIVITY_SUFFIX}")):
            clip = clip_name(path, ACTIVITY_SUFFIX)
            activity_paths[(source, clip)] = path
            meta = clip_table.get(clip)
            if meta is None:
                rejected["missing_clip_table"] += 1
                continue
            camera, site = camera_from_clip(clip)
            if camera in INCOMPATIBLE_LOCAL_CAMERAS:
                rejected["incompatible_local_camera"] += 1
                continue
            if meta["precision"] < 0 or meta["camera_file"] == "no-camera-model":
                rejected["unsynchronized_or_missing_krtd"] += 1
                continue
            for act in parse_kpf_records(path, "act"):
                labels = act.get("act2")
                if not isinstance(labels, dict) or len(labels) != 1:
                    raise ValueError(f"expected one act2 label in {path}")
                activity = next(iter(labels))
                if activity not in activity_scope:
                    continue
                spans = act.get("timespan", [])
                if len(spans) != 1 or "tsr0" not in spans[0]:
                    raise ValueError(f"unexpected activity timespan in {path}")
                start, end = map(int, spans[0]["tsr0"])
                actor_ids = tuple(sorted({int(actor["id1"]) for actor in act.get("actors", [])}))
                id2 = int(act["id2"])
                uid = f"{source}|{clip}|{id2}"
                event = Event(
                    uid=uid,
                    source=source,
                    clip=clip,
                    camera=camera,
                    site=site,
                    activity=activity,
                    id2=id2,
                    start=start,
                    end=end,
                    actor_ids=actor_ids,
                    slot=meta["slot"],
                    offset=int(meta["offset"]),
                    model_file=meta["camera_file"],
                )
                events[uid] = event
                aliases[(clip, activity, start, end, actor_ids)].append(uid)
    return events, aliases, activity_paths, rejected


def choose_representatives(
    events: dict[str, Event], aliases: dict[tuple[Any, ...], list[str]]
) -> tuple[dict[str, Event], dict[str, str], dict[str, list[str]]]:
    representatives: dict[str, Event] = {}
    alias_to_rep: dict[str, str] = {}
    rep_aliases: dict[str, list[str]] = {}
    for values in aliases.values():
        ordered = sorted(values, key=lambda uid: (SOURCE_PRIORITY[events[uid].source], uid))
        rep = ordered[0]
        representatives[rep] = events[rep]
        rep_aliases[rep] = ordered
        for uid in ordered:
            alias_to_rep[uid] = rep
    return representatives, alias_to_rep, rep_aliases


def overlap_metadata_paths(repo_root: Path, data_root: Path) -> list[Path]:
    paths: set[Path] = set()
    root_names = (
        "selected",
        "split",
        "manifest",
        "samples",
        "selection",
        "freeze",
        "sealed",
    )
    allowed_suffixes = {".json", ".jsonl", ".csv", ".txt"}
    skip_parts = {
        "responses",
        "media",
        "crops",
        "crops_v2",
        "crops_v4",
        "frames_m3",
        "videos_v2",
        "remote_frames_v4",
        "visual_validation_v2",
        "visual_validation_v4",
    }
    for root in (repo_root / "outputs", data_root):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                continue
            relative_parts = {part.lower() for part in path.relative_to(root).parts[:-1]}
            if relative_parts & skip_parts:
                continue
            if any(token in path.name.lower() for token in root_names):
                paths.add(path.resolve())
    return sorted(paths)


def collect_prior_use(paths: list[Path]) -> dict[str, Any]:
    event_ids: set[str] = set()
    raw_uids: set[str] = set()
    ledger_rows: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8-sig", errors="strict")
        found_ids = sorted(set(MEVA_EVENT_RE.findall(text)))
        found_uids = sorted(set(RAW_UID_RE.findall(text)))
        if not found_ids and not found_uids:
            continue
        event_ids.update(found_ids)
        raw_uids.update(found_uids)
        ledger_rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "meva_event_ids": len(found_ids),
                "raw_event_uids": len(found_uids),
            }
        )
    ledger_hash = sha256_bytes(canonical_bytes(ledger_rows))
    return {
        "event_ids": event_ids,
        "raw_uids": raw_uids,
        "files": ledger_rows,
        "ledger_sha256": ledger_hash,
    }


def attach_m1_components(
    union: UnionFind,
    m1_rows: list[dict[str, str]],
    alias_to_rep: dict[str, str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    m1_to_reps: dict[str, set[str]] = defaultdict(set)
    rep_to_m1: dict[str, set[str]] = defaultdict(set)
    for row in m1_rows:
        event_id = row["independent_event_id"]
        reps = {alias_to_rep[uid] for uid in row["source_event_uids"].split(";") if uid in alias_to_rep}
        for rep in reps:
            m1_to_reps[event_id].add(rep)
            rep_to_m1[rep].add(event_id)
        ordered = sorted(reps)
        for rep in ordered[1:]:
            union.union(ordered[0], rep)
    return m1_to_reps, rep_to_m1


def scan_profiles(
    representatives: dict[str, Event],
    activity_paths: dict[tuple[str, str], Path],
    models_dir: Path,
    clip_index: dict[tuple[str, str], list[dict[str, Any]]],
    inventory: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, PositiveProfile],
    dict[str, AnnotationProfile],
    Counter[str],
    dict[str, str],
]:
    by_source_clip: dict[tuple[str, str], list[str]] = defaultdict(list)
    for uid, event in representatives.items():
        by_source_clip[(event.source, event.clip)].append(uid)

    model_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    envelope_cache: dict[str, tuple[float, float, float, float]] = {}
    model_hashes: dict[str, str] = {}

    def model(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if name not in model_cache:
            path = models_dir / name
            model_cache[name] = parse_krtd(path)
            model_hashes[name] = sha256_file(path)
        return model_cache[name]

    def envelope(name: str) -> tuple[float, float, float, float]:
        if name not in envelope_cache:
            envelope_cache[name] = angular_envelope(model(name))
        return envelope_cache[name]

    profiles: dict[str, PositiveProfile] = {}
    annotation_profiles: dict[str, AnnotationProfile] = {}
    rejected: Counter[str] = Counter()
    keys = sorted(by_source_clip)
    for index, key in enumerate(keys, 1):
        source, clip = key
        path = activity_paths.get(key)
        if path is None:
            raise FileNotFoundError(f"missing activity path for {key}")
        types_path = path.with_name(f"{clip}.types.yml")
        geom_path = path.with_name(f"{clip}.geom.yml")
        types: dict[int, str] = {}
        for record in parse_kpf_records(types_path, "types"):
            labels = record.get("cset3", {})
            if not isinstance(labels, dict) or len(labels) != 1:
                raise ValueError(f"unexpected type record in {types_path}")
            types[int(record["id1"])] = next(iter(labels))

        local_uids = sorted(by_source_clip[key])
        person_ids: dict[str, tuple[int, ...]] = {
            uid: tuple(actor for actor in representatives[uid].actor_ids if types.get(actor) == "person")
            for uid in local_uids
        }
        actor_to_uids: dict[int, list[str]] = defaultdict(list)
        for uid, actors in person_ids.items():
            for actor in actors:
                actor_to_uids[actor].append(uid)
        boxes: dict[str, dict[tuple[int, int], dict[str, Any]]] = defaultdict(dict)
        with geom_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                g0 = G0_RE.search(line)
                id1 = ID1_RE.search(line)
                ts0 = TS0_RE.search(line)
                if not g0 or not id1 or not ts0:
                    continue
                actor_id = int(id1.group(1))
                if actor_id not in actor_to_uids:
                    continue
                frame = int(ts0.group(1))
                box = tuple(map(float, g0.group(1, 2, 3, 4)))
                x1, y1, x2, y2 = box
                if not (0.0 <= x1 < x2 <= IMAGE_W and 0.0 <= y1 < y2 <= IMAGE_H):
                    continue
                area = (x2 - x1) * (y2 - y1)
                for uid in actor_to_uids[actor_id]:
                    event = representatives[uid]
                    if not event.start <= frame <= event.end:
                        continue
                    key_box = (actor_id, frame)
                    previous = boxes[uid].get(key_box)
                    if previous is None or area > previous["area"]:
                        boxes[uid][key_box] = {
                            "box": box,
                            "height": y2 - y1,
                            "area": area,
                        }

        annotation_hashes = {
            "activities_path": str(path.resolve()),
            "activities_sha256": sha256_file(path),
            "types_path": str(types_path.resolve()),
            "types_sha256": sha256_file(types_path),
            "geom_path": str(geom_path.resolve()),
            "geom_sha256": sha256_file(geom_path),
        }
        for uid in local_uids:
            event = representatives[uid]
            actors = person_ids[uid]
            if not actors:
                rejected["no_person_actor"] += 1
                continue
            if event.clip not in inventory:
                rejected["positive_missing_public_video_object"] += 1
                continue

            # Amendment correction: annotation-level cross-camera matching uses
            # every available projected actor sample, while a complete trajectory
            # is required only when this view anchors a full-window geometric
            # negative proof.  Requiring completeness before incident matching
            # silently imposed an anchor-only rule on non-anchor positives and
            # contradicted the original code comment.  M+ remains stricter than
            # M_ann+ because the frozen minimum-size rule is applied below.
            annotation_samples: dict[int, list[tuple[float, float]]] = defaultdict(list)
            positive_model = model(event.model_file)
            for (actor_id, frame), row in sorted(boxes[uid].items()):
                if actor_id not in actors or frame % POSITION_SAMPLE_STRIDE != 0:
                    continue
                box = row["box"]
                point = ground_point((box[0] + box[2]) / 2.0, box[3], positive_model)
                if point is not None:
                    annotation_samples[frame + event.offset].append(point)
            annotation_sampled_medians = {
                frame: tuple(map(float, np.median(np.asarray(points), axis=0)))
                for frame, points in annotation_samples.items()
            }
            if annotation_sampled_medians:
                annotation_profiles[uid] = AnnotationProfile(
                    event=event,
                    sampled_positions=annotation_sampled_medians,
                )
            else:
                rejected["annotation_no_sampled_world_positions"] += 1

            best: tuple[tuple[float, int, int], int, int, dict[str, Any]] | None = None
            for (actor_id, frame), row in boxes[uid].items():
                if actor_id not in actors or not event.start <= frame <= event.end:
                    continue
                if row["height"] >= MIN_PERSON_HEIGHT and row["area"] >= MIN_PERSON_AREA:
                    score = (float(row["area"]), -frame, -actor_id)
                    if best is None or score > best[0]:
                        best = (score, actor_id, frame, row)
            if best is None:
                rejected["no_minimum_size_box"] += 1
                continue

            # Complete tracks are an anchor-only requirement for conservative
            # full-window geometric negatives, not a per-view M+ requirement.
            trajectory: list[tuple[int, int, tuple[float, float]]] = []
            failure: str | None = None
            for frame in range(event.start, event.end + 1):
                for actor_id in actors:
                    row = boxes[uid].get((actor_id, frame))
                    if row is None:
                        failure = "incomplete_positive_track"
                        break
                    box = row["box"]
                    point = ground_point((box[0] + box[2]) / 2.0, box[3], positive_model)
                    if point is None:
                        failure = "backprojection_failure"
                        break
                    ref_frame = frame + event.offset
                    trajectory.append((actor_id, ref_frame, point))
                if failure:
                    break

            negative_by_camera: dict[str, dict[str, Any]] = {}
            if failure is None:
                for target in clip_index.get((event.slot, event.site), []):
                    if target["camera"] == event.camera:
                        continue
                    local_start = event.ref_start - target["offset"]
                    local_end = event.ref_end - target["offset"]
                    if local_start < 0 or local_end >= target["duration_frames"]:
                        continue
                    clearances: list[float] = []
                    target_model = model(target["model_file"])
                    target_envelope = envelope(target["model_file"])
                    for _, _, point in trajectory:
                        clearance = outside_clearance_deg(point, target_model, target_envelope)
                        if clearance is None:
                            clearances = []
                            break
                        clearances.append(clearance)
                    if not clearances:
                        continue
                    candidate = {
                        **target,
                        "minimum_outside_clearance_deg_beyond_buffer": round(min(clearances), 6),
                        "actor_frame_proofs": len(clearances),
                        "target_krtd_sha256": model_hashes[target["model_file"]],
                    }
                    previous = negative_by_camera.get(target["camera"])
                    order = (
                        -candidate["minimum_outside_clearance_deg_beyond_buffer"],
                        candidate["clip"],
                    )
                    if previous is None or order < (
                        -previous["minimum_outside_clearance_deg_beyond_buffer"],
                        previous["clip"],
                    ):
                        negative_by_camera[target["camera"]] = candidate
                if not negative_by_camera:
                    failure = "no_geometric_negative_camera"
            if failure is not None:
                rejected[f"anchor_{failure}"] += 1
            best_row = best[3]
            profiles[uid] = PositiveProfile(
                event=event,
                person_ids=actors,
                best_actor_id=best[1],
                best_local_frame=best[2],
                best_ref_frame=best[2] + event.offset,
                best_box=best_row["box"],
                best_height=float(best_row["height"]),
                best_area=float(best_row["area"]),
                sampled_positions=annotation_sampled_medians,
                negative_candidates=negative_by_camera,
                annotation=annotation_hashes,
                positive_krtd_sha256=model_hashes[event.model_file],
                positive_s3=inventory[event.clip],
            )
        if index % 100 == 0 or index == len(keys):
            print(f"metadata_geometry {index}/{len(keys)}", flush=True)
    return profiles, annotation_profiles, rejected, model_hashes


def event_distance(left: AnnotationProfile, right: AnnotationProfile) -> float | None:
    aligned: list[float] = []
    right_times = sorted(right.sampled_positions)
    for left_time, left_point in left.sampled_positions.items():
        candidates = [time for time in right_times if abs(time - left_time) <= MAX_TIME_ALIGNMENT_FRAMES]
        if not candidates:
            continue
        right_time = min(candidates, key=lambda value: (abs(value - left_time), value))
        aligned.append(
            float(np.linalg.norm(np.asarray(left_point) - np.asarray(right.sampled_positions[right_time])))
        )
    return float(np.median(aligned)) if aligned else None


def match_positive_incidents(
    union: UnionFind, profiles: dict[str, AnnotationProfile]
) -> dict[str, int]:
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for uid, profile in profiles.items():
        if not profile.sampled_positions:
            continue
        event = profile.event
        grouped[(event.slot, event.site, event.activity)].append(uid)
    possible = 0
    matched = 0
    for key in sorted(grouped):
        by_camera: dict[str, list[str]] = defaultdict(list)
        for uid in grouped[key]:
            by_camera[profiles[uid].event.camera].append(uid)
        cameras = sorted(by_camera)
        for index, camera_left in enumerate(cameras):
            for camera_right in cameras[index + 1 :]:
                candidates: list[tuple[float, int, str, str]] = []
                for uid_left in by_camera[camera_left]:
                    left = profiles[uid_left]
                    for uid_right in by_camera[camera_right]:
                        right = profiles[uid_right]
                        if (
                            left.event.ref_end < right.event.ref_start
                            or right.event.ref_end < left.event.ref_start
                        ):
                            continue
                        distance = event_distance(left, right)
                        if distance is None or distance > LOCATION_THRESHOLD_M:
                            continue
                        overlap = (
                            min(left.event.ref_end, right.event.ref_end)
                            - max(left.event.ref_start, right.event.ref_start)
                            + 1
                        )
                        candidates.append((distance, -overlap, uid_left, uid_right))
                possible += len(candidates)
                used_left: set[str] = set()
                used_right: set[str] = set()
                for _, _, uid_left, uid_right in sorted(candidates):
                    if uid_left in used_left or uid_right in used_right:
                        continue
                    used_left.add(uid_left)
                    used_right.add(uid_right)
                    union.union(uid_left, uid_right)
                    matched += 1
    return {"possible_pair_links": possible, "greedy_one_to_one_links": matched}


def select_component_configuration(
    reps: list[str],
    profiles: dict[str, PositiveProfile],
    annotation_profiles: dict[str, AnnotationProfile],
    rep_aliases: dict[str, list[str]],
    rep_to_m1: dict[str, set[str]],
) -> dict[str, Any] | None:
    valid = [profiles[rep] for rep in reps if rep in profiles]
    if not valid:
        return None
    activities = {profile.event.activity for profile in valid}
    sites = {profile.event.site for profile in valid}
    if len(activities) != 1 or len(sites) != 1:
        return None
    annotated_cameras = {
        annotation_profiles[rep].event.camera for rep in reps if rep in annotation_profiles
    }
    positive_by_camera: dict[str, PositiveProfile] = {}
    for profile in valid:
        camera = profile.event.camera
        current = positive_by_camera.get(camera)
        score = (-profile.best_area, -profile.best_height, profile.event.uid)
        if current is None or score < (-current.best_area, -current.best_height, current.event.uid):
            positive_by_camera[camera] = profile
    m_plus = len(positive_by_camera)
    m_ann_plus = len(annotated_cameras)

    configs: list[tuple[tuple[Any, ...], PositiveProfile, list[dict[str, Any]]]] = []
    for anchor in valid:
        if not anchor.negative_candidates:
            continue
        negatives = [
            value for camera, value in anchor.negative_candidates.items() if camera not in annotated_cameras
        ]
        negatives.sort(
            key=lambda row: (
                -row["minimum_outside_clearance_deg_beyond_buffer"],
                row["camera"],
                row["clip"],
            )
        )
        if not negatives:
            continue
        n_cand = m_plus + len(negatives)
        tie = sha256_text(f"{SELECTION_SEED}|anchor|{anchor.event.uid}")
        configs.append(
            (
                (
                    -n_cand,
                    -int(m_plus >= 2),
                    -m_plus,
                    -len(negatives),
                    -min(row["minimum_outside_clearance_deg_beyond_buffer"] for row in negatives),
                    tie,
                    anchor.event.uid,
                ),
                anchor,
                negatives,
            )
        )
    if not configs:
        return None
    _, anchor, negatives = min(configs, key=lambda item: item[0])

    component_payload = [
        {
            "clip": profiles[rep].event.clip if rep in profiles else rep.split("|", 2)[1],
            "activity": profiles[rep].event.activity if rep in profiles else "unknown",
            "aliases": rep_aliases[rep],
        }
        for rep in sorted(reps)
    ]
    component_sha = sha256_bytes(canonical_bytes(component_payload))
    event_id = f"MEVA-W4-{component_sha[:16].upper()}"
    date = anchor.event.clip.split(".")[0]
    block = f"{anchor.event.site}|{date}"
    views: list[dict[str, Any]] = []

    for camera, profile in sorted(positive_by_camera.items()):
        view = {
            "view_id": f"{event_id}|{camera}",
            "camera_id": camera,
            "role": "positive",
            "clip": profile.event.clip,
            "native_frame": profile.best_local_frame,
            "reference_frame": profile.best_ref_frame,
            "source_event_uid": profile.event.uid,
            "selection_actor_id": profile.best_actor_id,
            "selection_bbox_xyxy": [round(value, 3) for value in profile.best_box],
            "selection_height_px": round(profile.best_height, 3),
            "selection_area_px2": round(profile.best_area, 3),
            "model_file": profile.event.model_file,
            "krtd_sha256": profile.positive_krtd_sha256,
            "annotation": profile.annotation,
            "s3": profile.positive_s3,
            "frame_spec": {
                "frames_per_view": 1,
                "source_resolution": "1920x1080; 1920x1072 allowed only for the documented MEVA source exception",
                "orientation": "canonical MEVA KRTD orientation before extraction",
                "answerer_input": "one full frame, aspect-preserving max-side 1024, plus the frozen native square target crop",
            },
        }
        view["view_spec_sha256"] = sha256_bytes(canonical_bytes(view))
        views.append(view)

    for target in sorted(negatives, key=lambda row: row["camera"]):
        native_frame = anchor.best_ref_frame - int(target["offset"])
        if native_frame < 0 or native_frame >= int(target["duration_frames"]):
            raise RuntimeError("negative frame escaped frozen full-window coverage")
        view = {
            "view_id": f"{event_id}|{target['camera']}",
            "camera_id": target["camera"],
            "role": "negative_geometric",
            "clip": target["clip"],
            "native_frame": native_frame,
            "reference_frame": anchor.best_ref_frame,
            "proof_source_event_uid": anchor.event.uid,
            "proof_actor_frame_count": target["actor_frame_proofs"],
            "minimum_outside_clearance_deg_beyond_buffer": target[
                "minimum_outside_clearance_deg_beyond_buffer"
            ],
            "fov_buffer_degrees_per_axis": FOV_BUFFER_DEG,
            "model_file": target["model_file"],
            "krtd_sha256": target["target_krtd_sha256"],
            "anchor_annotation": anchor.annotation,
            "anchor_krtd_sha256": anchor.positive_krtd_sha256,
            "s3": target["s3"],
            "frame_spec": {
                "frames_per_view": 1,
                "source_resolution": "1920x1080; 1920x1072 allowed only for the documented MEVA source exception",
                "orientation": "canonical MEVA KRTD orientation before extraction",
                "answerer_input": "one full frame, aspect-preserving max-side 1024, plus the frozen native square target crop",
            },
        }
        view["view_spec_sha256"] = sha256_bytes(canonical_bytes(view))
        views.append(view)

    m_minus = sum(view["role"] == "negative_geometric" for view in views)
    if len(views) != m_plus + m_minus:
        raise RuntimeError("candidate-view accounting mismatch")
    source_m1 = sorted({event_id for rep in reps for event_id in rep_to_m1.get(rep, set())})
    event = {
        "event_id": event_id,
        "activity": anchor.event.activity,
        "site": anchor.event.site,
        "date": date,
        "location_date_block": block,
        "slot": anchor.event.slot,
        "component_sha256": component_sha,
        "source_m1_metadata_components": source_m1,
        "representative_source_event_uids": sorted(reps),
        "anchor_positive_event_uid": anchor.event.uid,
        "n_cand": len(views),
        "m_plus": m_plus,
        "m_ann_plus": m_ann_plus,
        "m_minus": m_minus,
        "strata": {
            "activity": anchor.event.activity,
            "site": anchor.event.site,
            "location_date_block": block,
            "n_cand_band": "4_plus" if len(views) >= 4 else "3" if len(views) == 3 else "2",
            "multi_positive": m_plus >= 2,
        },
        "views": sorted(views, key=lambda row: row["camera_id"]),
    }
    event["selection_sort_key_sha256"] = sha256_text(
        f"{SELECTION_SEED}|event|{event_id}"
    )
    event["event_spec_sha256"] = sha256_bytes(canonical_bytes(event))
    return event


def incident_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(events)
    activity_counts = Counter(event["activity"] for event in events)
    camera_incidence = Counter()
    positive_cameras: set[str] = set()
    negative_cameras: set[str] = set()
    for event in events:
        cameras = {view["camera_id"] for view in event["views"]}
        camera_incidence.update(cameras)
        positive_cameras.update(view["camera_id"] for view in event["views"] if view["role"] == "positive")
        negative_cameras.update(
            view["camera_id"] for view in event["views"] if view["role"] == "negative_geometric"
        )
    blocks = {event["location_date_block"] for event in events}
    max_activity = max(activity_counts.values(), default=0)
    max_camera = max(camera_incidence.values(), default=0)
    return {
        "independent_events": total,
        "n_cand_ge_3": sum(event["n_cand"] >= 3 for event in events),
        "n_cand_ge_4": sum(event["n_cand"] >= 4 for event in events),
        "n_cand_ge_4_m_plus_ge_2": sum(event["n_cand"] >= 4 and event["m_plus"] >= 2 for event in events),
        "activities": len(activity_counts),
        "min_events_per_activity": min(activity_counts.values(), default=0),
        "max_activity_share": (max_activity / total) if total else 0.0,
        "two_role_cameras": len(positive_cameras & negative_cameras),
        "max_camera_share": (max_camera / total) if total else 0.0,
        "location_date_blocks": len(blocks),
        "total_candidate_views": sum(event["n_cand"] for event in events),
        "activity_counts": dict(sorted(activity_counts.items())),
        "camera_event_incidence": dict(sorted(camera_incidence.items())),
        "two_role_camera_ids": sorted(positive_cameras & negative_cameras),
        "location_date_block_counts": dict(
            sorted(Counter(event["location_date_block"] for event in events).items())
        ),
        "n_cand_distribution": dict(sorted(Counter(event["n_cand"] for event in events).items())),
        "m_plus_distribution": dict(sorted(Counter(event["m_plus"] for event in events).items())),
        "m_ann_plus_distribution": dict(
            sorted(Counter(event["m_ann_plus"] for event in events).items())
        ),
    }


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


def solve_selection(events: list[dict[str, Any]], target: int) -> list[dict[str, Any]] | None:
    ordered = sorted(
        events,
        key=lambda event: (
            sha256_text(f"{SELECTION_SEED}|event|{event['event_id']}"),
            event["event_id"],
        ),
    )
    n_events = len(ordered)
    activities = sorted(
        activity
        for activity, count in Counter(event["activity"] for event in ordered).items()
        if count >= COHORT_STOP_LINES["min_events_per_activity"]
    )
    blocks = sorted({event["location_date_block"] for event in ordered})
    cameras = sorted({view["camera_id"] for event in ordered for view in event["views"]})

    activity_var = {activity: n_events + index for index, activity in enumerate(activities)}
    block_offset = n_events + len(activities)
    block_var = {block: block_offset + index for index, block in enumerate(blocks)}
    camera_offset = block_offset + len(blocks)
    camera_vars = {
        camera: (camera_offset + 3 * index, camera_offset + 3 * index + 1, camera_offset + 3 * index + 2)
        for index, camera in enumerate(cameras)
    }
    variable_count = camera_offset + 3 * len(cameras)

    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []
    inf = np.inf
    add_constraint(rows, lower, upper, {index: 1.0 for index in range(n_events)}, target, target)
    add_constraint(
        rows,
        lower,
        upper,
        {index: 1.0 for index, event in enumerate(ordered) if event["n_cand"] >= 3},
        COHORT_STOP_LINES["n_cand_ge_3"],
        inf,
    )
    add_constraint(
        rows,
        lower,
        upper,
        {index: 1.0 for index, event in enumerate(ordered) if event["n_cand"] >= 4},
        COHORT_STOP_LINES["n_cand_ge_4"],
        inf,
    )

    activity_cap = math.floor(COHORT_STOP_LINES["max_activity_share"] * target + 1e-12)
    for activity in activities:
        indices = [index for index, event in enumerate(ordered) if event["activity"] == activity]
        coefficients = {index: 1.0 for index in indices}
        y = activity_var[activity]
        add_constraint(
            rows,
            lower,
            upper,
            coefficients | {y: -float(COHORT_STOP_LINES["min_events_per_activity"])},
            0.0,
            inf,
        )
        add_constraint(rows, lower, upper, coefficients | {y: -float(target)}, -inf, 0.0)
        add_constraint(rows, lower, upper, coefficients, -inf, activity_cap)
    add_constraint(
        rows,
        lower,
        upper,
        {activity_var[activity]: 1.0 for activity in activities},
        COHORT_STOP_LINES["activities"],
        inf,
    )

    for block in blocks:
        indices = [index for index, event in enumerate(ordered) if event["location_date_block"] == block]
        coefficients = {index: 1.0 for index in indices}
        y = block_var[block]
        add_constraint(rows, lower, upper, coefficients | {y: -1.0}, 0.0, inf)
        add_constraint(rows, lower, upper, coefficients | {y: -float(target)}, -inf, 0.0)
    add_constraint(
        rows,
        lower,
        upper,
        {block_var[block]: 1.0 for block in blocks},
        COHORT_STOP_LINES["location_date_blocks"],
        inf,
    )

    camera_cap = math.floor(COHORT_STOP_LINES["max_camera_share"] * target + 1e-12)
    for camera in cameras:
        incidence = {
            index: 1.0
            for index, event in enumerate(ordered)
            if camera in {view["camera_id"] for view in event["views"]}
        }
        add_constraint(rows, lower, upper, incidence, -inf, camera_cap)
        positive = {
            index: 1.0
            for index, event in enumerate(ordered)
            if any(view["camera_id"] == camera and view["role"] == "positive" for view in event["views"])
        }
        negative = {
            index: 1.0
            for index, event in enumerate(ordered)
            if any(
                view["camera_id"] == camera and view["role"] == "negative_geometric"
                for view in event["views"]
            )
        }
        p_var, n_var, both_var = camera_vars[camera]
        add_constraint(rows, lower, upper, positive | {p_var: -1.0}, 0.0, inf)
        add_constraint(rows, lower, upper, positive | {p_var: -float(target)}, -inf, 0.0)
        add_constraint(rows, lower, upper, negative | {n_var: -1.0}, 0.0, inf)
        add_constraint(rows, lower, upper, negative | {n_var: -float(target)}, -inf, 0.0)
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
        COHORT_STOP_LINES["two_role_cameras"],
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
    matrix = csr_matrix((matrix_data, (matrix_rows, matrix_cols)), shape=(len(rows), variable_count))
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
        return None
    selected = [event for index, event in enumerate(ordered) if result.x[index] >= 0.5]
    if len(selected) != target:
        raise RuntimeError(f"MILP selected {len(selected)} events, expected {target}")
    return sorted(selected, key=lambda event: event["event_id"])


def deterministic_selection(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    upper = min(COHORT_TARGET_MAX, len(events))
    for target in range(upper, COHORT_TARGET_MIN - 1, -1):
        first = solve_selection(events, target)
        attempts.append({"target": target, "feasible": first is not None})
        if first is None:
            continue
        second = solve_selection(events, target)
        if second is None or [row["event_id"] for row in first] != [row["event_id"] for row in second]:
            raise RuntimeError("deterministic cohort regeneration mismatch")
        return first, {"attempts": attempts, "regeneration_identical": True}
    return None, {"attempts": attempts, "regeneration_identical": False}


def stop_line_table(metrics: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    directions = {
        "independent_events": "min",
        "n_cand_ge_3": "min",
        "n_cand_ge_4": "min",
        "activities": "min",
        "min_events_per_activity": "min",
        "max_activity_share": "max",
        "two_role_cameras": "min",
        "max_camera_share": "max",
        "location_date_blocks": "min",
    }
    rows = []
    for name, threshold in COHORT_STOP_LINES.items():
        measured = metrics.get(name, 0)
        passed = measured >= threshold if directions[name] == "min" else measured <= threshold + 1e-12
        deficit = (
            max(0, threshold - measured) if directions[name] == "min" else max(0.0, measured - threshold)
        )
        rows.append(
            {
                "item": name,
                "direction": directions[name],
                "threshold": threshold,
                "measured": measured,
                "passed": bool(passed),
                "deficit_or_excess": deficit,
            }
        )
    return rows, all(row["passed"] for row in rows)


def fusion_prerequisite_row(metrics: dict[str, Any]) -> dict[str, Any]:
    name, threshold = next(iter(FUSION_AUXILIARY_PREREQUISITE.items()))
    measured = metrics.get(name, 0)
    passed = measured >= threshold
    return {
        "item": name,
        "classification": "fusion_auxiliary_analysis_prerequisite",
        "direction": "min",
        "threshold": threshold,
        "measured": measured,
        "passed": passed,
        "effect_if_unmet": "delete fusion, complementarity, availability-plus-diversity, and multi-positive combination claims; H1 cohort selection continues",
    }


def make_manifest(
    selected: list[dict[str, Any]],
    source: dict[str, Any],
    exclusion: dict[str, Any],
    population_metrics: dict[str, Any],
    selection_audit: dict[str, Any],
) -> dict[str, Any]:
    selected_metrics = incident_metrics(selected)
    return {
        "schema_version": "meva_w4_unused_cohort_manifest_v1",
        "status": "metadata_only_seal_candidate_no_media_materialized",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seed": SELECTION_SEED,
        "source": source,
        "exclusion": exclusion,
        "eligibility": {
            "activity_scope": list(ACTIVITIES),
            "positive_view": {
                "activity_annotation_required": True,
                "person_actor_required": True,
                "complete_actor_track_required": False,
                "minimum_height_px": MIN_PERSON_HEIGHT,
                "minimum_area_px2": MIN_PERSON_AREA,
                "minimum_applies_to_at_least_one_actor_box_in_the_activity_window": True,
                "M_plus_definition": "distinct cameras satisfying annotation, matched-incident, materializable-input, person-actor, and frozen minimum-size rules",
            },
            "annotation_positive_view_descriptive_only": {
                "M_ann_plus_definition": "distinct matched annotated cameras before the frozen minimum-size rule",
                "used_for_cohort_or_success": False,
            },
            "negative_view_geometric_proof": {
                "complete_anchor_actor_track_required": True,
                "same_site": True,
                "synchronized_clip_covers_entire_anchor_activity_window": True,
                "ground_plane_z_m": 0.0,
                "actor_ground_point": "positive person box bottom center",
                "temporal_rule": "every integer activity frame inclusively",
                "actor_rule": "every annotated person actor must be outside at every activity frame",
                "fov_boundary_buffer_degrees_per_axis": FOV_BUFFER_DEG,
                "outside_rule": "behind camera or strictly outside the expanded angular envelope",
                "annotation_absence_is_negative_evidence": False,
            },
            "independent_incident_rule": {
                "exact_alias_deduplication": "clip, activity, frame interval, actor IDs",
                "cross_camera_match": "same slot, site, activity, overlapping time, median ground distance <= 5m",
                "pairing": "deterministic per-camera-pair one-to-one greedy by distance, overlap, and UID",
                "prior_m1_components": "retained as union constraints",
            },
        },
        "frame_spec": {
            "frames_per_view": 1,
            "positive_frame": "largest qualifying person box; ties earliest local frame then actor ID",
            "negative_frame": "anchor positive selection reference frame mapped by the synchronization offset",
            "source_resolution": "1920x1080; 1920x1072 allowed only for the documented MEVA source exception",
            "orientation": "canonical MEVA KRTD orientation",
            "answerer_preprocessing": "aspect-preserving full frame max-side 1024 plus frozen native square target crop",
        },
        "stratification": {
            "target_size_search": "largest feasible integer from 260 down to the frozen minimum 220",
            "cohort_stop_constraints": COHORT_STOP_LINES,
            "fusion_auxiliary_prerequisite": FUSION_AUXILIARY_PREREQUISITE,
            "event_order": "ascending SHA-256(seed + '|event|' + event_id), then event_id",
            "manifest_serialization_order": "events by event_id; views by camera_id",
            "solver": "SciPy HiGHS MILP; exact target and all stop-line constraints",
            "regeneration_identical": selection_audit["regeneration_identical"],
            "attempts": selection_audit["attempts"],
        },
        "population_metrics": population_metrics,
        "cohort_metrics": selected_metrics,
        "fusion_auxiliary_analysis": {
            "prerequisite": fusion_prerequisite_row(selected_metrics),
            "claims_deleted_if_unmet": [
                "fusion",
                "complementarity",
                "availability_plus_diversity_as_fusion",
                "multi_positive_combination_benefit",
            ],
        },
        "events": selected,
    }


def make_amended_registration(
    repo_root: Path,
    data_root: Path,
    manifest_path: Path,
    manifest_sha: str,
    gate_report_path: Path,
    source_hashes: dict[str, str],
    cohort_metrics: dict[str, Any],
) -> dict[str, Any]:
    selection_path = data_root / "w5a_fair_baselines" / "w5a_dev_selection.json"
    master_plan_path = data_root / "master_plan_final.md"
    stop_report_v1_path = manifest_path.parent / "w4_metadata_gate_report_v1_stop.json"
    definition_audit_path = manifest_path.parent / "w4_definition_audit_codex_86.json"
    consultation_path = data_root / "w4_branch_consultation.md"
    consultation_record_path = manifest_path.parent / "w4_branch_consultation_record.json"
    comparator_runner_path = repo_root / "scripts" / "w4_dev_comparator_runner.py"
    evidence_paths = (
        stop_report_v1_path,
        definition_audit_path,
        consultation_path,
        consultation_record_path,
        comparator_runner_path,
    )
    missing = [str(path) for path in evidence_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing amendment evidence: {missing}")
    if sha256_file(stop_report_v1_path) != ORIGINAL_STOP_REPORT_SHA256:
        raise RuntimeError("original W4 stop report hash drift")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    answer_rule = selection["selected_dev_rule_per_arm"]["answer_abstain_sft"]
    if selection["primary_answerer"] != "answer_abstain_sft":
        raise RuntimeError("W5a primary answerer drift")
    if selection["strongest_same_qwen_deployable_control"] != "explicit_abstention_prompt":
        raise RuntimeError("W5a strongest control drift")
    if answer_rule["rule"] != "dev_max_yield_role_guardrails" or not math.isclose(
        float(answer_rule["threshold"]), 0.981128697326919, rel_tol=0.0, abs_tol=1e-15
    ):
        raise RuntimeError("W5a primary release rule drift")
    script_path = Path(__file__).resolve()
    section_6 = markdown_section(master_plan_path, 6)
    gate_post_sha = sha256_file(script_path)
    consultation_sha = sha256_file(consultation_path)
    return {
        "amendment_designation": "완전한 사전등록이 아니라 메타데이터 게이트 후·주 결과 전 수정 (post-gate, pre-outcome amendment)",
        "schema_version": "meva_w4_post_gate_pre_outcome_amendment_v2",
        "status": "sealed_amendment_pending_dev_comparator_selection_before_any_new_cohort_media",
        "approved_at_kst": USER_APPROVAL_KST,
        "authority": {
            "master_plan_path": str(master_plan_path.relative_to(repo_root)).replace("\\", "/"),
            "master_plan_sha256": source_hashes["master_plan_final.md"],
            "sections": ["6", "11 stage 2", "12"],
            "original_section_6_exact_text_sha256": sha256_text(section_6),
            "original_section_6_boundary": "from heading '## 6.' through the line before heading '## 7.'",
            "original_conflict_disclosure": "section 6 line 253 made every row a hard stop, while lines 280-282 kept H1 valid when M+ yield was insufficient; this amendment resolves the conflict after observing the metadata stop",
        },
        "sealed_evidence": {
            "original_stop_report_v1": {
                "path": str(stop_report_v1_path.relative_to(repo_root)).replace("\\", "/"),
                "sha256": ORIGINAL_STOP_REPORT_SHA256,
                "status": "STOP_NO_MANIFEST",
            },
            "definition_audit_codex_86": {
                "path": str(definition_audit_path.relative_to(repo_root)).replace("\\", "/"),
                "sha256": sha256_file(definition_audit_path),
            },
            "gate_code_change": {
                "path": str(script_path.relative_to(repo_root)).replace("\\", "/"),
                "sha256_before": PRE_AMENDMENT_GATE_SHA256,
                "sha256_after": gate_post_sha,
                "diff_summary": [
                    "complete actor trajectories are anchor-only for full-window geometric-negative proof; non-anchor positive matching uses available projected annotation samples",
                    "M+ retains the frozen minimum-size rule and M_ann+ is emitted as size-free descriptive metadata only",
                    "N>=4 and M+>=2 count >=50 moved from nine H1 cohort stop constraints to a fusion auxiliary-analysis prerequisite",
                ],
            },
            "Pro_consultation": {
                "path": str(consultation_path.relative_to(repo_root)).replace("\\", "/"),
                "sha256": consultation_sha,
                "record_path": str(consultation_record_path.relative_to(repo_root)).replace(
                    "\\", "/"
                ),
                "record_sha256": sha256_file(consultation_record_path),
            },
            "user_approval_at_kst": USER_APPROVAL_KST,
        },
        "result_nonobservation_declaration": {
            "scope": "new W4 cohort through this amendment seal",
            "new_cohort_video_files_materialized": 0,
            "new_cohort_pixels_created_or_viewed": 0,
            "new_cohort_model_inferences": 0,
            "new_cohort_model_outputs_created_or_read": 0,
            "gpu_jobs_for_amendment_and_gate": 0,
            "sealed_test_and_reserve_use": "identifier overlap exclusion only; no content, pixels, sampling, or inference",
            "manuscript_files_opened_or_modified": 0,
            "qualification": "metadata aggregates, the v1 stop reason, the codex_86 definition audit, and deterministic feasibility facts were observed and are fully disclosed; this is not a claim of complete nonobservation",
        },
        "cohort": {
            "manifest_path": str(manifest_path.relative_to(repo_root)).replace("\\", "/"),
            "manifest_sha256": manifest_sha,
            "gate_report_path": str(gate_report_path.relative_to(repo_root)).replace("\\", "/"),
            "events": cohort_metrics["independent_events"],
            "candidate_views": cohort_metrics["total_candidate_views"],
            "eligibility_and_frame_spec": "frozen verbatim in the cohort manifest",
            "first_post_correction_deterministic_run_accepted_without_reoptimization": True,
            "camera_distribution_disclosure": {
                "v1_raw_eligible_population_max_camera_share": 0.5547723935389134,
                "v1_raw_eligible_population_percent": 55.47723935389134,
                "selected_cohort_cap": 0.4,
                "selected_cohort_observed": cohort_metrics["max_camera_share"],
                "constraint_scope": "the 40% constraint applies to the selected cohort; the v1 raw-population imbalance is retained as a disclosed diagnostic",
            },
        },
        "answer_and_release": {
            "primary_answerer": selection["primary_answerer"],
            "substitution_allowed": selection["primary_answerer_substitution_allowed"],
            "release_rule": answer_rule["rule"],
            "release_threshold": answer_rule["threshold"],
            "dev_pplus_correct_released_yield": answer_rule["pplus_correct_released_yield"],
            "strongest_same_qwen_control": selection["strongest_same_qwen_deployable_control"],
        },
        "routing": {
            "focal_policy": "availability",
            "deployable_policy_order": [
                "deterministic_random_exact_expectation",
                "person_count_area",
                "siglip_semantic_relevance",
                "task_matched_selector",
                "availability",
                "availability_plus_diversity",
            ],
            "policy_definitions": {
                "deterministic_random_exact_expectation": "enumerate all size-k subsets for N<=6 and average the event outcome exactly; no seed retries",
                "person_count_area": "rank by deployable detector person count, then union person-box area, then camera ID",
                "siglip_semantic_relevance": "rank frozen SigLIP2 image-question similarity, then camera ID",
                "task_matched_selector": "train-only selector for per-view rule-supported correct release; apply once to dev and the new cohort",
                "availability": "frozen pre-generation SigLIP2+DINO rule-availability score, then camera ID",
                "availability_plus_diversity": "availability-first greedy selection with frozen embedding redundancy penalty; secondary for k>=2",
            },
            "diagnostic_only": [
                "geometry_oracle",
                "true_rule_availability",
                "outcome_aware_selector",
            ],
            "diagnostic_policies_are_never_deployable_comparators": True,
        },
        "dev_comparator_selection": {
            "status": "pending_before_new_cohort_materialization",
            "focal_policy": "availability",
            "eligible_comparator_policies": [
                "deterministic_random_exact_expectation",
                "person_count_area",
                "siglip_semantic_relevance",
                "task_matched_selector",
            ],
            "selection_metric": "dev U(1) with the frozen primary answerer and release rule",
            "selection_rule": "highest dev U(1); ties lower V(1), lower measured GPU-seconds, then eligible comparator order",
            "all_deployable_policies_reported": [
                "deterministic_random_exact_expectation",
                "person_count_area",
                "siglip_semantic_relevance",
                "task_matched_selector",
                "availability",
                "availability_plus_diversity",
            ],
            "focal_and_k1_equivalent_exclusion": "availability is the H1 focal policy and availability_plus_diversity has no redundancy term at k=1; both are reported on dev but are not alternative primary comparator candidates",
            "new_cohort_information_allowed": False,
            "runner": {
                "path": str(comparator_runner_path.relative_to(repo_root)).replace("\\", "/"),
                "sha256": sha256_file(comparator_runner_path),
                "dry_run_must_not_import_or_load_models": True,
            },
            "required_inputs": {
                "train_events": 203,
                "train_candidate_views": 592,
                "dev_events": 82,
                "dev_candidate_views": 282,
                "geometry_candidates": "outputs/meva_full_20260713/v2_geometry_candidates.csv",
                "train_dev_assignment": "outputs/meva_full_20260713/v4_selected_pairs.csv",
                "primary_answerer": "answer_abstain_sft frozen checkpoint",
                "feature_models": [
                    "google/siglip2-base-patch16-384",
                    "IDEA-Research/grounding-dino-tiny with frozen thresholds",
                ],
            },
            "execution_plan": [
                "materialize only the 592 train and 282 dev candidate-view frames/crops; do not touch test, reserve, or the new cohort",
                "extract detector and SigLIP2 features once and cache by view SHA",
                "run the frozen answer_abstain_sft answerer once on the 874 candidate views with checkpoint resume only",
                "fit the task-matched selector on train only; compute every deployable policy on dev",
                "compute exact-random and observed policy U(1), V(1), and GPU-seconds; write one selection artifact and hash it",
                "replace pending comparator with the selected policy and hash-seal the final registration before cohort materialization",
            ],
            "estimated_gpu_hours": {
                "qwen_candidate_view_inference_874": [0.10, 0.18],
                "siglip_dino_feature_extraction_874": [0.12, 0.25],
                "selector_fit_and_policy_replay": [0.0, 0.02],
                "total": [0.22, 0.45],
                "basis": "observed W5a 246-row inference took 79-105 seconds; range includes container startup, feature extraction, and checkpoint writes",
            },
        },
        "hypotheses": {
            "order": ["H1", "H2", "ordered_secondary_analyses"],
            "H1": {
                "estimand": "availability focal policy minus the dev-prefrozen strongest deployable comparator",
                "primary_metric": "Delta U(1)",
                "effect_gate": ">= +0.10 absolute",
                "paired_ci": "4000-independent-event bootstrap two-sided percentile 95% CI lower bound > 0",
                "safety": "Delta V(1) one-sided percentile 95% upper bound < +0.05",
                "cost": "preprocessing-inclusive total GPU-seconds below all-camera invocation",
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            },
            "H2": selection["h2_new_cohort"],
            "strong_tmm_gate": "H1 and H2 must both pass",
            "frozen_value_changes_in_this_amendment": 0,
        },
        "cost": {
            "primary_unit": "GPU-seconds",
            "policy_formula": "router/preprocessing GPU-seconds over all candidate views + sum of measured VLM GPU-seconds for selected view invocations",
            "all_camera_formula": "sum of measured VLM GPU-seconds for every candidate view plus only preprocessing required by that all-camera path",
            "measurement": "CUDA-event or equivalent device-active elapsed seconds per stage; never estimate from policy count alone",
            "secondary": ["fixed-scheduler wall-clock", "CPU preprocessing time", "materialization bytes"],
        },
        "claim_boundary": {
            "abandoned": [
                "multi-view fusion gain",
                "cross-view complementarity",
                "availability-plus-diversity benefit as a fusion claim",
                "multi-positive-view combination benefit",
            ],
            "N_cand_ge_4": "prespecified secondary allocation-difficulty and sensitivity analysis only",
            "U_k_gt_1": "secondary cumulative recovery-versus-cost curve; never a fusion or combination-gain estimand",
            "headline": "query-conditioned multi-camera evidence allocation under a fixed generation-call budget",
        },
        "ordered_secondary_analyses": [
            "remaining deployable policies in frozen order",
            "U(k), V(k), and GPU-seconds for k>1 as cumulative recovery-cost curves",
            "N=3 versus N>=4 allocation-difficulty sensitivity",
            "remaining W5a arms in their frozen order",
            "geometry and true-label diagnostic ceilings",
        ],
        "stops_and_nonclaims": [
            "do not revive formal alpha=0.10 risk guarantees",
            "do not claim validated safety noninferiority, an operational safety standard, or an externally justified safety margin",
            "do not call M2-M4 model transfer",
            "without human judgment do not claim semantic answerability, human sufficiency, or deployment safety",
            "do not reuse the old mixed 27.45%/1.79% as batch coverage/risk",
            "do not call the old one-positive/one-negative routing result N-camera end-to-end utility",
            "do not require simultaneous superiority over every baseline or change the primary comparator after results",
            "fusion, complementarity, and multi-positive combination claims are deleted because the auxiliary prerequisite is unmet",
            "do not claim general model independence from one W5b model",
            "do not rerun completed outputs, recontact sealed test, or use the existing reserve",
            "if H1 or H2 fails, do not retune or retry the new cohort; hand the Red branch decision to the user",
        ],
        "seal_requirements": {
            "metadata_gate_script": {
                "path": str(script_path.relative_to(repo_root)).replace("\\", "/"),
                "sha256": sha256_file(script_path),
            },
            "files_before_any_dev_gpu": [
                str(script_path.relative_to(repo_root)).replace("\\", "/"),
                str(manifest_path.relative_to(repo_root)).replace("\\", "/"),
                str(
                    manifest_path.with_suffix(manifest_path.suffix + ".sha256").relative_to(repo_root)
                ).replace("\\", "/"),
                "outputs/meva_full_20260713/w4_prereg_20260714/w4_amended_registration.json",
                "outputs/meva_full_20260713/w4_prereg_20260714/w4_amended_registration.json.sha256",
                str(comparator_runner_path.relative_to(repo_root)).replace("\\", "/"),
            ],
            "dev_comparator_runner_sha256": sha256_file(comparator_runner_path),
            "final_pre_materialization_additions": [
                "dev comparator selection artifact and its SHA",
                "registration JSON with pending comparator replaced by the selected comparator",
            ],
            "commit_performed_by_this_script": False,
        },
    }


def git_state(repo_root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"head": head, "dirty": bool(status), "status_line_count": len(status)}


def main(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    meva_repo = args.meva_repo.resolve()
    data_root = args.data_root.resolve()
    pilot_root = args.pilot_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "w4_cohort_manifest.json"
    registration_path = output_root / "w4_amended_registration.json"
    report_path = output_root / "w4_metadata_gate_report.json"
    stop_report_v1_path = output_root / "w4_metadata_gate_report_v1_stop.json"
    if report_path.is_file():
        if sha256_file(report_path) != ORIGINAL_STOP_REPORT_SHA256:
            raise RuntimeError("refusing to overwrite a non-v1 W4 gate report")
        original_payload = report_path.read_bytes()
        if stop_report_v1_path.is_file() and sha256_file(stop_report_v1_path) != ORIGINAL_STOP_REPORT_SHA256:
            raise RuntimeError("archived v1 W4 stop report hash drift")
        if not stop_report_v1_path.exists():
            stop_report_v1_path.write_bytes(original_payload)
            write_sha_sidecar(stop_report_v1_path, ORIGINAL_STOP_REPORT_SHA256)
    for stale in (
        manifest_path,
        manifest_path.with_suffix(manifest_path.suffix + ".sha256"),
        registration_path,
        registration_path.with_suffix(registration_path.suffix + ".sha256"),
    ):
        if stale.exists():
            stale.unlink()

    annotation_root = meva_repo / "annotation" / "DIVA-phase-2" / "MEVA"
    models_dir = meva_repo / "metadata" / "camera-models" / "krtd"
    clip_table_path = meva_repo / "metadata" / "meva-clip-camera-and-time-table.txt"
    m1_path = pilot_root / "independent_multiview_events_5m.csv"
    inventory_path = pilot_root / "s3_object_inventory.csv"
    master_plan_path = data_root / "master_plan_final.md"
    w5a_selection_path = data_root / "w5a_fair_baselines" / "w5a_dev_selection.json"
    required = (
        annotation_root,
        models_dir,
        clip_table_path,
        m1_path,
        inventory_path,
        master_plan_path,
        w5a_selection_path,
        stop_report_v1_path,
        output_root / "w4_definition_audit_codex_86.json",
        output_root / "w4_branch_consultation_record.json",
        data_root / "w4_branch_consultation.md",
        repo_root / "scripts" / "w4_dev_comparator_runner.py",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required metadata inputs: {missing}")

    meva_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=meva_repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    clip_table = parse_clip_table(clip_table_path)
    inventory = build_inventory(read_csv(inventory_path))
    clip_index, clip_rejections = build_clip_index(clip_table, models_dir, inventory)
    events, aliases, activity_paths, event_rejections = load_events(annotation_root, clip_table)
    representatives, alias_to_rep, rep_aliases = choose_representatives(events, aliases)
    union = UnionFind(representatives)
    m1_rows = [row for row in read_csv(m1_path) if row["activity"] in set(ACTIVITIES)]
    m1_to_reps, rep_to_m1 = attach_m1_components(union, m1_rows, alias_to_rep)

    overlap_paths = overlap_metadata_paths(repo_root, data_root)
    prior = collect_prior_use(overlap_paths)
    used_reps: set[str] = set()
    for event_id in prior["event_ids"]:
        used_reps.update(m1_to_reps.get(event_id, set()))
    for raw_uid in prior["raw_uids"]:
        rep = alias_to_rep.get(raw_uid)
        if rep:
            used_reps.add(rep)

    profiles, annotation_profiles, profile_rejections, model_hashes = scan_profiles(
        representatives, activity_paths, models_dir, clip_index, inventory
    )
    match_stats = match_positive_incidents(union, annotation_profiles)

    components: dict[str, list[str]] = defaultdict(list)
    for rep in representatives:
        components[union.find(rep)].append(rep)
    used_roots = {union.find(rep) for rep in used_reps}
    eligible: list[dict[str, Any]] = []
    component_rejections: Counter[str] = Counter()
    for root in sorted(components):
        reps = sorted(components[root])
        if root in used_roots:
            component_rejections["prior_manifest_or_physical_overlap"] += 1
            continue
        configured = select_component_configuration(
            reps, profiles, annotation_profiles, rep_aliases, rep_to_m1
        )
        if configured is None:
            component_rejections["no_eligible_candidate_configuration"] += 1
            continue
        eligible.append(configured)

    eligible.sort(key=lambda event: event["event_id"])
    population_metrics = incident_metrics(eligible)
    selected: list[dict[str, Any]] | None = None
    selection_audit: dict[str, Any] = {"attempts": [], "regeneration_identical": False}
    if len(eligible) >= COHORT_TARGET_MIN:
        selected, selection_audit = deterministic_selection(eligible)

    source_hashes = {
        "master_plan_final.md": sha256_file(master_plan_path),
        "w5a_dev_selection.json": sha256_file(w5a_selection_path),
        "independent_multiview_events_5m.csv": sha256_file(m1_path),
        "s3_object_inventory.csv": sha256_file(inventory_path),
        "clip_table": sha256_file(clip_table_path),
    }
    source = {
        "meva_repo": str(meva_repo),
        "meva_repo_commit": meva_commit,
        "metadata_hashes": source_hashes,
        "krtd_files_used": len(model_hashes),
    }
    exclusion = {
        "definition": "appearance in a prior train/dev/test/reserve, selection, selected-pair, or sample manifest; metadata-only enumeration without manifest adoption is not model-use exposure",
        "prior_event_ids": len(prior["event_ids"]),
        "prior_raw_uids": len(prior["raw_uids"]),
        "overlap_metadata_files": len(prior["files"]),
        "overlap_ledger_sha256": prior["ledger_sha256"],
        "excluded_physical_components": component_rejections["prior_manifest_or_physical_overlap"],
        "sealed_test_and_reserve_fields_read": "identifiers only",
    }

    selected_metrics = incident_metrics(selected or [])
    gate_metrics = selected_metrics if selected is not None else population_metrics
    stop_rows, passed = stop_line_table(gate_metrics)
    fusion_row = fusion_prerequisite_row(gate_metrics)
    if selected is None:
        passed = False
    artifacts: dict[str, Any] = {}
    if passed and selected is not None:
        manifest = make_manifest(selected, source, exclusion, population_metrics, selection_audit)
        manifest_sha = write_json(manifest_path, manifest)
        write_sha_sidecar(manifest_path, manifest_sha)
        registration = make_amended_registration(
            repo_root,
            data_root,
            manifest_path,
            manifest_sha,
            report_path,
            source_hashes,
            selected_metrics,
        )
        registration_sha = write_json(registration_path, registration)
        write_sha_sidecar(registration_path, registration_sha)
        artifacts = {
            "cohort_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
            "cohort_manifest_sidecar": str(manifest_path.with_suffix(manifest_path.suffix + ".sha256")),
            "amended_registration": {
                "path": str(registration_path),
                "sha256": registration_sha,
            },
            "amended_registration_sidecar": str(
                registration_path.with_suffix(registration_path.suffix + ".sha256")
            ),
        }

    report = {
        "schema_version": "meva_w4_metadata_gate_report_v2_amended_h1_only",
        "status": "PASS_MANIFEST_CREATED" if passed else "STOP_NO_MANIFEST",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "invariants": {
            "gpu_count": 0,
            "video_files_opened": 0,
            "pixels_viewed": 0,
            "model_outputs_created_or_read": 0,
            "manuscript_files_opened_or_modified": 0,
            "sealed_test_and_reserve_use": "identifier overlap exclusion only",
        },
        "source": source,
        "git": git_state(repo_root),
        "raw_population": {
            "top12_annotation_records_after_sync_krtd_filter": len(events),
            "exact_alias_components": len(representatives),
            "m1_top12_metadata_components": len(m1_rows),
            "physical_components_after_matching": len(components),
            "unused_components_before_eligibility": len(components) - len(used_roots),
            "eligible_unused_components": len(eligible),
            "eligibility_rate": len(eligible) / max(1, len(components) - len(used_roots)),
        },
        "eligibility_population_metrics": population_metrics,
        "stop_line_scope": (
            "deterministically selected cohort"
            if selected is not None
            else "eligible unused population because no stop-line-feasible cohort exists"
        ),
        "stop_line_metrics": gate_metrics,
        "selected_cohort_metrics": selected_metrics,
        "stop_lines": stop_rows,
        "all_nine_h1_cohort_stop_lines_passed": passed,
        "fusion_auxiliary_analysis_prerequisite": fusion_row,
        "fusion_claim_deletion_flag": not fusion_row["passed"],
        "selection": selection_audit,
        "overlap_exclusion": exclusion,
        "diagnostics": {
            "clip_index_rejections": dict(sorted(clip_rejections.items())),
            "event_loader_rejections": dict(sorted(event_rejections.items())),
            "positive_profile_rejections": dict(sorted(profile_rejections.items())),
            "component_rejections": dict(sorted(component_rejections.items())),
            "cross_camera_matching": match_stats,
        },
        "artifacts": artifacts,
        "commit_performed": False,
    }
    report_sha = write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "eligible_unused_components": len(eligible),
                "selected_cohort_metrics": selected_metrics,
                "artifacts": artifacts,
                "report_sha256": report_sha,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if passed else 2


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_meva = (
        Path.home() / "AppData" / "Local" / "Temp" / "mvevirag_meva_pilot_20260713" / "meva-data-repo"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--meva-repo", type=Path, default=default_meva)
    parser.add_argument("--data-root", type=Path, default=repo_root / "outputs" / "meva_full_20260713")
    parser.add_argument("--pilot-root", type=Path, default=repo_root / "outputs" / "meva_pilot_20260713")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "outputs" / "meva_full_20260713" / "w4_prereg_20260714",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
