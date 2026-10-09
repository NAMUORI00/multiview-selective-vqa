"""Shared train/dev-only inputs for the frozen W3 and W5a runners."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

SEED = 20260713
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
VLLM_IMAGE = "vllm/vllm-openai@sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f"
ABSTAIN = "The visual evidence is insufficient to determine this."
EXPECTED_EVENTS = {"train": 203, "dev": 82}
EXPECTED_ROWS = {"train": 406, "dev": 164}

ACTIVITY_TEXT = {
    "hand_interacts_with_person": (
        "Is a person physically interacting with another person by hand?",
        "A person is physically interacting with another person by hand.",
    ),
    "person_carries_heavy_object": (
        "Is a person carrying a heavy object?",
        "A person is carrying a heavy object.",
    ),
    "person_closes_vehicle_door": (
        "Is a person closing a vehicle door?",
        "A person is closing a vehicle door.",
    ),
    "person_enters_scene_through_structure": (
        "Is a person entering the scene through a structure?",
        "A person is entering the scene through a structure.",
    ),
    "person_enters_vehicle": (
        "Is a person entering a vehicle?",
        "A person is entering a vehicle.",
    ),
    "person_exits_scene_through_structure": (
        "Is a person exiting the scene through a structure?",
        "A person is exiting the scene through a structure.",
    ),
    "person_exits_vehicle": (
        "Is a person exiting a vehicle?",
        "A person is exiting a vehicle.",
    ),
    "person_opens_facility_door": (
        "Is a person opening a facility door?",
        "A person is opening a facility door.",
    ),
    "person_opens_vehicle_door": (
        "Is a person opening a vehicle door?",
        "A person is opening a vehicle door.",
    ),
    "person_talks_on_phone": (
        "Is a person talking on a phone?",
        "A person is talking on a phone.",
    ),
    "person_talks_to_person": (
        "Is a person talking to another person?",
        "A person is talking to another person.",
    ),
    "person_texts_on_phone": (
        "Is a person texting on a phone?",
        "A person is texting on a phone.",
    ),
}

GEOM_RE = re.compile(
    r"'g0':\s*'(?P<x1>-?\d+(?:\.\d+)?) (?P<y1>-?\d+(?:\.\d+)?) "
    r"(?P<x2>-?\d+(?:\.\d+)?) (?P<y2>-?\d+(?:\.\d+)?)'.*"
    r"'id1':\s*(?P<id>\d+).*'ts0':\s*(?P<frame>\d+)"
)
TYPE_RE = re.compile(r"'cset3':\s*\{(?P<classes>[^}]*)\}.*'id1':\s*(?P<id>\d+)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_root() -> Path:
    return repo_root() / "outputs" / "meva_full_20260713"


def default_meva_repo() -> Path:
    return Path.home() / "AppData" / "Local" / "Temp" / "mvevirag_meva_pilot_20260713" / "meva-data-repo"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(f"{SEED}|{text}".encode()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def target_payload(decision: str, answer: str) -> str:
    return json.dumps({"decision": decision, "answer": answer}, ensure_ascii=False, separators=(",", ":"))


def negative_answer(activity: str) -> str:
    question = ACTIVITY_TEXT[activity][0]
    subject = question.removeprefix("Is ").removesuffix("?")
    return f"No, {subject[0].lower() + subject[1:]} is not shown."


def _sample_table(data_root: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(data_root / "v4_samples.csv")
    table = {row["sample_id"]: row for row in rows if row["split"] in EXPECTED_ROWS}
    if len(table) != sum(EXPECTED_ROWS.values()):
        raise RuntimeError(f"train/dev v4 sample count drift: {len(table)}")
    return table


def load_rows(data_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Load only allowlisted train/dev manifests and attach frozen v4 geometry fields."""
    sample_table = _sample_table(data_root)
    result: dict[str, list[dict[str, Any]]] = {}
    event_sets: dict[str, set[str]] = {}
    for split in ("train", "dev"):
        manifest = data_root / f"v4_{split}_samples_m3.csv"
        source_rows = read_csv(manifest)
        if len(source_rows) != EXPECTED_ROWS[split]:
            raise RuntimeError(f"{split} row count drift: {len(source_rows)}")
        rows = []
        for source in source_rows:
            if source["split"] != split or source["label"] not in {"positive", "negative"}:
                raise RuntimeError(f"train/dev firewall violation: {source['sample_id']}")
            sample = sample_table[source["sample_id"]]
            frame = Path(source["frame_path"])
            crop = Path(source["crop_path"])
            if not frame.is_file() or not crop.is_file():
                raise FileNotFoundError(f"missing media: {source['sample_id']}")
            if sha256_file(crop) != source["crop_sha256"].lower():
                raise RuntimeError(f"crop hash mismatch: {source['sample_id']}")
            label = 1 if source["label"] == "positive" else 0
            question, affirmative = ACTIVITY_TEXT[source["activity"]]
            rows.append(
                {
                    "sample_id": source["sample_id"],
                    "row_id": source["sample_id"],
                    "event_id": source["independent_event_id"],
                    "group_id": source["independent_event_id"],
                    "activity": source["activity"],
                    "split": split,
                    "label": label,
                    "camera_id": source["camera_id"],
                    "clip": source["clip"],
                    "native_frame": int(sample["native_frame"]),
                    "full_path": str(frame.resolve()),
                    "full_sha256": sha256_file(frame),
                    "crop_path": str(crop.resolve()),
                    "crop_sha256": source["crop_sha256"].lower(),
                    "crop_bounds_xyxy": [int(value) for value in sample["crop_bounds_xyxy"].split()],
                    "question": question,
                    "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                    "target_answer": affirmative if label else ABSTAIN,
                    "target_type": "answer" if label else "abstain",
                }
            )
        rows.sort(key=lambda row: row["row_id"])
        events = {row["event_id"] for row in rows}
        if len(events) != EXPECTED_EVENTS[split]:
            raise RuntimeError(f"{split} event count drift: {len(events)}")
        if any(sum(row["event_id"] == event for row in rows) != 2 for event in events):
            raise RuntimeError(f"{split} paired-row invariant failed")
        result[split] = rows
        event_sets[split] = events
    if event_sets["train"] & event_sets["dev"]:
        raise RuntimeError("train/dev event overlap")
    return result


def annotation_index(meva_repo: Path, clips: set[str]) -> dict[str, tuple[Path, Path]]:
    if not meva_repo.is_dir():
        raise FileNotFoundError(f"MEVA annotation repository missing: {meva_repo}")
    wanted = {
        filename: clip
        for clip in clips
        for filename in (f"{clip}.geom.yml", f"{clip}-geom.yml")
    }
    found: dict[str, list[Path]] = defaultdict(list)
    for path in meva_repo.rglob("*geom.yml"):
        clip = wanted.get(path.name)
        if clip is not None:
            found[clip].append(path)
    index = {}

    def source_rank(path: Path) -> tuple[int, str]:
        normalized = path.as_posix().lower()
        priorities = ("/kitware-meva-training/", "/kitware/", "/contrib/cmu/", "/contrib/ibm/")
        rank = next((number for number, token in enumerate(priorities) if token in normalized), len(priorities))
        return rank, normalized

    for clip in sorted(clips):
        candidates = found.get(clip, [])
        if not candidates:
            raise RuntimeError(f"no geometry file for {clip}")
        geom = min(candidates, key=source_rank)
        if geom.name.endswith(".geom.yml"):
            types = geom.with_name(geom.name.replace(".geom.yml", ".types.yml"))
        else:
            types = geom.with_name(geom.name.replace("-geom.yml", "-types.yml"))
        if not types.is_file():
            raise FileNotFoundError(types)
        index[clip] = (geom, types)
    return index


def _person_ids(types_path: Path) -> set[int]:
    ids = set()
    with types_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = TYPE_RE.search(line)
            if match and re.search(r"['\"]person['\"]\s*:\s*1(?:\.0+)?", match.group("classes")):
                ids.add(int(match.group("id")))
    return ids


def person_boxes(geom_path: Path, types_path: Path, frame: int) -> list[list[float]]:
    people = _person_ids(types_path)
    boxes = []
    with geom_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = GEOM_RE.search(line)
            if not match or int(match.group("frame")) != frame or int(match.group("id")) not in people:
                continue
            box = [float(match.group(name)) for name in ("x1", "y1", "x2", "y2")]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise RuntimeError(f"invalid person box in {geom_path.name}: {box}")
            boxes.append(box)
    return sorted(boxes)


def union_area(boxes: list[list[float]], width: int = 1920, height: int = 1080) -> float:
    """Exact union area for a small axis-aligned rectangle set."""
    clipped = [
        [max(0.0, b[0]), max(0.0, b[1]), min(float(width), b[2]), min(float(height), b[3])]
        for b in boxes
        if min(float(width), b[2]) > max(0.0, b[0]) and min(float(height), b[3]) > max(0.0, b[1])
    ]
    xs = sorted({value for box in clipped for value in (box[0], box[2])})
    area = 0.0
    for left, right in pairwise(xs):
        intervals = sorted((box[1], box[3]) for box in clipped if box[0] < right and box[2] > left)
        covered = 0.0
        start = end = None
        for low, high in intervals:
            if start is None:
                start, end = low, high
            elif low > end:
                covered += end - start
                start, end = low, high
            else:
                end = max(end, high)
        if start is not None:
            covered += end - start
        area += (right - left) * covered
    return area


def attach_person_geometry(rows: dict[str, list[dict[str, Any]]], meva_repo: Path) -> dict[str, Any]:
    all_rows = rows["train"] + rows["dev"]
    index = annotation_index(meva_repo, {row["clip"] for row in all_rows})
    annotation_hashes = {}
    for clip, (geom, types) in index.items():
        annotation_hashes[clip] = {
            "geom_path": str(geom.resolve()),
            "geom_sha256": sha256_file(geom),
            "types_path": str(types.resolve()),
            "types_sha256": sha256_file(types),
        }
    for row in all_rows:
        geom, types = index[row["clip"]]
        boxes = person_boxes(geom, types, row["native_frame"])
        row["person_boxes_xyxy"] = boxes
        row["person_present"] = bool(boxes)
        row["person_count"] = len(boxes)
        row["person_union_area_ratio"] = union_area(boxes) / (1920 * 1080)
        row["annotation_geom_sha256"] = annotation_hashes[row["clip"]]["geom_sha256"]
        row["annotation_types_sha256"] = annotation_hashes[row["clip"]]["types_sha256"]
        row["annotation_geom_path"] = annotation_hashes[row["clip"]]["geom_path"]
        row["annotation_types_path"] = annotation_hashes[row["clip"]]["types_path"]
    return annotation_hashes


def directory_stats(path: Path, pattern: str = "*.png") -> dict[str, int]:
    files = list(path.glob(pattern))
    return {"files": len(files), "bytes": sum(item.stat().st_size for item in files)}
