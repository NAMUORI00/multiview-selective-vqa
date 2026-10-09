"""Detector-tracker geometric solver for the polarity-balanced partition.

Frozen protocol: outputs/postreview_detector_solver_20260819/protocol_frozen.json

The solver sees only what the answerer saw: the three-frame panel and the
question template identifier. Annotated boxes, annotation-derived quantities,
reference polarity, the surrounding full frame and the event class are never
read during inference; the manifest's reference fields are loaded for scoring
only, after a decision has been emitted.

Two phases, so the GPU work happens once per partition:

    --extract   panel -> per-frame person detections -> appearance-assisted
                association -> displacement, area ratio, simultaneous count.
                Writes features.<partition>.jsonl.

    --fit       grid search on the training partition, single configuration
                selected on validation by four-cell macro accuracy, written to
                frozen_config.json. No test data is read.

    --apply     applies frozen_config.json to a feature file and scores it.

Detector and appearance encoder are the frozen models the article already uses:
IDEA-Research/grounding-dino-base and google/siglip2-base-patch16-384.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ACQ = REPO / "outputs" / "postreview_acceptance_20260814"
OUT = REPO / "outputs" / "postreview_detector_solver_20260819"
PROTOCOL = OUT / "protocol_frozen.json"

TEMPLATES = ("moves_right", "moves_down", "grows_larger", "multiple_people")
CELLS = (("available", "yes"), ("available", "no"),
         ("unavailable", "yes"), ("unavailable", "no"))

# fields the protocol forbids the solver from reading before it answers
FORBIDDEN_AT_INFERENCE = ("target", "targets", "target_text", "question_polarity",
                          "evidence_state", "event_class", "label_derivation",
                          "insufficiency_operator", "full_path")


# ----------------------------------------------------------------- features

def split_panel(img, n: int = 3):
    """A panel is n square frames laid out left to right."""
    w, h = img.size
    step = w // n
    if step != h:
        raise SystemExit(f"panel {w}x{h} is not {n} square frames")
    return [img.crop((i * step, 0, (i + 1) * step, h)) for i in range(n)]


def _configure_backend():
    """cuDNN is broken on the execution host: a trivial conv2d raises
    CUDNN_STATUS_NOT_INITIALIZED on every GPU while cuBLAS is fine. The same
    fault was recorded for the MEVA alignment run in
    outputs/meva_full_20260713/m3_cudnn_patch_deviation_note.json. Disabling
    cuDNN routes convolutions through native CUDA kernels; results are identical
    and no threshold, seed, input or metric changes."""
    import torch
    torch.backends.cudnn.enabled = False
    return {"cudnn_enabled": torch.backends.cudnn.enabled,
            "reason": "host cuDNN raises CUDNN_STATUS_NOT_INITIALIZED on conv2d"}


def build_models(device: str):
    import torch
    from transformers import (AutoModel, AutoProcessor,
                              AutoModelForZeroShotObjectDetection)
    det_id = "IDEA-Research/grounding-dino-base"
    emb_id = "google/siglip2-base-patch16-384"
    det_proc = AutoProcessor.from_pretrained(det_id)
    det = AutoModelForZeroShotObjectDetection.from_pretrained(
        det_id, dtype=torch.float32).to(device).eval()
    emb_proc = AutoProcessor.from_pretrained(emb_id)
    emb = AutoModel.from_pretrained(emb_id, dtype=torch.float32).to(device).eval()
    return {"det_id": det_id, "emb_id": emb_id, "det_proc": det_proc,
            "det": det, "emb_proc": emb_proc, "emb": emb}


def detect_people(models, frames, device, box_threshold=0.25, text_threshold=0.25):
    """Grounding DINO, one prompt, applied to each frame independently."""
    import torch
    proc, det = models["det_proc"], models["det"]
    prompts = ["a person."] * len(frames)
    inputs = proc(images=frames, text=prompts, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = det(**inputs)
    results = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[f.size[::-1] for f in frames])
    per_frame = []
    for res in results:
        boxes = res["boxes"].detach().float().cpu().tolist()
        scores = res["scores"].detach().float().cpu().tolist()
        keep = sorted(zip(scores, boxes), key=lambda t: -t[0])[:8]
        per_frame.append([{"box": b, "score": s} for s, b in keep])
    return per_frame


def embed_boxes(models, frames, per_frame, device):
    """Appearance vector per detection, used for cross-frame association."""
    import torch
    crops, index = [], []
    for fi, (frame, dets) in enumerate(zip(frames, per_frame)):
        for di, d in enumerate(dets):
            x0, y0, x1, y1 = [int(round(v)) for v in d["box"]]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(frame.size[0], max(x1, x0 + 2)), min(frame.size[1], max(y1, y0 + 2))
            crops.append(frame.crop((x0, y0, x1, y1)).resize((384, 384)))
            index.append((fi, di))
    if not crops:
        return
    proc, emb = models["emb_proc"], models["emb"]
    vecs = []
    for i in range(0, len(crops), 32):
        batch = proc(images=crops[i:i + 32], return_tensors="pt").to(device)
        with torch.no_grad():
            v = emb.get_image_features(**batch).float()
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        vecs.append(v.cpu())
    vecs = torch.cat(vecs, 0)
    for (fi, di), v in zip(index, vecs):
        per_frame[fi][di]["emb"] = v.tolist()


def associate(per_frame, appearance_weight=0.7):
    """Hungarian matching between consecutive frames on appearance plus centre distance."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    tracks = [[{"frame": 0, **d}] for d in per_frame[0]] if per_frame and per_frame[0] else []
    for fi in range(1, len(per_frame)):
        curr = per_frame[fi]
        open_tracks = [t for t in tracks if t[-1]["frame"] == fi - 1]
        if not open_tracks or not curr:
            tracks.extend([[{"frame": fi, **d}] for d in curr])
            continue
        cost = np.zeros((len(open_tracks), len(curr)), dtype=float)
        for i, t in enumerate(open_tracks):
            ta = np.asarray(t[-1]["emb"])
            tb = t[-1]["box"]
            tc = ((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)
            for j, d in enumerate(curr):
                ca = np.asarray(d["emb"])
                cb = d["box"]
                cc = ((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2)
                appear = 1.0 - float(ta @ ca)
                dist = ((tc[0] - cc[0]) ** 2 + (tc[1] - cc[1]) ** 2) ** 0.5 / 384.0
                cost[i, j] = appearance_weight * appear + (1 - appearance_weight) * dist
        ri, ci = linear_sum_assignment(cost)
        taken = set()
        for i, j in zip(ri, ci):
            if cost[i, j] < 0.85:
                open_tracks[i].append({"frame": fi, **curr[j]})
                taken.add(j)
        for j, d in enumerate(curr):
            if j not in taken:
                tracks.append([{"frame": fi, **d}])
    return tracks


def panel_features(models, path: Path, device) -> dict:
    from PIL import Image
    t0 = time.time()
    img = Image.open(path).convert("RGB")
    frames = split_panel(img)
    per_frame = detect_people(models, frames, device)
    embed_boxes(models, frames, per_frame, device)
    tracks = associate(per_frame)

    counts = [len(d) for d in per_frame]
    best, best_key = None, None
    for t in tracks:
        key = (len(t), sum(d["score"] for d in t) / len(t))
        if best_key is None or key > best_key:
            best, best_key = t, key

    if best is None:
        feat = dict(track_len=0, track_score=0.0, dx=0.0, dy=0.0, area_ratio=1.0)
    else:
        first, last = best[0], best[-1]
        fb, lb = first["box"], last["box"]
        side = float(frames[0].size[0])
        fc = ((fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2)
        lc = ((lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2)
        fa = max(1.0, (fb[2] - fb[0]) * (fb[3] - fb[1]))
        la = max(1.0, (lb[2] - lb[0]) * (lb[3] - lb[1]))
        feat = dict(track_len=len(best),
                    track_score=sum(d["score"] for d in best) / len(best),
                    dx=(lc[0] - fc[0]) / side,
                    dy=(lc[1] - fc[1]) / side,
                    area_ratio=la / fa)
    feat.update(n_frames=len(frames), max_count=float(max(counts) if counts else 0),
                frame_counts=counts, seconds=time.time() - t0)
    return feat


def cmd_extract(args) -> int:
    import torch
    backend = _configure_backend()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"backend: {backend}", flush=True)
    rows = load_rows(args.partition)
    if args.limit:
        rows = rows[:args.limit]
    # one panel serves several rows; extract once per distinct panel
    panels = {}
    for r in rows:
        panels.setdefault(r["crop_path"], []).append(r)
    print(f"partition {args.partition}: {len(rows)} rows over {len(panels)} panels", flush=True)

    models = build_models(device)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"features.{args.partition}.jsonl"
    done = set()
    if dest.exists() and args.resume:
        for line in dest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["crop_path"])
        print(f"resuming, {len(done)} panels already extracted", flush=True)

    with dest.open("a", encoding="utf-8") as fh:
        for i, (rel, group) in enumerate(sorted(panels.items()), 1):
            if rel in done:
                continue
            feat = panel_features(models, ACQ / rel, device)
            rec = {"crop_path": rel, "event_id": group[0]["event_id"],
                   "question_ids": sorted({g["question_id"] for g in group}),
                   "partition": args.partition, **feat}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(panels):
                print(f"  {i}/{len(panels)} panels", flush=True)
    print(f"wrote {dest}")
    return 0


# ---------------------------------------------------------------- decisions

def decide(feat: dict, q: str, cfg: dict) -> str:
    """Emit yes, no or abstain from panel-derived quantities alone."""
    if feat["track_len"] < cfg["track_len_min"] or feat["track_score"] < cfg["det_score_min"]:
        return "abstain"
    if q == "multiple_people":
        return "yes" if feat["max_count"] >= 2 else "no"
    if q == "moves_right":
        return "yes" if feat["dx"] >= cfg["t_move"] else "no"
    if q == "moves_down":
        return "yes" if feat["dy"] >= cfg["t_move"] else "no"
    if q == "grows_larger":
        return "yes" if feat["area_ratio"] >= cfg["t_grow"] else "no"
    raise SystemExit(f"unknown template {q}")


def score(rows, feats, cfg):
    by_panel = {f["crop_path"]: f for f in feats}
    per_cell = {c: [0, 0] for c in CELLS}
    per_event = {}
    confusion = {}
    for r in rows:
        f = by_panel.get(r["crop_path"])
        if f is None:
            continue
        pred = decide(f, r["question_id"], cfg)
        ok = pred == r["target"]
        cell = (r["evidence_state"], r["question_polarity"])
        per_cell[cell][0] += int(ok)
        per_cell[cell][1] += 1
        per_event.setdefault(r["event_id"], []).append(ok)
        confusion[(r["target"], pred)] = confusion.get((r["target"], pred), 0) + 1
    cells = {f"{s}-{p}": (100.0 * n / d if d else float("nan"))
             for (s, p), (n, d) in per_cell.items()}
    macro = sum(v for v in cells.values()) / len(cells)
    allfour = 100.0 * sum(1 for v in per_event.values() if len(v) == 4 and all(v)) / max(1, len(per_event))
    return {"cells": cells, "macro": macro, "all_four": allfour,
            "events": len(per_event),
            "confusion": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items())}}


C1 = REPO / "outputs" / "postreview_c1_test_20260818"
ACQ_MARKER = "outputs/postreview_acceptance_20260814/"


def load_rows(partition):
    """Development rows come from the acquisition manifest; the test rows live
    with the opened prospective run, which records the panel hash per row."""
    if partition != "test":
        rows = [json.loads(l) for l in (ACQ / "manifests" / "final_rows.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
        return [r for r in rows if r["partition"] == partition]

    src = C1 / "responses" / "paired_sft.jsonl"
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        raw = r["crop_path"].replace("\\", "/")
        if ACQ_MARKER not in raw:
            raise SystemExit(f"unexpected test crop path: {raw}")
        rows.append({**r, "partition": "test",
                     "crop_path": raw.split(ACQ_MARKER, 1)[1]})
    return rows


def cmd_fit(args) -> int:
    train_rows, dev_rows = load_rows("train"), load_rows("dev")
    train_f = [json.loads(l) for l in (OUT / "features.train.jsonl")
               .read_text(encoding="utf-8").splitlines() if l.strip()]
    dev_f = [json.loads(l) for l in (OUT / "features.dev.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]

    grid = [dict(track_len_min=a, det_score_min=b, t_move=c, t_grow=d)
            for a in (2, 3)
            for b in (0.25, 0.30, 0.35, 0.40)
            for c in (0.0, 0.01, 0.02, 0.03, 0.05)
            for d in (1.00, 1.05, 1.10, 1.15, 1.25)]
    print(f"grid: {len(grid)} configurations", flush=True)

    scored = []
    for cfg in grid:
        tr = score(train_rows, train_f, cfg)
        scored.append((tr["macro"], tr["all_four"], -cfg["track_len_min"], cfg, tr))
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    shortlist = [s[3] for s in scored[:12]]

    best = None
    for cfg in shortlist:
        dv = score(dev_rows, dev_f, cfg)
        key = (dv["macro"], dv["all_four"], -sum(
            n for k, n in dv["confusion"].items() if k.endswith("->abstain")))
        if best is None or key > best[0]:
            best = (key, cfg, dv)
    _, cfg, dv = best
    frozen = {"schema_version": "postreview_detector_solver_config_v1",
              "selected_on": "validation partition, four-cell macro then all-four then fewer abstentions",
              "config": cfg,
              "train_score": score(train_rows, train_f, cfg),
              "validation_score": dv,
              "grid_size": len(grid), "shortlist_size": len(shortlist),
              "test_data_read": False}
    (OUT / "frozen_config.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"config": cfg, "train_macro": frozen["train_score"]["macro"],
                      "dev_macro": dv["macro"], "dev_all_four": dv["all_four"]},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_apply(args) -> int:
    cfg_doc = json.loads((OUT / "frozen_config.json").read_text(encoding="utf-8"))
    cfg = cfg_doc["config"]
    rows = load_rows(args.partition)
    feats = [json.loads(l) for l in (OUT / f"features.{args.partition}.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    res = score(rows, feats, cfg)
    per_template = {}
    for q in TEMPLATES:
        sub = [r for r in rows if r["question_id"] == q]
        per_template[q] = score(sub, feats, cfg)
    doc = {"schema_version": "postreview_detector_solver_result_v1",
           "evidence_status": "post_hoc_added_baseline_with_pre_execution_frozen_protocol",
           "partition": args.partition, "config": cfg,
           "overall": res, "per_template": per_template,
           "seconds_per_panel": sum(f["seconds"] for f in feats) / max(1, len(feats))}
    (OUT / f"result.{args.partition}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"macro": res["macro"], "all_four": res["all_four"],
                      "cells": res["cells"], "confusion": res["confusion"]},
                     ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--partition", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not PROTOCOL.exists():
        raise SystemExit("frozen protocol missing; refusing to run")
    if args.extract:
        return cmd_extract(args)
    if args.fit:
        return cmd_fit(args)
    if args.apply:
        return cmd_apply(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
