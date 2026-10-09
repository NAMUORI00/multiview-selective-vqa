"""Frozen same-Qwen W5a training, inference, scoring, and dev selection runner."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.operator import answerer_alignment_lora as base
from scripts.operator.w3w5a_common import (
    ABSTAIN,
    ACTIVITY_TEXT,
    MODEL_ID,
    SEED,
    VLLM_IMAGE,
    default_data_root,
    load_rows,
    read_jsonl,
    repo_root,
    sha256_file,
    target_payload,
    write_json,
    write_jsonl,
)

ARMS = (
    "zero_shot",
    "explicit_abstention_prompt",
    "answer_only_sft",
    "answer_abstain_sft",
    "correctness_selector",
    "action_classification_head",
)
TRAINED_ARMS = ARMS[2:]
ANSWERER_SFT_ARMS = ("answer_only_sft", "answer_abstain_sft")
HEAD_ARMS = ("correctness_selector", "action_classification_head")
SEQUENCE_SLOTS = 1536
TRAIN_ROWS = 406
TRAIN_PASSES = 3
UPDATE_TOKEN_SLOTS = TRAIN_ROWS * TRAIN_PASSES * SEQUENCE_SLOTS
GRADIENT_ACCUMULATION = 8
UPDATES_PER_PASS = math.ceil(TRAIN_ROWS / GRADIENT_ACCUMULATION)
TOTAL_UPDATES = UPDATES_PER_PASS * TRAIN_PASSES
NEUTRAL_SYSTEM_PROMPT = "Answer the question using only the supplied visual evidence. Output one JSON object with keys decision and answer."
EXPLICIT_SYSTEM_PROMPT = (
    "Use only the supplied visual evidence. Answer yes or no when the evidence supports a determination. "
    f"If it does not, answer exactly '{ABSTAIN}'. Output one JSON object with keys decision and answer; "
    "decision is answer or abstain."
)
GPU_WAVES = {
    1: {0: "answer_only_sft", 1: "answer_abstain_sft", 2: "correctness_selector"},
    2: {0: "zero_shot", 1: "explicit_abstention_prompt", 2: "action_classification_head"},
}


def output_root_default() -> Path:
    return default_data_root() / "w5a_fair_baselines"


def _negative_answer(activity: str) -> str:
    subject = ACTIVITY_TEXT[activity][0].removeprefix("Is ").removesuffix("?")
    return f"No, {subject[0].lower() + subject[1:]} is not shown."


def _base_row(row: dict[str, Any], data_root: Path) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "row_id",
            "sample_id",
            "event_id",
            "group_id",
            "activity",
            "split",
            "label",
            "camera_id",
            "question",
            "target_answer",
            "target_type",
        )
    } | {
        "full_path": Path(row["full_path"]).resolve().relative_to(data_root.resolve()).as_posix(),
        "full_sha256": row["full_sha256"],
        "crop_path": Path(row["crop_path"]).resolve().relative_to(data_root.resolve()).as_posix(),
        "crop_sha256": row["crop_sha256"],
        "input_condition": "full_native_crop",
    }


def _targets(row: dict[str, Any]) -> dict[str, Any]:
    affirmative = ACTIVITY_TEXT[row["activity"]][1]
    return {
        "answer_only_sft": target_payload(
            "answer", affirmative if row["label"] else _negative_answer(row["activity"])
        ),
        "answer_abstain_sft": target_payload("answer", affirmative)
        if row["label"]
        else target_payload("abstain", ABSTAIN),
        "correctness_selector": int(row["label"]),
        "action_classification_head": row["activity"] if row["label"] else "none",
    }


def build_manifests(data_root: Path, output_root: Path, materialize: bool) -> dict[str, Any]:
    rows = load_rows(data_root)
    catalog_path = output_root / "manifests" / "question_targets.json"
    write_json(
        catalog_path,
        {
            "schema_version": "meva_w5a_question_targets_v1",
            "activities": {
                activity: {
                    "question": values[0],
                    "affirmative": values[1],
                    "negative": _negative_answer(activity),
                }
                for activity, values in sorted(ACTIVITY_TEXT.items())
            },
            "abstention": ABSTAIN,
        },
    )
    catalog_sha = sha256_file(catalog_path)
    train = []
    for row in rows["train"]:
        item = _base_row(row, data_root)
        item["targets"] = _targets(row)
        item["question_target_catalog"] = {"relpath": "manifests/question_targets.json", "sha256": catalog_sha}
        item["question_sha256"] = hashlib.sha256(item["question"].encode()).hexdigest()
        train.append(item)
    dev = []
    for row in rows["dev"]:
        item = _base_row(row, data_root)
        item["targets"] = _targets(row)
        item["question_target_catalog"] = {"relpath": "manifests/question_targets.json", "sha256": catalog_sha}
        item["question_sha256"] = hashlib.sha256(item["question"].encode()).hexdigest()
        item["evaluation_role"] = "pplus" if row["label"] else "eminus"
        dev.append(item)
        if row["label"]:
            dev.append(
                item
                | {
                    "row_id": f"{row['row_id']}::pminus",
                    "sample_id": f"{row['sample_id']}::pminus",
                    "label": 0,
                    "target_answer": ABSTAIN,
                    "target_type": "abstain",
                    "evaluation_role": "pminus",
                    "full_path": "media/neutral_384.png",
                    "crop_path": "media/neutral_384.png",
                    "full_sha256": None,
                    "crop_sha256": None,
                    "input_condition": "crop_only",
                    "media_root": "output",
                    "targets": {
                        "answer_only_sft": target_payload("answer", _negative_answer(row["activity"])),
                        "answer_abstain_sft": target_payload("abstain", ABSTAIN),
                        "correctness_selector": 0,
                        "action_classification_head": "none",
                    },
                }
            )
    train.sort(key=lambda row: row["row_id"])
    dev.sort(key=lambda row: row["row_id"])
    if len(train) != 406 or len(dev) != 246:
        raise RuntimeError("W5a train/dev manifest count invariant failed")
    if len({row["event_id"] for row in train}) != 203 or len({row["event_id"] for row in dev}) != 82:
        raise RuntimeError("W5a independent-event invariant failed")
    write_jsonl(output_root / "manifests" / "train.jsonl", train)
    write_jsonl(output_root / "manifests" / "eval_dev.jsonl", dev)
    train_sha = sha256_file(output_root / "manifests" / "train.jsonl")
    dev_sha = sha256_file(output_root / "manifests" / "eval_dev.jsonl")
    for arm in ARMS:
        write_json(
            output_root / "manifests" / f"{arm}_input.json",
            {
                "schema_version": "meva_w5a_arm_input_v1",
                "arm": arm,
                "train": None if arm in ARMS[:2] else {"path": "manifests/train.jsonl", "rows": 406, "sha256": train_sha},
                "dev": {"path": "manifests/eval_dev.jsonl", "rows": 246, "sha256": dev_sha},
                "question_targets": {"path": "manifests/question_targets.json", "sha256": catalog_sha},
            },
        )
    if materialize:
        from PIL import Image

        neutral = output_root / "media" / "neutral_384.png"
        neutral.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (384, 384), (127, 127, 127)).save(neutral)
    source_bytes = sum(Path(row["full_path"]).stat().st_size + Path(row["crop_path"]).stat().st_size for row in rows["train"] + rows["dev"])
    report = {
        "schema_version": "meva_w5a_dry_run_v1",
        "scope": "MEVA train/dev only; no sealed test, reserve, or new-cohort access",
        "seed": SEED,
        "base_model": MODEL_ID,
        "train": {"events": 203, "rows": 406, "full_frames": 406, "crops": 406},
        "dev": {"events": 82, "source_rows": 164, "evaluation_rows": 246, "full_frames": 164, "crops": 164},
        "fairness_budget": {
            "trained_arms": list(TRAINED_ARMS),
            "independent_events": 203,
            "supervised_rows_per_pass": TRAIN_ROWS,
            "passes": TRAIN_PASSES,
            "sequence_token_slots_per_row": SEQUENCE_SLOTS,
            "update_token_slots": UPDATE_TOKEN_SLOTS,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "optimizer_updates": TOTAL_UPDATES,
        },
        "source_media_bytes_already_on_server": source_bytes,
        "new_transfer_estimate_bytes": 4096,
        "new_transfer_note": "only one neutral proxy image; train/dev full frames and crops are already verified on gpu-host",
        "gpu_execution_started": False,
        "model_loaded": False,
        "materialized": materialize,
    }
    write_json(output_root / "w5a_dry_run_report.json", report)
    return report


def _runtime_row(row: dict[str, Any], data_root: Path, output_root: Path) -> dict[str, Any]:
    root = output_root if row.get("media_root") == "output" else data_root
    result = row | {
        "full_path": str((root / row["full_path"]).resolve()),
        "crop_path": str((root / row["crop_path"]).resolve()),
    }
    for field in ("full_path", "crop_path"):
        if not Path(result[field]).is_file():
            raise FileNotFoundError(result[field])
    return result


def _messages(row: dict[str, Any], system_prompt: str, target: str | None = None) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": base.user_content(row)},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


def _encode_sft(processor: Any, rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    import torch

    prompts, full_texts, image_batches = [], [], []
    for row in rows:
        prompt_messages = _messages(row, EXPLICIT_SYSTEM_PROMPT)
        prompts.append(processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True))
        full_texts.append(
            processor.apply_chat_template(
                _messages(row, EXPLICIT_SYSTEM_PROMPT, row["targets"][arm]), tokenize=False, add_generation_prompt=False
            )
        )
        image_batches.append(base.row_images(row))
    prompt = processor(
        text=prompts,
        images=image_batches,
        padding="max_length",
        truncation=True,
        max_length=SEQUENCE_SLOTS,
        return_tensors="pt",
    )
    encoded = processor(
        text=full_texts,
        images=image_batches,
        padding="max_length",
        truncation=True,
        max_length=SEQUENCE_SLOTS,
        return_tensors="pt",
    )
    if encoded["input_ids"].shape[1] != SEQUENCE_SLOTS:
        raise RuntimeError("fixed update-token slot invariant failed")
    labels = encoded["input_ids"].clone()
    for index, prompt_length in enumerate(prompt["attention_mask"].sum(dim=1)):
        labels[index, : int(prompt_length)] = -100
    labels[encoded["attention_mask"] == 0] = -100
    if any(int((labels[index] != -100).sum()) == 0 for index in range(len(rows))):
        raise RuntimeError("SFT target was truncated from the fixed sequence")
    encoded["labels"] = labels
    for images in image_batches:
        for image in images:
            image.close()
    return {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}


class RowDataset:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def train_sft(args: argparse.Namespace, train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    if args.arm not in ANSWERER_SFT_ARMS:
        raise ValueError(args.arm)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False
    model, processor = base.load_model_and_processor(adapter=None, trainable=True)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(RowDataset(train), batch_size=1, shuffle=True, generator=generator, collate_fn=lambda x: x)
    dev_loader = DataLoader(RowDataset(dev), batch_size=1, shuffle=False, collate_fn=lambda x: x)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=1e-4, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(TOTAL_UPDATES * 0.03)), TOTAL_UPDATES)
    run_dir = args.output_root / "training" / args.arm
    run_dir.mkdir(parents=True, exist_ok=True)
    best = (float("inf"), 0, None)
    global_update = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for epoch in range(1, TRAIN_PASSES + 1):
        model.train()
        accumulated = 0
        for step, batch_rows in enumerate(train_loader, 1):
            batch = {key: value.to("cuda:0") for key, value in _encode_sft(processor, batch_rows, args.arm).items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite SFT loss: {args.arm} epoch={epoch} step={step}")
            (loss / GRADIENT_ACCUMULATION).backward()
            accumulated += 1
            if step == 1 or accumulated == GRADIENT_ACCUMULATION or step == len(train_loader):
                print(
                    f"arm={args.arm} epoch={epoch} step={step}/{len(train_loader)} "
                    f"update={global_update} loss={float(loss.detach().cpu()):.6f}",
                    flush=True,
                )
            if accumulated == GRADIENT_ACCUMULATION or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                accumulated = 0
        model.eval()
        loss_sum = token_sum = 0
        with torch.inference_mode():
            for batch_rows in dev_loader:
                batch = {key: value.to("cuda:0") for key, value in _encode_sft(processor, batch_rows, args.arm).items()}
                supervised = int((batch["labels"] != -100).sum())
                loss_sum += float(model(**batch).loss.detach().cpu()) * supervised
                token_sum += supervised
        dev_loss = loss_sum / token_sum
        checkpoint = run_dir / f"epoch-{epoch}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        processor.save_pretrained(checkpoint)
        if (dev_loss, epoch) < (best[0], best[1] or 10**9):
            best = (dev_loss, epoch, checkpoint)
    if global_update != TOTAL_UPDATES or best[2] is None:
        raise RuntimeError("SFT update/checkpoint invariant failed")
    write_json(
        run_dir / "checkpoint_selection.json",
        {
            "arm": args.arm,
            "best_checkpoint": str(best[2].resolve()),
            "best_epoch": best[1],
            "best_validation_loss": best[0],
            "selection": "minimum arm-specific dev loss; exact tie chooses earlier epoch",
            "update_token_slots": UPDATE_TOKEN_SLOTS,
            "optimizer_updates": global_update,
            "cudnn_enabled": False,
            "elapsed_seconds": time.time() - started,
        },
    )


def _head_prompt(row: dict[str, Any], arm: str) -> str:
    if arm == "correctness_selector":
        return f"Predict whether the visual evidence supports a correct answer to: {row['question']}"
    return "Classify the visible target action as one of the frozen activity labels or none."


def _encode_head(processor: Any, rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    import torch

    allow_sequence_expansion = bool(rows) and all(
        row.get("split") == "b1_sealed_cohort" for row in rows
    )
    texts, image_batches = [], []
    for row in rows:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": _head_prompt(row, arm)}]},
            {"role": "user", "content": base.user_content(row)},
        ]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        image_batches.append(base.row_images(row))
    encoded = processor(
        text=texts,
        images=image_batches,
        padding="max_length",
        truncation=not allow_sequence_expansion,
        max_length=SEQUENCE_SLOTS,
        return_tensors="pt",
    )
    encoded_slots = encoded["input_ids"].shape[1]
    if encoded_slots < SEQUENCE_SLOTS or (
        not allow_sequence_expansion and encoded_slots != SEQUENCE_SLOTS
    ):
        raise RuntimeError("head update-token slot invariant failed")
    for images in image_batches:
        for image in images:
            image.close()
    return {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}


def _head_labels(rows: list[dict[str, Any]], arm: str, classes: list[str]) -> np.ndarray:
    if arm == "correctness_selector":
        return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    return np.asarray([classes.index(row["activity"] if row["label"] else "none") for row in rows], dtype=np.int64)


def train_head(args: argparse.Namespace, train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> None:
    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    if args.arm not in HEAD_ARMS:
        raise ValueError(args.arm)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False
    model, processor = base.load_model_and_processor(adapter=None, trainable=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    classes = ["unsupported", "supported"] if args.arm == "correctness_selector" else ["none"] + sorted(ACTIVITY_TEXT)
    hidden_size = int(model.config.text_config.hidden_size)
    head = nn.Linear(hidden_size, len(classes)).to(device="cuda:0", dtype=torch.float32)
    optimizer = AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(RowDataset(train), batch_size=1, shuffle=True, generator=generator, collate_fn=lambda x: x)
    dev_loader = DataLoader(RowDataset(dev), batch_size=1, shuffle=False, collate_fn=lambda x: x)
    run_dir = args.output_root / "training" / args.arm
    run_dir.mkdir(parents=True, exist_ok=True)
    best = (float("inf"), 0, None)
    global_update = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for epoch in range(1, TRAIN_PASSES + 1):
        head.train()
        accumulated = 0
        for step, batch_rows in enumerate(train_loader, 1):
            batch = {key: value.to("cuda:0") for key, value in _encode_head(processor, batch_rows, args.arm).items()}
            with torch.no_grad():
                output = model(**batch, output_hidden_states=True, return_dict=True)
                positions = batch["attention_mask"].sum(dim=1) - 1
                last = output.hidden_states[-1][torch.arange(len(batch_rows), device="cuda:0"), positions].float()
            labels = torch.as_tensor(_head_labels(batch_rows, args.arm, classes), device="cuda:0")
            loss = nn.functional.cross_entropy(head(last), labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite head loss: {args.arm} epoch={epoch} step={step}")
            (loss / GRADIENT_ACCUMULATION).backward()
            accumulated += 1
            if step == 1 or accumulated == GRADIENT_ACCUMULATION or step == len(train_loader):
                print(
                    f"arm={args.arm} epoch={epoch} step={step}/{len(train_loader)} "
                    f"update={global_update} loss={float(loss.detach().cpu()):.6f}",
                    flush=True,
                )
            if accumulated == GRADIENT_ACCUMULATION or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                accumulated = 0
        head.eval()
        losses = []
        with torch.inference_mode():
            for batch_rows in dev_loader:
                batch = {key: value.to("cuda:0") for key, value in _encode_head(processor, batch_rows, args.arm).items()}
                output = model(**batch, output_hidden_states=True, return_dict=True)
                labels = torch.as_tensor(_head_labels(batch_rows, args.arm, classes), device="cuda:0")
                positions = batch["attention_mask"].sum(dim=1) - 1
                hidden = output.hidden_states[-1][torch.arange(len(batch_rows), device="cuda:0"), positions].float()
                losses.append(float(nn.functional.cross_entropy(head(hidden), labels).cpu()))
        dev_loss = float(np.mean(losses))
        checkpoint = run_dir / f"epoch-{epoch}.pt"
        torch.save({"state_dict": head.state_dict(), "classes": classes, "hidden_size": hidden_size}, checkpoint)
        if (dev_loss, epoch) < (best[0], best[1] or 10**9):
            best = (dev_loss, epoch, checkpoint)
    if global_update != TOTAL_UPDATES or best[2] is None:
        raise RuntimeError("head update/checkpoint invariant failed")
    write_json(
        run_dir / "checkpoint_selection.json",
        {
            "arm": args.arm,
            "best_checkpoint": str(best[2].resolve()),
            "best_epoch": best[1],
            "best_validation_loss": best[0],
            "selection": "minimum arm-specific dev cross-entropy; exact tie chooses earlier epoch",
            "update_token_slots": UPDATE_TOKEN_SLOTS,
            "optimizer_updates": global_update,
            "cudnn_enabled": False,
            "elapsed_seconds": time.time() - started,
        },
    )


def infer_answerer(args: argparse.Namespace, dev: list[dict[str, Any]]) -> None:
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter = None
    if args.arm in ANSWERER_SFT_ARMS:
        selection = json.loads((args.output_root / "training" / args.arm / "checkpoint_selection.json").read_text(encoding="utf-8"))
        adapter = Path(selection["best_checkpoint"])
        if not adapter.is_dir():
            raise FileNotFoundError(adapter)
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
    request = LoRARequest(args.arm, 1, lora_path=str(adapter)) if adapter else None
    system = NEUTRAL_SYSTEM_PROMPT if args.arm == "zero_shot" else EXPLICIT_SYSTEM_PROMPT
    result_path = args.output_root / "responses" / f"{args.arm}.jsonl"
    existing = {row["row_id"]: row for row in read_jsonl(result_path)} if result_path.is_file() else {}
    pending = [row for row in dev if row["row_id"] not in existing]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8", newline="\n") as handle:
        for offset in range(0, len(pending), args.batch_size):
            batch_rows = pending[offset : offset + args.batch_size]
            prompts, opened = [], []
            for row in batch_rows:
                images = base.row_images(row)
                opened.extend(images)
                prompt = processor.apply_chat_template(_messages(row, system), tokenize=False, add_generation_prompt=True)
                prompts.append({"prompt": prompt, "multi_modal_data": {"image": images}})
            generated = llm.generate(prompts, sampling_params=sampling, lora_request=request, use_tqdm=False)
            for row, output in zip(batch_rows, generated, strict=True):
                candidate = output.outputs[0]
                if not candidate.token_ids:
                    raise RuntimeError(f"empty generation: {args.arm}/{row['row_id']}")
                decision, answer, abstained = base.parse_answer(candidate.text.strip())
                correct = (
                    base.positive_correct(answer, decision, abstained, row["target_answer"])
                    if row["evaluation_role"] == "pplus"
                    else abstained
                )
                handle.write(
                    json.dumps(
                        row
                        | {
                            "arm": args.arm,
                            "decision": decision,
                            "answer": answer,
                            "was_abstained": bool(abstained),
                            "target_correct": bool(correct),
                            "confidence": math.exp(candidate.cumulative_logprob / len(candidate.token_ids)),
                            "raw_response": candidate.text.strip(),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                handle.flush()
            for image in opened:
                image.close()
    if len(read_jsonl(result_path)) != 246:
        raise RuntimeError(f"incomplete W5a answerer inference: {args.arm}")


def infer_head(args: argparse.Namespace, dev: list[dict[str, Any]]) -> None:
    import torch
    from torch import nn

    torch.backends.cudnn.enabled = False
    selection = json.loads((args.output_root / "training" / args.arm / "checkpoint_selection.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(selection["best_checkpoint"], map_location="cpu", weights_only=True)
    model, processor = base.load_model_and_processor(adapter=None, trainable=False)
    model.eval()
    head = nn.Linear(int(checkpoint["hidden_size"]), len(checkpoint["classes"])).to("cuda:0", dtype=torch.float32)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    result_path = args.output_root / "responses" / f"{args.arm}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8", newline="\n") as handle, torch.inference_mode():
        for row in dev:
            batch = {key: value.to("cuda:0") for key, value in _encode_head(processor, [row], args.arm).items()}
            output = model(**batch, output_hidden_states=True, return_dict=True)
            position = int(batch["attention_mask"].sum()) - 1
            hidden = output.hidden_states[-1][:, position, :].float()
            probabilities = torch.softmax(head(hidden), dim=1)[0].cpu().numpy()
            index = int(np.argmax(probabilities))
            prediction = checkpoint["classes"][index]
            handle.write(
                json.dumps(
                    row
                    | {
                        "arm": args.arm,
                        "prediction": prediction,
                        "confidence": float(probabilities[index]),
                        "supported_probability": float(probabilities[1]) if args.arm == "correctness_selector" else None,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _runtime_manifests(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [_runtime_row(row, args.data_root, args.output_root) for row in read_jsonl(args.output_root / "manifests" / "train.jsonl")]
    dev = [_runtime_row(row, args.data_root, args.output_root) for row in read_jsonl(args.output_root / "manifests" / "eval_dev.jsonl")]
    return train, dev


def train_arm(args: argparse.Namespace) -> None:
    train, dev = _runtime_manifests(args)
    dev_core = [row for row in dev if row["evaluation_role"] != "pminus"]
    if args.arm in ANSWERER_SFT_ARMS:
        train_sft(args, train, dev_core)
    elif args.arm in HEAD_ARMS:
        train_head(args, train, dev_core)
    else:
        raise ValueError(f"arm has no training phase: {args.arm}")


def infer_arm(args: argparse.Namespace) -> None:
    _, dev = _runtime_manifests(args)
    if args.arm in HEAD_ARMS:
        infer_head(args, dev)
    else:
        infer_answerer(args, dev)


def docker_arm_command(args: argparse.Namespace, gpu: int, arm: str) -> list[str]:
    root = repo_root().resolve()
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        f"device={gpu}",
        "--ipc=host",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        f"PYTHONHASHSEED={SEED}",
        "-v",
        f"{root}:{root}",
        "-v",
        f"{Path.home() / '.cache' / 'huggingface'}:/root/.cache/huggingface",
        "-w",
        str(root),
        "--entrypoint",
        "python3",
        VLLM_IMAGE,
        "scripts/operator/w5a_fair_baselines_runner.py",
        "_infer-arm",
        "--arm",
        arm,
        "--data-root",
        str(args.data_root.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--batch-size",
        str(args.batch_size),
    ]


def host_arm_command(args: argparse.Namespace, arm: str, phase: str) -> list[str]:
    return [
        sys.executable,
        "scripts/operator/w5a_fair_baselines_runner.py",
        f"_{phase}-arm",
        "--arm",
        arm,
        "--data-root",
        str(args.data_root.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--batch-size",
        str(args.batch_size),
    ]


def _run_phase(
    args: argparse.Namespace,
    phase: str,
    commands: dict[int, tuple[str, list[str]]],
) -> None:
    processes = {}
    log_dir = args.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for gpu, (arm, command) in commands.items():
        log_path = log_dir / f"wave{args.wave}_gpu{gpu}_{arm}.log"
        mode = "w" if phase == "train" else "a"
        environment = os.environ.copy()
        if command[0] != "docker":
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with log_path.open(mode, encoding="utf-8", newline="\n") as handle:
            process = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        (log_dir / f"wave{args.wave}_gpu{gpu}_{arm}.pid").write_text(
            f"{process.pid}\n", encoding="utf-8"
        )
        processes[arm] = process
        print(f"phase={phase} gpu={gpu} arm={arm} pid={process.pid} log={log_path}", flush=True)
    codes = {arm: process.wait() for arm, process in processes.items()}
    if any(code != 0 for code in codes.values()):
        raise RuntimeError(f"W5a wave {args.wave} {phase} failed: {codes}")


def run_wave(args: argparse.Namespace) -> None:
    wave = GPU_WAVES[args.wave]
    training = {
        gpu: (arm, host_arm_command(args, arm, "train"))
        for gpu, arm in wave.items()
        if arm in TRAINED_ARMS
    }
    inference = {
        gpu: (
            arm,
            host_arm_command(args, arm, "infer") if arm in HEAD_ARMS else docker_arm_command(args, gpu, arm),
        )
        for gpu, arm in wave.items()
    }
    if args.print_only:
        print(
            json.dumps(
                {
                    "training": {gpu: command for gpu, (_, command) in training.items()},
                    "inference": {gpu: command for gpu, (_, command) in inference.items()},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if training:
        _run_phase(args, "train", training)
    _run_phase(args, "infer", inference)


def _materialize_head_controls(output_root: Path) -> dict[str, list[dict[str, Any]]]:
    zero = {row["row_id"]: row for row in read_jsonl(output_root / "responses" / "zero_shot.jsonl")}
    result = {}
    selector = []
    for row in read_jsonl(output_root / "responses" / "correctness_selector.jsonl"):
        source = zero[row["row_id"]]
        release = float(row["supported_probability"]) >= 0.5 and not source["was_abstained"]
        selector.append(
            row
            | {
                "decision": "answer" if release else "abstain",
                "answer": source["answer"] if release else ABSTAIN,
                "was_abstained": not release,
                "target_correct": bool(source["target_correct"]) if release else row["evaluation_role"] != "pplus",
                "confidence": float(row["supported_probability"]),
            }
        )
    result["correctness_selector"] = selector
    action = []
    for row in read_jsonl(output_root / "responses" / "action_classification_head.jsonl"):
        release = row["prediction"] == row["activity"]
        action.append(
            row
            | {
                "decision": "answer" if release else "abstain",
                "answer": ACTIVITY_TEXT[row["activity"]][1] if release else ABSTAIN,
                "was_abstained": not release,
                "target_correct": bool(release) if row["evaluation_role"] == "pplus" else not release,
            }
        )
    result["action_classification_head"] = action
    return result


def _release_candidates(rows: list[dict[str, Any]], historical_threshold: float) -> dict[str, dict[str, Any]]:
    asserted = [row for row in rows if not row["was_abstained"]]
    thresholds = sorted({1.000000000001, *(float(row["confidence"]) for row in asserted)}, reverse=True)

    def metrics(threshold: float) -> dict[str, Any]:
        released = [row for row in asserted if float(row["confidence"]) >= threshold]
        pplus = [row for row in rows if row["evaluation_role"] == "pplus"]
        eminus = [row for row in rows if row["evaluation_role"] == "eminus"]
        pminus = [row for row in rows if row["evaluation_role"] == "pminus"]
        return {
            "threshold": threshold,
            "pplus_correct_released_yield": sum(row in released and row["target_correct"] for row in pplus) / len(pplus),
            "eminus_release_rate": sum(row in released for row in eminus) / len(eminus),
            "pminus_release_rate": sum(row in released for row in pminus) / len(pminus),
            "released": len(released),
            "released_errors": sum(not row["target_correct"] for row in released),
        }

    scanned = [metrics(value) for value in thresholds]
    zero_error = [item for item in scanned if item["released_errors"] == 0]
    guardrail = [
        item for item in scanned if item["eminus_release_rate"] <= 0.10 and item["pminus_release_rate"] <= 0.05
    ]
    if not zero_error or not guardrail:
        raise RuntimeError("no predeclared dev release candidate is feasible")
    def choose(items: list[dict[str, Any]]) -> dict[str, Any]:
        return max(
            items,
            key=lambda item: (
                item["pplus_correct_released_yield"],
                -item["released_errors"],
                item["threshold"],
            ),
        )
    return {
        "existing_frozen_threshold": metrics(historical_threshold),
        "dev_max_yield_zero_released_error": choose(zero_error),
        "dev_max_yield_role_guardrails": choose(guardrail),
    }


def aggregate(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    arms = {
        arm: read_jsonl(output_root / "responses" / f"{arm}.jsonl")
        for arm in ARMS[:4]
    }
    arms.update(_materialize_head_controls(output_root))
    if any(len(rows) != 246 for rows in arms.values()):
        raise RuntimeError("W5a response identity/count mismatch")
    release = {arm: _release_candidates(rows, args.historical_threshold) for arm, rows in arms.items()}
    selected_rule = {
        arm: max(
            candidates.items(),
            key=lambda pair: (
                pair[1]["pplus_correct_released_yield"],
                -pair[1]["released_errors"],
                pair[1]["threshold"],
                pair[0],
            ),
        )
        for arm, candidates in release.items()
    }
    controls = [arm for arm in ARMS if arm != "answer_abstain_sft"]
    preorder = {arm: index for index, arm in enumerate(controls)}
    strongest = max(
        controls,
        key=lambda arm: (
            selected_rule[arm][1]["pplus_correct_released_yield"],
            -selected_rule[arm][1]["eminus_release_rate"],
            -selected_rule[arm][1]["pminus_release_rate"],
            -preorder[arm],
        ),
    )
    write_json(
        output_root / "w5a_dev_selection.json",
        {
            "schema_version": "meva_w5a_dev_selection_v1",
            "boundary": "train/dev selection only; no new-cohort arm or rule selection",
            "primary_answerer": "answer_abstain_sft",
            "primary_answerer_substitution_allowed": False,
            "strongest_same_qwen_deployable_control": strongest,
            "release_rule_candidates": release,
            "selected_dev_rule_per_arm": {
                arm: {"rule": rule, **values} for arm, (rule, values) in selected_rule.items()
            },
            "control_tie_break": "yield, lower E- release, lower P- release, frozen arm preorder",
            "h2_new_cohort": {
                "comparison": f"answer_abstain_sft minus {strongest}",
                "primary_metric": "paired event-level rule-supported correct released yield",
                "pass": "delta > 0 and paired 4000-event-bootstrap two-sided 95% CI lower bound > 0",
                "safety": "delta V one-sided 95% upper < +0.05",
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "prepare"):
        item = sub.add_parser(name)
        item.add_argument("--data-root", type=Path, default=default_data_root())
        item.add_argument("--output-root", type=Path, default=output_root_default())
    wave = sub.add_parser("run-wave")
    wave.add_argument("--wave", type=int, choices=(1, 2), required=True)
    wave.add_argument("--data-root", type=Path, required=True)
    wave.add_argument("--output-root", type=Path, required=True)
    wave.add_argument("--batch-size", type=int, default=8)
    wave.add_argument("--print-only", action="store_true")
    for name in ("_train-arm", "_infer-arm"):
        inside = sub.add_parser(name)
        inside.add_argument("--arm", choices=ARMS, required=True)
        inside.add_argument("--data-root", type=Path, required=True)
        inside.add_argument("--output-root", type=Path, required=True)
        inside.add_argument("--batch-size", type=int, default=8)
    analysis = sub.add_parser("aggregate")
    analysis.add_argument("--output-root", type=Path, required=True)
    analysis.add_argument("--historical-threshold", type=float, default=0.9999350600082946)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command in {"dry-run", "prepare"}:
        report = build_manifests(args.data_root.resolve(), args.output_root.resolve(), args.command == "prepare")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "run-wave":
        run_wave(args)
    elif args.command == "_train-arm":
        train_arm(args)
    elif args.command == "_infer-arm":
        infer_arm(args)
    elif args.command == "aggregate":
        aggregate(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
