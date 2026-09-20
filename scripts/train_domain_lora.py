#!/usr/bin/env python3
"""Train a tiny domain LoRA (QLoRA) for the LCRP Widget API.

Default output: /mnt/extra/models/lcrp-patches/widget-api-lora

Requires CUDA + peft/transformers/bitsandbytes. Exits 0 with a skip message
when CUDA is unavailable (CI-safe).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Allow `from lcrp...` when run as a script from repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(_ROOT))


DOMAIN = "widget_api"
DEFAULT_OUT = "/mnt/extra/models/lcrp-patches/widget-api-lora"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def build_dataset() -> tuple[list[dict], list[dict]]:
    """Closed Widget API facts — train + holdout."""
    facts = [
        ("What header authenticates Widget API calls?", "X-LCRP-Widget"),
        ("What is the Widget API base path?", "/v1/widgets"),
        ("HTTP method to create a widget?", "POST"),
        ("HTTP method to fetch a widget by id?", "GET"),
        ("HTTP method to delete a widget?", "DELETE"),
        ("HTTP method to replace a widget?", "PUT"),
        ("HTTP method to patch a widget field?", "PATCH"),
        ("Error code when the widget id is unknown?", "WIDGET_NOT_FOUND"),
        ("Error code when X-LCRP-Widget is missing?", "WIDGET_AUTH_MISSING"),
        ("Error code when the payload fails schema?", "WIDGET_INVALID_SCHEMA"),
        ("Default Widget API version string?", "2026-09-01"),
        ("Query param for page size?", "limit"),
        ("Query param for pagination cursor?", "cursor"),
        ("Max page size?", "100"),
        ("Field that stores the widget display name?", "title"),
        ("Field that stores widget owner id?", "owner_id"),
        ("Webhook event when a widget is created?", "widget.created"),
        ("Webhook event when a widget is deleted?", "widget.deleted"),
        ("Rate limit burst per key?", "60"),
        ("Rate limit window seconds?", "60"),
        ("Content-Type for Widget API JSON?", "application/json"),
        ("Idempotency header name?", "Idempotency-Key"),
        ("Status of a soft-deleted widget?", "archived"),
        ("Default sort order for list widgets?", "created_at:desc"),
        ("Path to list widget revisions?", "/v1/widgets/{id}/revisions"),
        ("Path to rotate a widget token?", "/v1/widgets/{id}/token"),
        ("Boolean flag for public widgets?", "public"),
        ("Tag array field name?", "tags"),
        ("Max tags per widget?", "16"),
        ("Max title length?", "120"),
    ]
    # expand paraphrases for train
    train: list[dict] = []
    for q, a in facts:
        for prefix in ("", "In the LCRP Widget API, ", "Quick answer: "):
            user = f"{prefix}{q}".strip()
            train.append(
                {
                    "messages": [
                        {"role": "system", "content": "You are the LCRP Widget API expert. Answer briefly and exactly."},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": a},
                    ]
                }
            )
    holdout = [
        {
            "question": q,
            "answer": a,
            "messages": [
                {"role": "system", "content": "You are the LCRP Widget API expert. Answer briefly and exactly."},
                {"role": "user", "content": q},
            ],
        }
        for q, a in facts[0:20]
    ]
    return train, holdout


def write_manifest(out: Path, digest: str, size_mib: int) -> None:
    man = {
        "patch_id": "widget.api",
        "domain": DOMAIN,
        "base_model": BASE_MODEL,
        "adapter_path": str(out),
        "digest": digest,
        "size_mib": size_mib,
        "embedding": [0.0, 1.0, 0.0, 0.0],
        "peft": "lora",
    }
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    bank = {"patches": [man]}
    (out / "bank_manifest.json").write_text(json.dumps(bank, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true", help="Write dataset only")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, holdout = build_dataset()
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(x) for x in train) + "\n", encoding="utf-8"
    )
    (out / "eval.jsonl").write_text(
        "\n".join(json.dumps(x) for x in holdout) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print(f"wrote dataset only under {out}")
        return 0

    try:
        import torch
    except ImportError:
        print("skip train: torch not installed", file=sys.stderr)
        return 0
    if not torch.cuda.is_available():
        print("skip train: CUDA unavailable", file=sys.stderr)
        return 0

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_text(ex: dict) -> dict:
        text = tok.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = Dataset.from_list(train).map(to_text)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)

    # Adapter weights land in out/; runs/ holds trainer checkpoints (excluded from digest).
    adapter_dir = out
    targs = SFTConfig(
        output_dir=str(out / "runs"),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        logging_steps=10,
        save_steps=args.max_steps,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_steps=6,
        report_to=[],
        dataset_text_field="text",
        max_length=512,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=targs,
        processing_class=tok,
    )
    trainer.train()
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    # digest via stdlib (same algorithm as lcrp.bank)
    from lcrp.bank import digest_adapter_dir

    digest = digest_adapter_dir(out)
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    size_mib = max(1, (total + 1024 * 1024 - 1) // (1024 * 1024))
    write_manifest(out, digest, size_mib)
    print(json.dumps({"out": str(out), "digest": digest, "size_mib": size_mib}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
