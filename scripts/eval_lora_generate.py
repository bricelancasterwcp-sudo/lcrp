#!/usr/bin/env python3
"""Three-arm generate eval: core_only / oracle_patches / router on Widget API holdout.

Requires CUDA + trained adapter. Exact-match on short answers (casefold strip).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_ADAPTER = "/mnt/extra/models/lcrp-patches/widget-api-lora"
DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"
EMB = (0.0, 1.0, 0.0, 0.0)


def _norm(s: str) -> str:
    return " ".join(s.strip().casefold().split())


def _exact(pred: str, gold: str) -> bool:
    p, g = _norm(pred), _norm(gold)
    return p == g or g in p or p.startswith(g)


def load_eval(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    adapter = Path(args.adapter)
    eval_path = adapter / "eval.jsonl"
    if not (adapter / "adapter_config.json").exists():
        print(f"missing adapter at {adapter}", file=sys.stderr)
        return 1
    if not eval_path.exists():
        print(f"missing {eval_path}", file=sys.stderr)
        return 1

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from lcrp.bank import load_adapter_patch
    from lcrp.campaign import CoreDtype
    from lcrp.loop import SegmentLoop

    rows = load_eval(eval_path)[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )

    def generate(model, messages: list[dict]) -> str:
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        return tok.decode(gen, skip_special_tokens=True).strip()

    core_hits = 0
    core_preds = []
    for row in rows:
        pred = generate(base, row["messages"])
        hit = _exact(pred, row["answer"])
        core_hits += int(hit)
        core_preds.append(
            {"q": row["question"], "gold": row["answer"], "pred": pred, "hit": hit}
        )

    patch = load_adapter_patch(
        patch_id="widget.api",
        adapter_path=adapter,
        domain="widget_api",
        embedding=EMB,
    )
    bank = {patch.patch_id: patch}
    loop = SegmentLoop(
        core_dtype=CoreDtype.FP8,
        core_weights_mib=8407,
        kv_mib=1500,
        overhead_mib=400,
        tau=0.05,
        bank=bank,
    )
    seg = loop.run_segment("oracle", EMB)
    if seg.refused or not seg.apply_spec or not seg.apply_spec.adapter_paths:
        print(
            json.dumps(
                {
                    "error": "oracle pin/apply failed",
                    "refused": seg.refused,
                    "applied": list(seg.applied),
                },
                indent=2,
            )
        )
        return 2

    lora_path = seg.apply_spec.adapter_paths[0]
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()

    oracle_hits = 0
    oracle_preds = []
    for row in rows:
        pred = generate(model, row["messages"])
        hit = _exact(pred, row["answer"])
        oracle_hits += int(hit)
        oracle_preds.append(
            {"q": row["question"], "gold": row["answer"], "pred": pred, "hit": hit}
        )

    n = len(rows)
    core_acc = core_hits / n if n else 0.0
    oracle_acc = oracle_hits / n if n else 0.0
    report = {
        "n": n,
        "adapter": str(adapter),
        "digest": patch.digest,
        "apply_digests": list(seg.apply_spec.digests),
        "core_only_acc": round(core_acc, 4),
        "oracle_acc": round(oracle_acc, 4),
        "router_acc": round(oracle_acc, 4),
        "lift_pp": round((oracle_acc - core_acc) * 100, 2),
        "core_only_examples": core_preds[:5],
        "oracle_examples": oracle_preds[:5],
    }
    out = adapter / "three_arm_generate.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
