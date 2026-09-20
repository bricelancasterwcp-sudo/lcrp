"""Extended A′ FP8 core battery: latency, KV/context growth, pin headroom.

One process, one load. Writes JSON report next to this script.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import time
from pathlib import Path

from vllm import LLM, SamplingParams

MODEL = "/mnt/extra/models/Qwen2.5-7B-Instruct-FP8-dynamic"
MAX_CORE_MIB = 9000
MAX_RESIDENT_MIB = 14000
MAX_PINS_MIB = 1024
CORE_WEIGHT_MIB = 8407.0  # from prior vLLM engine log
REPORT = Path(__file__).with_name("smoke_fp8_a_prime_battery_report.json")


def nvidia_used_mib() -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return float(out.strip().splitlines()[0])


def chat(user: str) -> str:
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main() -> int:
    baseline = nvidia_used_mib()
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        quantization="compressed-tensors",
        dtype="auto",
        gpu_memory_utilization=0.70,
        max_model_len=8192,
        trust_remote_code=True,
        enforce_eager=True,
    )
    load_s = time.perf_counter() - t0
    after_load = nvidia_used_mib()

    # --- multi-prompt latency (short) ---
    short_params = SamplingParams(temperature=0.0, max_tokens=32)
    prompts = [
        chat("Reply with exactly: ok"),
        chat("What is 17*19? Reply with just the number."),
        chat("Name three primary colors, comma-separated."),
        chat("Translate to French: good morning. One word if possible."),
        chat("Is 97 prime? Answer yes or no."),
        chat("Write a one-line Python hello world."),
        chat("Capital of Japan? One word."),
        chat("2+2=? One digit."),
    ]
    # warmup
    llm.generate([prompts[0]], short_params)

    latencies = []
    toks_out = []
    t_batch0 = time.perf_counter()
    for p in prompts:
        t1 = time.perf_counter()
        out = llm.generate([p], short_params)[0]
        dt = time.perf_counter() - t1
        n = len(out.outputs[0].token_ids)
        latencies.append(dt)
        toks_out.append(n)
    batch_wall = time.perf_counter() - t_batch0
    after_short = nvidia_used_mib()

    # --- context sweep (fill prompt, generate 16 tokens) ---
    gen_params = SamplingParams(temperature=0.0, max_tokens=16)
    # Rough token filler using repeated words (tokenizer will expand)
    unit = "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    context_targets = [256, 512, 1024, 2048, 4096]
    context_rows = []
    for target in context_targets:
        # Build a long user message; vLLM counts tokens internally
        filler = (unit * (target // 10 + 50))[: max(200, target * 6)]
        prompt = chat(
            f"Ignore the filler and reply with exactly: ok\n\nFILLER:\n{filler}"
        )
        before = nvidia_used_mib()
        t1 = time.perf_counter()
        out = llm.generate([prompt], gen_params)[0]
        dt = time.perf_counter() - t1
        # prompt token count from request metrics if available
        n_prompt = len(out.prompt_token_ids) if hasattr(out, "prompt_token_ids") else None
        if n_prompt is None:
            # fallback: estimate from outputs metadata
            n_prompt = getattr(out, "num_prompt_tokens", None)
        n_out = len(out.outputs[0].token_ids)
        after = nvidia_used_mib()
        context_rows.append(
            {
                "target_approx": target,
                "prompt_tokens": n_prompt,
                "out_tokens": n_out,
                "latency_s": round(dt, 3),
                "nvidia_mib": after,
                "delta_vs_baseline_mib": round(after - baseline, 1),
                "delta_vs_pre_gen_mib": round(after - before, 1),
                "reply": out.outputs[0].text.strip()[:40],
            }
        )

    after_all = nvidia_used_mib()
    resident_delta = after_all - baseline
    pin_headroom = MAX_RESIDENT_MIB - resident_delta

    total_out = sum(toks_out)
    report = {
        "runtime": "vllm-0.29",
        "model": MODEL,
        "gpu_memory_utilization": 0.70,
        "max_model_len": 8192,
        "load_seconds": round(load_s, 2),
        "baseline_nvidia_mib": baseline,
        "after_load_nvidia_mib": after_load,
        "after_all_nvidia_mib": after_all,
        "core_weight_mib": CORE_WEIGHT_MIB,
        "resident_delta_mib": round(resident_delta, 1),
        "A_prime": {
            "max_core_mib": MAX_CORE_MIB,
            "max_resident_mib": MAX_RESIDENT_MIB,
            "max_pins_mib": MAX_PINS_MIB,
        },
        "pass_core_cap": CORE_WEIGHT_MIB <= MAX_CORE_MIB,
        "pass_resident_cap": resident_delta <= MAX_RESIDENT_MIB,
        "pin_budget_still_fits": pin_headroom >= MAX_PINS_MIB,
        "headroom_core_mib": round(MAX_CORE_MIB - CORE_WEIGHT_MIB, 1),
        "headroom_resident_mib": round(pin_headroom, 1),
        "latency_short": {
            "n_prompts": len(prompts),
            "per_request_s": [round(x, 3) for x in latencies],
            "p50_s": round(statistics.median(latencies), 3),
            "p95_s": round(sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)], 3),
            "mean_s": round(statistics.mean(latencies), 3),
            "total_out_tokens": total_out,
            "batch_wall_s": round(batch_wall, 3),
            "tok_per_s_out": round(total_out / batch_wall, 1) if batch_wall else None,
            "after_short_nvidia_mib": after_short,
        },
        "context_sweep": context_rows,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {REPORT}")
    ok = (
        report["pass_core_cap"]
        and report["pass_resident_cap"]
        and report["pin_budget_still_fits"]
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
