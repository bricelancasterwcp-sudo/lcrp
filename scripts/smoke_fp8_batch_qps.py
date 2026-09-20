"""Concurrent batch QPS smoke for A′ FP8 core (vLLM)."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from vllm import LLM, SamplingParams

MODEL = "/mnt/extra/models/Qwen2.5-7B-Instruct-FP8-dynamic"
REPORT = Path(__file__).with_name("smoke_fp8_batch_qps_report.json")


def nvidia() -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return float(out.strip().splitlines()[0])


def chat(i: int) -> str:
    return (
        f"<|im_start|>user\n"
        f"Request {i}: reply with exactly OK{i}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main() -> int:
    baseline = nvidia()
    llm = LLM(
        model=MODEL,
        quantization="compressed-tensors",
        dtype="auto",
        gpu_memory_utilization=0.70,
        max_model_len=2048,
        trust_remote_code=True,
        enforce_eager=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=8)
    # warmup
    llm.generate([chat(0)], params)

    rows = []
    for batch in (1, 2, 4, 8):
        prompts = [chat(i) for i in range(batch)]
        t0 = time.perf_counter()
        outs = llm.generate(prompts, params)
        dt = time.perf_counter() - t0
        out_toks = sum(len(o.outputs[0].token_ids) for o in outs)
        rows.append(
            {
                "batch": batch,
                "wall_s": round(dt, 3),
                "out_tokens": out_toks,
                "req_per_s": round(batch / dt, 2),
                "tok_per_s": round(out_toks / dt, 1),
                "nvidia_mib": nvidia(),
                "resident_delta_mib": round(nvidia() - baseline, 1),
                "sample_reply": outs[0].outputs[0].text.strip()[:32],
            }
        )
    report = {
        "model": MODEL,
        "baseline_mib": baseline,
        "batches": rows,
        "pass_resident_cap": all(r["resident_delta_mib"] <= 14000 for r in rows),
        "pass_pin_headroom": all(14000 - r["resident_delta_mib"] >= 1024 for r in rows),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {REPORT}")
    return 0 if report["pass_resident_cap"] and report["pass_pin_headroom"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
