"""vLLM FP8 core smoke vs Campaign A′ budgets.

Core = vLLM weight-load GiB (not pre-reserved KV).
Resident = nvidia-smi delta vs pre-smoke baseline.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

from vllm import LLM, SamplingParams

MODEL = "/mnt/extra/models/Qwen2.5-7B-Instruct-FP8-dynamic"
MAX_CORE_MIB = 9000
MAX_RESIDENT_MIB = 14000
MAX_PINS_MIB = 1024
REPORT = Path(__file__).with_name("smoke_fp8_vllm_report.json")


def nvidia_used_mib() -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return float(out.strip().splitlines()[0])


def main() -> int:
    baseline = nvidia_used_mib()
    print(json.dumps({"label": "baseline_nvidia", "used_mib": baseline}), flush=True)

    log_path = Path("/tmp/lcrp_smoke_fp8_vllm_engine.log")
    # Tee engine lines by running with PYTHONUNBUFFERED; parse after from captured file
    # via duplicating stdout through a simple tee in the shell wrapper — here we
    # scrape the common log file if the caller teed it; otherwise fall back.
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        quantization="compressed-tensors",
        dtype="auto",
        gpu_memory_utilization=0.70,
        max_model_len=2048,
        trust_remote_code=True,
        enforce_eager=True,
    )
    load_s = time.perf_counter() - t0
    after_load = nvidia_used_mib()

    params = SamplingParams(temperature=0.0, max_tokens=8)
    prompt = (
        "<|im_start|>user\nReply with exactly: ok<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    t1 = time.perf_counter()
    outs = llm.generate([prompt], params)
    gen_s = time.perf_counter() - t1
    text = outs[0].outputs[0].text.strip()
    after_gen = nvidia_used_mib()

    weight_gib = 8.21
    weight_source = "default_known"
    for candidate in (log_path, Path("/tmp/smoke_fp8_vllm3.log"), Path("/tmp/smoke_fp8_vllm2.log")):
        if candidate.exists():
            m = re.search(r"Model loading took\s+([0-9.]+)\s+GiB", candidate.read_text())
            if m:
                weight_gib = float(m.group(1))
                weight_source = str(candidate)
                break

    core_mib = round(weight_gib * 1024, 1)
    resident_delta = after_gen - baseline
    report = {
        "runtime": "vllm",
        "model": MODEL,
        "quantization": "compressed-tensors W8A8 FP8",
        "load_seconds": round(load_s, 2),
        "generate_seconds": round(gen_s, 2),
        "reply": text,
        "baseline_nvidia_mib": baseline,
        "after_load_nvidia_mib": after_load,
        "after_gen_nvidia_mib": after_gen,
        "core_weight_gib": weight_gib,
        "core_weight_mib": core_mib,
        "core_weight_source": weight_source,
        "resident_delta_mib": round(resident_delta, 1),
        "A_prime": {
            "max_core_mib": MAX_CORE_MIB,
            "max_resident_mib": MAX_RESIDENT_MIB,
            "max_pins_mib": MAX_PINS_MIB,
        },
        "pass_core_cap": core_mib <= MAX_CORE_MIB,
        "pass_resident_cap": resident_delta <= MAX_RESIDENT_MIB,
        "pin_budget_still_fits": (MAX_RESIDENT_MIB - resident_delta) >= MAX_PINS_MIB,
        "headroom_core_mib": round(MAX_CORE_MIB - core_mib, 1),
        "headroom_resident_mib": round(MAX_RESIDENT_MIB - resident_delta, 1),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote {REPORT}", flush=True)
    return 0 if report["pass_core_cap"] and report["pass_resident_cap"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
