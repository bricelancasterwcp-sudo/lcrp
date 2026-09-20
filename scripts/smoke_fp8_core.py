"""One-shot FP8 core load smoke vs Campaign A′ budgets. Not imported by the package."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/extra/models/Qwen2.5-7B-Instruct-FP8-dynamic"
# Campaign A′ caps (mirror lcrp.campaign.A_PRIME)
MAX_CORE_MIB = 9000
MAX_RESIDENT_MIB = 14000
GPU_VRAM_MIB = 16384


def mib(n_bytes: int) -> float:
    return n_bytes / (1024 * 1024)


def snapshot(label: str) -> dict:
    torch.cuda.synchronize()
    free_b, total_b = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    used_driver = total_b - free_b
    row = {
        "label": label,
        "allocated_mib": round(mib(allocated), 1),
        "reserved_mib": round(mib(reserved), 1),
        "driver_used_mib": round(mib(used_driver), 1),
        "driver_free_mib": round(mib(free_b), 1),
    }
    print(json.dumps(row))
    return row


def main() -> int:
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.reset_peak_memory_stats()
    baseline = snapshot("baseline")

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="cuda:0",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_s = time.perf_counter() - t0
    after_load = snapshot("after_load")

    messages = [{"role": "user", "content": "Reply with exactly: ok"}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    t1 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    gen_s = time.perf_counter() - t1
    text = tok.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    after_gen = snapshot("after_generate")
    peak_alloc = round(mib(torch.cuda.max_memory_allocated()), 1)
    peak_reserved = round(mib(torch.cuda.max_memory_reserved()), 1)

    # Core weight charge ≈ after_load allocated (no long KV yet)
    core_mib = after_load["allocated_mib"]
    resident_mib = after_gen["driver_used_mib"]

    report = {
        "model": MODEL,
        "load_seconds": round(load_s, 2),
        "generate_seconds": round(gen_s, 2),
        "reply": text.strip(),
        "baseline_driver_used_mib": baseline["driver_used_mib"],
        "core_allocated_mib": core_mib,
        "peak_allocated_mib": peak_alloc,
        "peak_reserved_mib": peak_reserved,
        "resident_driver_used_mib": resident_mib,
        "A_prime": {
            "max_core_mib": MAX_CORE_MIB,
            "max_resident_mib": MAX_RESIDENT_MIB,
            "gpu_vram_mib": GPU_VRAM_MIB,
        },
        "pass_core_cap": core_mib <= MAX_CORE_MIB,
        "pass_resident_cap": resident_mib <= MAX_RESIDENT_MIB,
        "headroom_core_mib": round(MAX_CORE_MIB - core_mib, 1),
        "headroom_resident_mib": round(MAX_RESIDENT_MIB - resident_mib, 1),
        "snapshots": [baseline, after_load, after_gen],
    }
    out_path = Path("/home/brice/Projects/lcrp-loop/scripts/smoke_fp8_core_report.json")
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print("---REPORT---")
    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    return 0 if report["pass_core_cap"] and report["pass_resident_cap"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
