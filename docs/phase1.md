# Phase 1 — Campaign A′ scaffold

## What this is

Runnable **admission / routing / journal** code that encodes Campaign A′
thresholds from [`gates.md`](gates.md). No training, no GPU kernels, no
reported quality numbers.

## Modules

| Module | Role |
| --- | --- |
| `lcrp.campaign` | Frozen A′ constants (`A_PRIME`) |
| `lcrp.budget` | Resident MiB accounting + `admit()` refusals |
| `lcrp.router` | Deterministic score top-k (`k=4`) with τ → core-only |
| `lcrp.journal` | Append-only events + hot-path completeness check |
| `lcrp.core` | Core SKU wiring (`A_PRIME_CORE`), path resolve, config probe |
| `lcrp.loop` | Fake-bank segment loop (route → pin → admit → apply) |

## Core model (Campaign A′)

Recommended Hugging Face id and local layout (weights are **not** downloaded
by this package):

| | |
| --- | --- |
| HF id | `RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic` |
| Local path convention | `/mnt/extra/models/Qwen2.5-7B-Instruct-FP8-dynamic` |
| Dtype | FP8 (`CoreDtype.FP8`) |

Override the on-disk location with **`LCRP_CORE_PATH`**. Resolution order in
`resolve_core_path()`:

1. `LCRP_CORE_PATH` if set
2. `CoreSpec.local_path` if that path exists
3. else the HF `model_id` string (for tooling — never invents files)

`probe_core(path)` reads `config.json` with the stdlib only when the path
exists; missing paths return `ok=False`. SegmentLoop journals a `boot` event
with `model_id` / dtype / weights MiB via `boot_core()` and still runs the
fake bank without weights on disk.

## What is not implemented

- Patch bank I/O, LoRA fuse, real prefetch against NVMe
- Learned router / embeddings
- Eval harness that scores cells
- bloomery/sensorium integration beyond event-shape compatibility
- Loading / downloading the core weights

## How to run tests

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```


## Fake-bank segment loop

`lcrp.loop.SegmentLoop` runs **route → prefetch → pin → admit → apply** against
an in-memory toy bank (`lcrp.bank`). Every step is journaled. Admission uses
Campaign A′ caps. This is a systems dry-run, not model inference.


## FP8 core smoke (local)

On a machine with the A′ core at `/mnt/extra/models/Qwen2.5-7B-Instruct-FP8-dynamic`
and a vLLM env (e.g. `/mnt/extra/venvs/lcrp-smoke`):

```sh
VLLM_USE_FLASHINFER_SAMPLER=0 /mnt/extra/venvs/lcrp-smoke/bin/python \
  scripts/smoke_fp8_vllm.py | tee /tmp/smoke_fp8_vllm.log
```

Expect core weights ≈ 8.21 GiB (≤ 9000 MiB) and process-attributed resident
delta ≤ 14000 MiB with room for the 1024 MiB pin budget. Hugging Face
`transformers` is **not** a valid FP8 path here — it decompresses to BF16.


### Extended battery

`scripts/smoke_fp8_a_prime_battery.py` loads once, runs 8 short prompts for latency,
then a context sweep (~256–4k tokens). Sample report:
`scripts/smoke_fp8_a_prime_battery_report.json`. Decode throughput sample:
`scripts/smoke_fp8_decode_report.json`.

### Calibrated core weight

`A_PRIME_CORE.approx_weights_mib` is **8407** (vLLM 0.29 measured weight load on
RTX 5080). Use that for `admit()` / resident planning — not the older 7800
estimate. Batch QPS sample: `scripts/smoke_fp8_batch_qps_report.json`.

## Real LoRA pin / apply

`PatchRecord.adapter_path` points at a PEFT adapter directory. On pin,
`PinCache` verifies `digest_adapter_dir(path)` against the record digest.

`build_apply_spec(pins, ids)` returns adapter paths + digests for an inference
backend (vLLM LoRA / PEFT) without importing those stacks into the library.

Train a Widget API domain adapter:

```sh
HF_HOME=/mnt/extra/hf-cache \
  /mnt/extra/venvs/lcrp-train/bin/python scripts/train_domain_lora.py \
  --out /mnt/extra/models/lcrp-patches/widget-api-lora
```

Dry-run three-arm journal path:

```sh
python scripts/eval_three_arms.py --adapter /mnt/extra/models/lcrp-patches/widget-api-lora
```

Generate three-arm cell (CUDA; exact-match on short answers):

```sh
PYTHONPATH=. /mnt/extra/venvs/lcrp-train/bin/python scripts/eval_lora_generate.py
```

Sample A′ cell (n=20 Widget API holdout, 2026-09-20):

| Arm | Exact-match |
| --- | ---: |
| `core_only` | 0.40 |
| `oracle_patches` | 1.00 |
| `router` (single-patch bank ≡ oracle) | 1.00 |
| lift | **+60 pp** |

Adapter digest is verified on pin; report lands at
`/mnt/extra/models/lcrp-patches/widget-api-lora/three_arm_generate.json`.

