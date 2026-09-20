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
