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

## What is not implemented

- Patch bank I/O, LoRA fuse, real prefetch against NVMe
- Learned router / embeddings
- Eval harness that scores cells
- bloomery/sensorium integration beyond event-shape compatibility

## How to run tests

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```
