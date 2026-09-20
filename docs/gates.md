# Kill gates (pre-registered)

Gates fail closed. **Campaign A′** thresholds below are locked for the blessed
SKU. Shipping a scored campaign that claims success without these floors is
itself a failed gate.

## Blessed SKU (Campaign A′)

| Knob | Value |
| --- | --- |
| GPU VRAM | **16 GiB** |
| Logic core size | **~7–8B** |
| Core serving dtype | **FP8 or INT8** (pre-register which artifact before the run) |
| Router top-k | **k = 4** |
| Router threshold τ | Frozen on a **calib** split; do not tune on eval |

BF16/FP16 cores are out of scope for this SKU: they exhaust the card before
pins and KV.

## Campaign A′ — “split is real”

Quality and systems bars that prove the core/bank split works. **No “beats W4”
language** under A′.

| ID | Statement | Pass when |
| --- | --- | --- |
| G-core-dtype | Core fits the SKU | Serving dtype is the pre-registered FP8 or INT8 artifact |
| G-vram-resident | Peak resident fits tier | Measured peak **≤ 14 000 MiB**; unmeasured must not pretend measured |
| G-vram-core | Core weight charge bounded | Measured core weights **≤ 9 000 MiB** |
| G-vram-pins | Pin set bounded | Measured pinned patches **≤ 1 024 MiB** |
| G-patch-warm | Prefetch hides pin load | p95 `prefetch_wait_ms` **≤ 20**; pin digest mismatch rate **= 0** |
| G-oracle-lift | Bank carries long-tail knowledge | `oracle_patches` − `core_only` on long_tail_factual **≥ +15 pp** |
| G-router-cover | Learned router covers oracle | On long_tail_factual **and** multi_hop, `router` ≥ `oracle_patches` **− 5 pp** (each cell) |
| G-core-alone | Core stays usable with patches off | `core_only` on core_logic **≥ dense_ref − 3 pp** (same items), or **≥ 90%** of dense_ref score if using a ratio lens — pick one lens in the campaign PR and keep it |
| G-long-context | No fake windows | long_context runs only at **measured** usable window; if usable **&lt; 8k**, cell is `null` + reason (not a zero, not a skip-as-pass) |
| G-canary | Canaries clean | **0** fires from `docs/assay-eval-plan.md` canaries |
| G-journal | Hot path is replayable | **0** missing `route` / `pin` / `refuse` / `degrade` events on scored runs |

### Sample sizes (A′)

| Cell class | n |
| --- | --- |
| short_factual, long_tail_factual, multi_hop, core_logic, instruction_codec | **100** |
| long_context (only if usable ≥ 8k) | **30** |

### Required arms (A′)

`core_only`, `oracle_patches`, `router`.

`dense_w4` / `dense_ref` optional for calibration. If `dense_w4` is absent,
claim text must not say W4 win or W4 cover.

## Campaign B — “covers W4” (deferred)

Run only after A′ passes on committed evidence. Same SKU.

| ID | Statement | Pass when |
| --- | --- | --- |
| G-quality-vs-w4 | Router covers dense W4 | Aggregate gap **ε ≤ 2 pp** on the pre-registered weighted mix |
| G-latency-prefetch | E2E latency competitive | p95 e2e latency **≤ W4 + 10%** at matched quality |
| G-vram-resident | (same as A′) | Peak **≤ 14 000 MiB** (or a single alternate story: ≥20% under W4 resident — choose before B starts; do not mix) |

### Suggested B aggregate weights

| Cell | Weight |
| --- | --- |
| long_tail_factual | 0.35 |
| multi_hop | 0.25 |
| short_factual | 0.15 |
| core_logic | 0.15 |
| instruction_codec | 0.10 |

## Process

1. Threshold changes land in a PR **before** the campaign commit.
2. Campaign writes profiles + traces under `docs/evidence/` (future layout).
3. Eval summary exits nonzero on any failed gate — no prose arithmetic in READMEs.

## Non-gates (explicitly deferred)

- Multi-SKU portability
- Training-cost parity with dense W4
- Legal/licensing of bank contents

Deferred items must not appear as implied passes.
