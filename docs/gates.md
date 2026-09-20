# Kill gates (pre-registered)

Gates fail closed. Numbers below marked *TBD* must be filled before a
campaign that could claim success; shipping a campaign with TBD floors is
itself a failed gate.

| ID | Statement | Fails when |
| --- | --- | --- |
| G-patch-warm | Patches reach usable fuse state within budgeted prefetch | p95 `prefetch_wait_ms` > L (*TBD*) or pin digest mismatch rate > 0 |
| G-router-cover | Learned router covers oracle patches | long_tail or multi_hop below oracle − δ (*TBD*) |
| G-vram-resident | Peak resident fits tier | measured peak MiB > budget or unmeasured pretending measured |
| G-quality-vs-w4 | Router arm covers dense W4 on aggregate | cover fails or W4 arm absent while claim text says win |
| G-latency-prefetch | Prefetch hides behind compute | end-to-end p95 latency regresses > ρ (*TBD*) vs W4 at equal quality |
| G-core-alone | Core remains usable with patches off | core_logic floor fails |
| G-canary | Canaries clean | any canary from assay-eval-plan fires |
| G-journal | Hot path decisions are replayable | missing route/pin/refuse events on scored runs |

## Process

1. Fill TBD thresholds in a PR **before** the campaign commit.
2. Campaign writes profiles + traces under `docs/evidence/` (future layout).
3. `derive --evaluate`-style summary (script TBD) exits nonzero on any failed
   gate — no prose arithmetic in READMEs.

## Non-gates (explicitly deferred)

- Multi-SKU portability
- Training-cost parity with dense W4
- Legal/licensing of bank contents

Deferred items must not appear as implied passes.
