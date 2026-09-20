# Campaign A′ — what would kill LCRP

Half-pager for go/no-go. Thresholds live in [`gates.md`](gates.md); this sheet
states the **falsifiers** and the **near-term proofs** that should trip them
early. No prose arithmetic: a kill is a measured miss against a pre-registered
bar.

## Thesis under test

A small FP8/INT8 logic core stays resident; long-tail knowledge lives as
digest-checked patches; per-segment route → prefetch → pin → apply beats a
same-VRAM dense core on long-tail **without** blowing the 16 GiB SKU or
collapsing core_logic.

## Instant kills (any one stops A′)

| # | Falsifier | How we know |
| --- | --- | --- |
| K1 | No real FP8/INT8 serve path | Transformers-style decompress → OOM, or peak resident **> 14 000 MiB** / core weights **> 9 000 MiB** on the blessed runtime |
| K2 | Pin integrity is theater | Digest mismatch rate **≠ 0** on scored runs, or apply proceeds without a matching pin digest |
| K3 | Prefetch cannot hide load | p95 `prefetch_wait_ms` **> 20** with pins in budget |
| K4 | Bank carries no knowledge | `oracle_patches − core_only` on long_tail_factual **< +15 pp** (n=100) |
| K5 | Router misses the bank | On long_tail_factual **or** multi_hop, `router < oracle − 5 pp` (n=100, τ frozen on calib) |
| K6 | Core is a hollowed shell | `core_only` on core_logic **> 3 pp** below `dense_ref` on the same items (i.e. must stay within **≤ 3 pp** of dense_ref, or kill). No ratio lens. |
| K7 | Wrong patch is free lunch | Deliberate mis-route / wrong-domain pin **improves** the cell vs core_only without a refuse — contamination, not retrieval |
| K8 | Hot path is not the eval path | Quality claimed on PEFT/offline generate while serve is vLLM (or vice versa) with no paired rerun — claim is void |

K1–K3 are systems kills. K4–K7 are product kills. K8 is a process kill.

## Does **not** kill (by itself)

- A single closed FAQ canary (e.g. Widget n=20) looking good or bad.
- Latency spikes with cold cache before prefetch is wired.
- Optional `dense_w4` absent — A′ forbids W4 win language; absence ≠ fail.
- long_context null when measured usable window **< 8k**.

## Near-term proofs (run in order; trip kills early)

1. **Second domain + deliberate mis-route** — train a non-Widget patch; pin the wrong one on Widget items. Expect: no silent lift vs core_only (K7). If wrong patch helps, stop.
2. **Same Widget cell via vLLM `enable_lora`** — digest-gated pin → apply on the serve path. Expect: digests 0 mismatch (K2); quality not worse than the PEFT canary by enough to doubt the loop (K8).
3. **n=100 long_tail + multi_hop, τ frozen** — three arms. Expect: K4 and K5 either clear or kill; do not retune τ on eval.

Only after 1–3 clear do we treat A′ quality gates as *in play* for a scored campaign commit. Until then LCRP is a **runtime thesis** with an **unproven quality bet**.

## Staffing rule

Do not staff Castle or Acquisition on LCRP until K4 and K5 have a green n=100 sheet under the blessed serve path. Systems green alone is not product green.

## Change control

Edits to kill bars or proof order land in a PR **before** the campaign that
uses them. Evidence for a kill or a pass goes under `docs/evidence/` (when
that tree exists) with journal traces — not README screenshots.
