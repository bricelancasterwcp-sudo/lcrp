# Assay-style eval plan for LCRP (pre-registered)

Sibling: [assay](https://github.com/bricelancasterwcp-sudo/assay).

This plan measures **whether a configuration is fit for a job**, not whether a
model is “smart.” Floors are registered before the run. Incomplete comparisons
are not green. `None` is not `0`.

Numeric floors for the first scored campaign live in
[`gates.md`](gates.md) (**Campaign A′**: 16 GiB GPU, ~7–8B FP8/INT8 core).
This file defines arms, cells, metrics, cover semantics, and canaries.

## Arms

| Arm | Meaning |
| --- | --- |
| `core_only` | Logic core, patches forced off |
| `oracle_patches` | Core + gold patch ids for the item (upper bound) |
| `router` | Core + learned router top-k (**k = 4** for A′) |
| `dense_w4` | Dense baseline at W4 (when available on same box) |
| `dense_ref` | Dense higher-precision reference (optional, for calibration) |

Every published claim names the arms that actually ran.

**Campaign A′ required arms:** `core_only`, `oracle_patches`, `router`.

## Cells (families)

1. **short_factual** — common facts core should often handle alone.
2. **long_tail_factual** — cluster-specific / rare facts patches must carry.
3. **multi_hop** — needs ≥2 patches or patch+core composition.
4. **long_context** — only at **measured** usable window; if usable &lt; 8k on
   the A′ SKU, store `null` + reason (do not pretend 64k).
5. **instruction_codec** — format landing / tool-shaped outputs if the product
   cares (assay’s lesson: codec fitness ≠ intelligence).
6. **core_logic** — reasoning/coding items that must not regress with patches off.

Add cells only with a schema bump and an explicit unmeasured mark on old
profiles — never silently rescore.

### Sample sizes (A′)

- **n = 100** for short_factual, long_tail_factual, multi_hop, core_logic,
  instruction_codec.
- **n = 30** for long_context when the cell is measured (usable ≥ 8k).

## Metrics

Per cell, record:

- primary task score (accuracy / EM / pass@k — fixed per cell before the run)
- `pins_per_segment`, `prefetch_wait_ms`, `resident_mib_peak`
- `core_mib`, `pins_mib` when accounting is available
- refusal count and degrade count (must be zero for “clean pass”)
- `n`, Wilson or pre-registered sequential stopping where rates apply

**None vs 0:** if a cell was not run, store `null` + drop reason. Never write
`0.0` for “did not measure.”

## Floors and cover (pointer)

Authoritative numbers: [`gates.md`](gates.md).

| Gate id | Intent (A′) |
| --- | --- |
| G-oracle-lift | oracle recovers long_tail vs core_only by **≥ +15 pp** |
| G-router-cover | router within **5 pp** of oracle on long_tail **and** multi_hop |
| G-core-alone | core_only within **3 pp** of dense_ref on core_logic (or ≥ 90% ratio) |
| G-vram-resident / G-vram-core / G-vram-pins | **≤ 14 000 / 9 000 / 1 024 MiB** |
| G-patch-warm | prefetch_wait_ms p95 **≤ 20**; digest mismatches **0** |

**Cover** means: every floor cell the floor profile measured is present on the
candidate and not worse than δ beyond noise. Missing cell → incomplete (fail
closed), not pass.

Router **τ** is frozen on a calibration split before eval; retuning τ on the
eval set invalidates the campaign.

## Incomplete comparison

Exit taxonomy (assay-shaped):

| Code | Meaning |
| --- | --- |
| 0 | comparable; no regression beyond noise / floors held |
| 1 | regression vs floor or oracle cover failed |
| 2 | not comparable (identity / instrument mismatch) |
| 3 | incomplete (cell measured on only one side) |
| 4 | infra / profile unreadable |

CI must treat 2/3/4 as red. Never celebrate exit 0 when cells were skipped.

## Canaries

Fail the run if any fire:

1. **Silent truncation / contract-looking success** — bloomery/assay class:
   reply OK but canary prefix lost or stats missing.
2. **Wrong patch, fluent answer** — force incorrect top-1; score must drop or
   abstain; fluent wrong = fail.
3. **Pin fail → unlabeled core-only** — any degrade without `degrade` journal
   event = fail.
4. **Unmeasured VRAM claimed precise** — planner must journal degradation mode.

## Claiming a win over W4

**Not available under Campaign A′.**

Allowed only under **Campaign B** (see `gates.md`), after A′ passes, and only
if all hold on the blessed 16 GiB SKU:

1. `router` covers `dense_w4` within **ε ≤ 2 pp** on the pre-registered mix.
2. Resident budget still holds (A′ numbers or the single alternate B story).
3. E2E p95 latency **≤ W4 + 10%** at matched quality.
4. Canaries clean; journal complete.
5. Artifacts and digests pinned; assay-style profile committed under
   `docs/evidence/` (future).

Until then, language is “hypothesis” or “prototype,” never “beats W4.”
