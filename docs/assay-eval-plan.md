# Assay-style eval plan for LCRP (pre-registered)

Sibling: [assay](https://github.com/bricelancasterwcp-sudo/assay).

This plan measures **whether a configuration is fit for a job**, not whether a
model is “smart.” Floors are registered before the run. Incomplete comparisons
are not green. `None` is not `0`.

## Arms

| Arm | Meaning |
| --- | --- |
| `core_only` | Logic core, patches forced off |
| `oracle_patches` | Core + gold patch ids for the item (upper bound) |
| `router` | Core + learned router top-k |
| `dense_w4` | Dense baseline at W4 (when available on same box) |
| `dense_ref` | Dense higher-precision reference (optional, for calibration) |

Every published claim names the arms that actually ran.

## Cells (families)

1. **short_factual** — common facts core should often handle alone.
2. **long_tail_factual** — cluster-specific / rare facts patches must carry.
3. **multi_hop** — needs ≥2 patches or patch+core composition.
4. **long_context** — ≥64k when the core artifact supports it; expect W4-like cliffs.
5. **instruction_codec** — format landing / tool-shaped outputs if the product cares
   (assay’s lesson: codec fitness ≠ intelligence).
6. **core_logic** — reasoning/coding items that must not regress with patches off.

Add cells only with a schema bump and an explicit unmeasured mark on old
profiles — never silently rescore.

## Metrics

Per cell, record:

- primary task score (accuracy / EM / pass@k — fixed per cell)
- `pins_per_segment`, `prefetch_wait_ms`, `resident_mib_peak`
- refusal count and degrade count (must be zero for “clean pass”)
- `n`, Wilson or pre-registered sequential stopping where rates apply

**None vs 0:** if a cell was not run, store `null` + drop reason. Never write
`0.0` for “did not measure.”

## Floors and cover

Pre-register numeric floors before the campaign (examples below are
**placeholders to replace with chosen floors**, not results):

| Gate id | Intent |
| --- | --- |
| floor_core_logic | `core_only` stays usable on core_logic |
| floor_oracle_tail | `oracle_patches` recovers long_tail vs `core_only` |
| cover_router_oracle | `router` covers `oracle_patches` within δ on long_tail + multi_hop |
| cover_router_w4 | `router` covers `dense_w4` within ε on agreed aggregate |
| budget_resident | peak resident MiB ≤ tier budget |
| latency_prefetch | prefetch_wait_ms p95 ≤ L |

**Cover** means: every floor cell the floor profile measured is present on the
candidate and not worse than δ beyond noise. Missing cell → incomplete (fail
closed), not pass.

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

Allowed only if all hold on the blessed tier:

1. `router` covers `dense_w4` on the pre-registered aggregate.
2. `budget_resident` holds with headroom.
3. `latency_prefetch` holds.
4. Canaries clean.
5. Artifacts and digests pinned; assay-style profile committed under
   `docs/evidence/` (future).

Until then, language is “hypothesis” or “prototype,” never “beats W4.”
