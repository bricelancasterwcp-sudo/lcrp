# Bloomery accounting folded into LCRP

Sibling: [bloomery](https://github.com/bricelancasterwcp-sudo/bloomery).

Bloomery treats the model endpoint as unreliable hardware: probe it, budget it,
admission-control it, refuse with arithmetic. LCRP steals that stance and adds
**parameter patches** as a first-class resident class beside weights and KV.

## Laws that transfer

### 1. One pool

Everything that occupies VRAM spends from one budget:

```
resident = core_weights + pinned_patches + kv + ctx_overhead + runtime_buffers
```

No separate “patch fairy dust” line that the planner cannot see. Under-declare
→ OOM; over-declare → earlier honest refusal. Prefer refusal.

### 2. Computed windows, never configured ones

Usable context is min(training_ctx, bytes_left / kv_per_token, user_cap).
Patches change `bytes_left`. A route that would pin more than free VRAM allows
must refuse **before** allocate, with numbers.

### 3. Refusals, never silent degradation

If the segment needs patches that do not fit:

- HTTP/API shape can mirror bloomery (`409` / placeability payload).
- Payload fields (names illustrative): `bytes_needed`, `bytes_free`,
  `bytes_reclaimable`, `max_placeable_patches`, `blocker`
  (`weights` | `pins` | `kv` | `unmeasured`).

Never answer core-only after a failed pin without labeling the degradation.

### 4. Patches are pageable residents

| Concept | Bloomery analogue | LCRP |
| --- | --- | --- |
| Agent KV image | Suspend/resume | Patch pin image (tensors + digest) |
| Priority eviction | Strict priority then timeshare | Pin LRU + request priority |
| Model digest invalidation | Changed GGUF → cold start | Changed patch digest → must reload |
| Unmeasured VRAM | Cap residency; journal degradation | Cap pin count; journal degradation |

Pin slots are finite. Planner decides top-k **and** whether they fit.

### 5. Prefetch is part of admission

Segment N routes while segment N−1 computes. Journal `prefetch_wait_ms`.
If wait dominates TTFT/TBT on the target box, G-latency-prefetch fails — that
is a product kill, not a tuning footnote.

### 6. Journal everything

Append-only events (minimum):

- `boot` — digests, measured budget, profile id
- `route` — segment id, query hash, top-k ids, scores, τ decision
- `prefetch` — ids, bytes, wait_ms, hit/miss
- `pin` / `evict` — slot, id, digest, reason
- `apply` — layers touched, fuse mode
- `refuse` — arithmetic payload
- `degrade` — named (e.g. `core_only_after_pin_fail`) — should be rare and loud

### 7. Boot admission

Do not serve until a measured profile exists for:

`(core_digest, bank_digest, router_digest, hardware_tier)`

Borrow assay’s posture: unprofiled → refuse or provisional-with-journal, never
silent trust. Swap-candidate style checks: a new bank/router must **cover** the
blessed floor profile before promotion.

## What not to copy blindly

- Bloomery pages **KV / agents** for a dense model. LCRP pages **parameters**.
  Fuse cost and bank digest invalidation need their own gates
  (`G-patch-warm`, `G-router-cover`).
- Weights in bloomery are charged but not auto-evicted; patches **are** meant
  to evict. Different policy by design.
- Hybrid / MoE geometry quirks in serving still apply to the core blob —
  measure kv_per_token from the actual artifact (assay / gguf-geometry
  lessons).

## Planner sketch

```
avail = budget - overhead - core_weights - sum(pinned) - sum(kv_reservations)
if cost(top_k_patches) + cost(kv_for_window) > avail:
    try_evict_pins(priority_rules)
if still insufficient:
    refuse(arithmetic)
else:
    prefetch_and_pin(top_k)
```

Wire this before optimizing router ML. Wrong accounting makes good routers look
like OOMs.
