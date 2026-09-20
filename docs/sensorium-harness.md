# Sensorium harness notes for LCRP testing

Sibling: [sensorium](https://github.com/bricelancasterwcp-sudo/sensorium).

Sensorium records what a program **actually did** and answers from the trace —
or refuses by name. During LCRP prototypes, use it (or a sensorium-shaped
SQLite/JSONL journal) so routing and paging bugs are not debugged by vibe.

## What to record

Minimum event kinds on the hot path:

| Event | Fields |
| --- | --- |
| `segment_start` | run_id, segment_id, token_span |
| `route` | query_hash, top_k[{id,score}], tau, decision(`pins`\|`core_only`) |
| `prefetch` | ids, bytes, wait_ms, cache_hit |
| `pin` | slot, patch_id, digest, layers |
| `evict` | slot, patch_id, reason |
| `apply` | fuse_mode, layers_touched |
| `forward` | timing, tokens_in/out |
| `refuse` / `degrade` | arithmetic payload or degrade class |
| `task_outcome` | item_id, arm, score, notes |

Capture digests for core, bank, router at `boot`. If LINE/locals-style detail
is needed for router code, use sensorium `--focus` on the router module only —
keep overhead intentional.

## Questions the trace must answer

- Which patch ids were hot when item X failed?
- Was this a miss (never retrieved) or a hit (retrieved, still wrong)?
- Did we wait on PCIe before the first token?
- Did we silently run core-only after a pin failure?
- Did two multi-hop items require disjoint packs that thrashed the pin set?

If the trace cannot settle a question, the harness prints a refusal
(`NOTHING WAS CHECKED` / `REFUSED`) and the exact re-record command — never a
guess.

## Honesty rules (non-negotiable)

1. Truncated captures are marked, not repaired.
2. Unrecorded sites are counted as unchecked, not as zeros.
3. MATCH/diff of two runs is about **causal shape** unless values were
   captured — do not overclaim.
4. No answer fabricates a routing story from partial logs.

## Plug-in to the eval loop

```
for arm in {core_only, oracle, router}:
  for item in cell:
    with sensorium_or_journal.trace(run_id):
      result = serve(arm, item)
    score(item, result)
    assert_canaries(trace)
aggregate → assay profile fields + cover check
```

Store traces beside profiles (`runs/<campaign>/<run_id>/`). When investigating
a failed cover gate, open the trace first, not the spreadsheet.

## Prototype practicality

Until the pager is in Rust/bloomery, a Python prototype may write JSONL with
the same event names. Keep field names stable so a later sensorium converter
or MCP query layer can sit on top without renaming history.
