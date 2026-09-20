# lcrp

**Logic-Core + Retrieved Parametric Patches** — Phase 0 design and scaffold.

Keep a small always-resident *logic core* in VRAM. Store most parametric
knowledge as addressable weight patches (LoRA / delta) on CPU or NVMe. Route
and pin only the patches the current segment needs.

This repo follows the same honesty tradition as
[assay](https://github.com/bricelancasterwcp-sudo/assay),
[bloomery](https://github.com/bricelancasterwcp-sudo/bloomery), and
[sensorium](https://github.com/bricelancasterwcp-sudo/sensorium):
measure, refuse rather than guess, journal what actually ran.

## Status

**Phase 0 — design.** Docs and stub modules only. No training stack yet.

## Read in this order

1. [docs/architecture.md](docs/architecture.md) — system sketch
2. [docs/bloomery-accounting.md](docs/bloomery-accounting.md) — VRAM budget, pager, refusals
3. [docs/assay-eval-plan.md](docs/assay-eval-plan.md) — pre-registered eval
4. [docs/sensorium-harness.md](docs/sensorium-harness.md) — what to trace in tests
5. [docs/borrowed-laws.md](docs/borrowed-laws.md) — constitution from sibling repos
6. [docs/gates.md](docs/gates.md) — kill gates

## Sibling stack

| Repo | Role here |
| --- | --- |
| [assay](https://github.com/bricelancasterwcp-sudo/assay) | Capability profiles, floors, cover, None≠0 |
| [bloomery](https://github.com/bricelancasterwcp-sudo/bloomery) | Measured VRAM budgets, pager, arithmetic refusals |
| [sensorium](https://github.com/bricelancasterwcp-sudo/sensorium) | Execution traces that refuse rather than invent |

## Package stubs

Python package `lcrp` currently holds docstring-only modules (`budget`,
`router`, `journal`) that point at the docs. They exist so the design has a
code-shaped home; they are not an implementation.

## License

MIT. bloomery remains AGPL on its own terms; this repo does not relicense it.
