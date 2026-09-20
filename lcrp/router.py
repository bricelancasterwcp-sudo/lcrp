"""Patch router interface (Phase 0 stub).

Design authority: docs/architecture.md, docs/assay-eval-plan.md
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchScore:
    patch_id: str
    score: float


@dataclass(frozen=True)
class RouteDecision:
    segment_id: str
    top_k: tuple[PatchScore, ...]
    core_only: bool
    tau: float


class Router:
    """Segment-level patch router. Not implemented in Phase 0."""

    def route(self, segment_id: str, query_vector: object) -> RouteDecision:
        raise NotImplementedError("Phase 0 stub — see docs/architecture.md")
