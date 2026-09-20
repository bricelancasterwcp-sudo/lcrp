"""Segment-level patch router.

Design authority: docs/architecture.md, docs/gates.md
"""

from __future__ import annotations

from dataclasses import dataclass

from lcrp.campaign import A_PRIME, CampaignAPrime


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


class ScoreRouter:
    """Deterministic top-k router over precomputed patch scores."""

    def __init__(self, tau: float, campaign: CampaignAPrime = A_PRIME) -> None:
        self.tau = tau
        self.campaign = campaign

    def route(
        self,
        segment_id: str,
        scores: list[PatchScore] | tuple[PatchScore, ...],
    ) -> RouteDecision:
        ordered = tuple(
            sorted(scores, key=lambda s: (-s.score, s.patch_id))[
                : self.campaign.top_k
            ]
        )
        if not ordered or ordered[0].score < self.tau:
            return RouteDecision(
                segment_id=segment_id,
                top_k=(),
                core_only=True,
                tau=self.tau,
            )
        return RouteDecision(
            segment_id=segment_id,
            top_k=ordered,
            core_only=False,
            tau=self.tau,
        )
