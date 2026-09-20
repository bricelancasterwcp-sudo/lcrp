"""Segment loop: route → prefetch → pin → admit → apply.

Journals every step. Refuses with bloomery-style arithmetic when A′ caps fail.
"""

from __future__ import annotations

from dataclasses import dataclass

from lcrp.apply import ApplySpec, build_apply_spec
from lcrp.bank import PatchRecord, PinCache, make_toy_bank
from lcrp.budget import AdmissionResult, ResidentBudget, admit
from lcrp.campaign import A_PRIME, CampaignAPrime, CoreDtype
from lcrp.core import A_PRIME_CORE, CoreSpec, boot_core
from lcrp.journal import Journal
from lcrp.router import PatchScore, RouteDecision, ScoreRouter


def score_patches(
    query: tuple[float, ...],
    bank: dict[str, PatchRecord],
) -> list[PatchScore]:
    scores: list[PatchScore] = []
    for patch in bank.values():
        # Dot product; both vectors are short fixed tuples.
        s = sum(a * b for a, b in zip(query, patch.embedding, strict=False))
        scores.append(PatchScore(patch_id=patch.patch_id, score=float(s)))
    return scores


@dataclass(frozen=True)
class SegmentResult:
    segment_id: str
    route: RouteDecision
    admission: AdmissionResult
    applied: tuple[str, ...]
    evicted: tuple[str, ...]
    refused: bool
    apply_spec: ApplySpec | None = None


@dataclass
class SegmentLoop:
    """Stateful fake-bank runtime for one scored session."""

    core_dtype: CoreDtype
    core_weights_mib: int
    kv_mib: int
    overhead_mib: int
    tau: float
    prefetch_wait_ms: float = 5.0
    campaign: CampaignAPrime = A_PRIME
    journal: Journal | None = None
    bank: dict[str, PatchRecord] | None = None
    core_spec: CoreSpec | None = None

    def __post_init__(self) -> None:
        if self.journal is None:
            self.journal = Journal()
        if self.bank is None:
            self.bank = make_toy_bank()
        if self.core_spec is None:
            self.core_spec = A_PRIME_CORE
        self.router = ScoreRouter(tau=self.tau, campaign=self.campaign)
        self.pins = PinCache(max_mib=self.campaign.max_pins_mib)
        # Wiring-only boot: journals model_id/dtype/weights; does not load weights.
        boot_core(
            self.journal,
            self.core_spec,
            core_dtype=self.core_dtype,
            core_weights_mib=self.core_weights_mib,
            campaign=self.campaign.name,
            bank_size=len(self.bank),
        )

    def run_segment(
        self,
        segment_id: str,
        query: tuple[float, ...],
    ) -> SegmentResult:
        assert self.journal is not None and self.bank is not None
        scores = score_patches(query, self.bank)
        route = self.router.route(segment_id, scores)
        self.journal.record(
            "route",
            segment_id=segment_id,
            core_only=route.core_only,
            top_k=[p.patch_id for p in route.top_k],
            scores={p.patch_id: p.score for p in route.top_k},
            tau=route.tau,
        )

        evicted: list[str] = []
        applied: list[str] = []

        if route.core_only:
            budget = ResidentBudget(
                total_mib=self.campaign.gpu_vram_mib,
                core_weights_mib=self.core_weights_mib,
                pinned_patches_mib=self.pins.used_mib,
                kv_mib=self.kv_mib,
                overhead_mib=self.overhead_mib,
            )
            admission = admit(
                budget, core_dtype=self.core_dtype, campaign=self.campaign
            )
            if not admission.admitted:
                self.journal.record(
                    "refuse",
                    segment_id=segment_id,
                    blocker=admission.refusal.blocker if admission.refusal else None,
                    detail=admission.refusal.detail if admission.refusal else None,
                )
                return SegmentResult(
                    segment_id=segment_id,
                    route=route,
                    admission=admission,
                    applied=(),
                    evicted=(),
                    refused=True,
                )
            return SegmentResult(
                segment_id=segment_id,
                route=route,
                admission=admission,
                applied=(),
                evicted=(),
                refused=False,
            )

        for ps in route.top_k:
            patch = self.bank[ps.patch_id]
            self.journal.record(
                "prefetch",
                segment_id=segment_id,
                patch_id=patch.patch_id,
                digest=patch.digest,
                size_mib=patch.size_mib,
                wait_ms=self.prefetch_wait_ms,
            )
            try:
                evicted.extend(
                    self.pins.pin(patch, wait_ms=self.prefetch_wait_ms)
                )
            except MemoryError as exc:
                self.journal.record(
                    "refuse",
                    segment_id=segment_id,
                    blocker="pins",
                    detail=str(exc),
                )
                refusal_budget = ResidentBudget(
                    total_mib=self.campaign.gpu_vram_mib,
                    core_weights_mib=self.core_weights_mib,
                    pinned_patches_mib=self.pins.used_mib,
                    kv_mib=self.kv_mib,
                    overhead_mib=self.overhead_mib,
                )
                admission = admit(
                    refusal_budget,
                    core_dtype=self.core_dtype,
                    campaign=self.campaign,
                )
                # Force refuse semantics even if leftover pins fit.
                if admission.admitted:
                    from lcrp.budget import AdmissionResult, RefusalArithmetic

                    admission = AdmissionResult(
                        False,
                        RefusalArithmetic(
                            bytes_needed=patch.size_mib * 1024 * 1024,
                            bytes_free=None,
                            bytes_reclaimable=None,
                            max_placeable_patches=None,
                            blocker="pins",
                            detail=str(exc),
                        ),
                    )
                return SegmentResult(
                    segment_id=segment_id,
                    route=route,
                    admission=admission,
                    applied=tuple(applied),
                    evicted=tuple(evicted),
                    refused=True,
                )
            self.journal.record(
                "pin",
                segment_id=segment_id,
                patch_id=patch.patch_id,
                digest=patch.digest,
                used_mib=self.pins.used_mib,
            )
            applied.append(patch.patch_id)

        budget = ResidentBudget(
            total_mib=self.campaign.gpu_vram_mib,
            core_weights_mib=self.core_weights_mib,
            pinned_patches_mib=self.pins.used_mib,
            kv_mib=self.kv_mib,
            overhead_mib=self.overhead_mib,
        )
        admission = admit(
            budget, core_dtype=self.core_dtype, campaign=self.campaign
        )
        if not admission.admitted:
            self.journal.record(
                "refuse",
                segment_id=segment_id,
                blocker=admission.refusal.blocker if admission.refusal else None,
                detail=admission.refusal.detail if admission.refusal else None,
            )
            return SegmentResult(
                segment_id=segment_id,
                route=route,
                admission=admission,
                applied=(),
                evicted=tuple(evicted),
                refused=True,
            )

        apply_spec = build_apply_spec(self.pins, applied)
        self.journal.record(
            "apply",
            segment_id=segment_id,
            patch_ids=list(applied),
            pinned_mib=self.pins.used_mib,
            adapter_paths=list(apply_spec.adapter_paths),
            digests=list(apply_spec.digests),
        )
        return SegmentResult(
            segment_id=segment_id,
            route=route,
            admission=admission,
            applied=tuple(applied),
            evicted=tuple(evicted),
            refused=False,
            apply_spec=apply_spec,
        )
