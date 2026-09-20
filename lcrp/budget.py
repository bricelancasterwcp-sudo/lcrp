"""Resident VRAM budget helpers with Campaign A′ admission.

Design authority: docs/bloomery-accounting.md, docs/gates.md

Unmeasured values are None — never coerced to zero for pass/fail.
"""

from __future__ import annotations

from dataclasses import dataclass

from lcrp.campaign import A_PRIME, CampaignAPrime, CoreDtype


@dataclass(frozen=True)
class ResidentBudget:
    """Single-pool resident budget in MiB (None = unmeasured)."""

    total_mib: int | None
    core_weights_mib: int | None
    pinned_patches_mib: int | None
    kv_mib: int | None
    overhead_mib: int | None

    def free_mib(self) -> int | None:
        parts = (
            self.total_mib,
            self.core_weights_mib,
            self.pinned_patches_mib,
            self.kv_mib,
            self.overhead_mib,
        )
        if any(p is None for p in parts):
            return None
        assert self.total_mib is not None
        used = (
            (self.core_weights_mib or 0)
            + (self.pinned_patches_mib or 0)
            + (self.kv_mib or 0)
            + (self.overhead_mib or 0)
        )
        return self.total_mib - used

    def used_mib(self) -> int | None:
        parts = (
            self.core_weights_mib,
            self.pinned_patches_mib,
            self.kv_mib,
            self.overhead_mib,
        )
        if any(p is None for p in parts):
            return None
        return (
            (self.core_weights_mib or 0)
            + (self.pinned_patches_mib or 0)
            + (self.kv_mib or 0)
            + (self.overhead_mib or 0)
        )


@dataclass(frozen=True)
class RefusalArithmetic:
    bytes_needed: int | None
    bytes_free: int | None
    bytes_reclaimable: int | None
    max_placeable_patches: int | None
    blocker: str
    detail: str


@dataclass(frozen=True)
class AdmissionResult:
    admitted: bool
    refusal: RefusalArithmetic | None = None


def _mib_to_bytes(mib: int | None) -> int | None:
    if mib is None:
        return None
    return mib * 1024 * 1024


def admit(
    budget: ResidentBudget,
    *,
    core_dtype: CoreDtype | None,
    campaign: CampaignAPrime = A_PRIME,
    pretend_measured: bool = False,
) -> AdmissionResult:
    """Admit a resident set under Campaign A′ caps."""
    if pretend_measured:
        return AdmissionResult(
            False,
            RefusalArithmetic(
                bytes_needed=None,
                bytes_free=_mib_to_bytes(budget.free_mib()),
                bytes_reclaimable=None,
                max_placeable_patches=None,
                blocker="unmeasured",
                detail="unmeasured values presented as measured",
            ),
        )

    if core_dtype is None or core_dtype not in campaign.allowed_core_dtypes:
        return AdmissionResult(
            False,
            RefusalArithmetic(
                bytes_needed=None,
                bytes_free=_mib_to_bytes(budget.free_mib()),
                bytes_reclaimable=None,
                max_placeable_patches=None,
                blocker="core_dtype",
                detail=(
                    "core dtype must be one of "
                    f"{[d.value for d in campaign.allowed_core_dtypes]}"
                ),
            ),
        )

    required = (
        budget.core_weights_mib,
        budget.pinned_patches_mib,
        budget.kv_mib,
        budget.overhead_mib,
    )
    if any(v is None for v in required):
        return AdmissionResult(
            False,
            RefusalArithmetic(
                bytes_needed=None,
                bytes_free=_mib_to_bytes(budget.free_mib()),
                bytes_reclaimable=None,
                max_placeable_patches=None,
                blocker="unmeasured",
                detail="core/pins/kv/overhead must be measured for admission",
            ),
        )

    assert budget.core_weights_mib is not None
    assert budget.pinned_patches_mib is not None

    if budget.core_weights_mib > campaign.max_core_mib:
        return AdmissionResult(
            False,
            RefusalArithmetic(
                bytes_needed=_mib_to_bytes(budget.core_weights_mib),
                bytes_free=_mib_to_bytes(budget.free_mib()),
                bytes_reclaimable=None,
                max_placeable_patches=None,
                blocker="core",
                detail=(
                    f"core {budget.core_weights_mib} MiB > "
                    f"cap {campaign.max_core_mib}"
                ),
            ),
        )

    if budget.pinned_patches_mib > campaign.max_pins_mib:
        return AdmissionResult(
            False,
            RefusalArithmetic(
                bytes_needed=_mib_to_bytes(budget.pinned_patches_mib),
                bytes_free=_mib_to_bytes(budget.free_mib()),
                bytes_reclaimable=None,
                max_placeable_patches=None,
                blocker="pins",
                detail=(
                    f"pins {budget.pinned_patches_mib} MiB > "
                    f"cap {campaign.max_pins_mib}"
                ),
            ),
        )

    used = budget.used_mib()
    assert used is not None
    if used > campaign.max_resident_mib:
        return AdmissionResult(
            False,
            RefusalArithmetic(
                bytes_needed=_mib_to_bytes(used),
                bytes_free=_mib_to_bytes(budget.free_mib()),
                bytes_reclaimable=None,
                max_placeable_patches=None,
                blocker="resident",
                detail=f"resident {used} MiB > cap {campaign.max_resident_mib}",
            ),
        )

    return AdmissionResult(True, None)
