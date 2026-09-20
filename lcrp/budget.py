"""Resident VRAM budget helpers.

Design authority: docs/bloomery-accounting.md

Phase 0: no real allocator. Callers should treat missing measurements as
unmeasured (None), never as zero.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidentBudget:
    """Single-pool resident budget in bytes (None = unmeasured)."""

    total: int | None
    core_weights: int | None
    pinned_patches: int | None
    kv: int | None
    overhead: int | None

    def free(self) -> int | None:
        parts = (
            self.total,
            self.core_weights,
            self.pinned_patches,
            self.kv,
            self.overhead,
        )
        if any(p is None for p in parts):
            return None
        assert self.total is not None
        used = (
            (self.core_weights or 0)
            + (self.pinned_patches or 0)
            + (self.kv or 0)
            + (self.overhead or 0)
        )
        return self.total - used


@dataclass(frozen=True)
class RefusalArithmetic:
    bytes_needed: int
    bytes_free: int | None
    bytes_reclaimable: int | None
    max_placeable_patches: int | None
    blocker: str
