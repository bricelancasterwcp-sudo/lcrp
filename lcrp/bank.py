"""In-memory fake patch bank + pin cache for Phase 1 loops.

Not a real NVMe/LoRA store — digests and MiB charges are synthetic so the
admission / prefetch / pin path can be exercised under Campaign A′.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class PatchRecord:
    patch_id: str
    size_mib: int
    digest: str
    domain: str
    # Fixed embedding used by the fake scorer (dot with query).
    embedding: tuple[float, ...]


def _digest(patch_id: str, size_mib: int, domain: str) -> str:
    raw = f"{patch_id}:{size_mib}:{domain}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def make_toy_bank() -> dict[str, PatchRecord]:
    """Small deterministic bank spanning a few domains."""
    specs = [
        ("law.torts", 64, "law", (1.0, 0.0, 0.0, 0.1)),
        ("law.contracts", 64, "law", (0.9, 0.1, 0.0, 0.0)),
        ("code.python", 96, "code", (0.0, 1.0, 0.0, 0.0)),
        ("code.rust", 96, "code", (0.1, 0.9, 0.0, 0.0)),
        ("med.cardio", 80, "med", (0.0, 0.0, 1.0, 0.0)),
        ("med.neuro", 80, "med", (0.0, 0.1, 0.9, 0.0)),
        ("gen.trivia", 48, "gen", (0.2, 0.2, 0.2, 1.0)),
        ("gen.history", 48, "gen", (0.1, 0.1, 0.1, 0.9)),
    ]
    bank: dict[str, PatchRecord] = {}
    for patch_id, size_mib, domain, emb in specs:
        bank[patch_id] = PatchRecord(
            patch_id=patch_id,
            size_mib=size_mib,
            digest=_digest(patch_id, size_mib, domain),
            domain=domain,
            embedding=emb,
        )
    return bank


@dataclass
class PinSlot:
    patch: PatchRecord
    wait_ms: float


class PinCache:
    """Simple LRU pin cache charged in MiB."""

    def __init__(self, max_mib: int) -> None:
        self.max_mib = max_mib
        self._slots: OrderedDict[str, PinSlot] = OrderedDict()

    @property
    def used_mib(self) -> int:
        return sum(s.patch.size_mib for s in self._slots.values())

    def pinned_ids(self) -> tuple[str, ...]:
        return tuple(self._slots.keys())

    def get(self, patch_id: str) -> PinSlot | None:
        slot = self._slots.get(patch_id)
        if slot is not None:
            self._slots.move_to_end(patch_id)
        return slot

    def pin(self, patch: PatchRecord, *, wait_ms: float) -> list[str]:
        """Pin patch; return list of evicted patch ids."""
        if patch.patch_id in self._slots:
            self._slots.move_to_end(patch.patch_id)
            self._slots[patch.patch_id] = PinSlot(patch=patch, wait_ms=wait_ms)
            return []
        evicted: list[str] = []
        while self._slots and self.used_mib + patch.size_mib > self.max_mib:
            eid, _ = self._slots.popitem(last=False)
            evicted.append(eid)
        if self.used_mib + patch.size_mib > self.max_mib:
            raise MemoryError(
                f"cannot pin {patch.patch_id}: "
                f"{patch.size_mib} MiB > cache max {self.max_mib}"
            )
        self._slots[patch.patch_id] = PinSlot(patch=patch, wait_ms=wait_ms)
        return evicted
