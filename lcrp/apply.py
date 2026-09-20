"""Apply specs for pinned patches — backend-agnostic.

The loop journals apply events; inference backends (vLLM LoRA, PEFT) consume
ApplySpec without the library importing heavy ML deps.
"""

from __future__ import annotations

from dataclasses import dataclass

from lcrp.bank import PatchRecord, PinCache


@dataclass(frozen=True)
class ApplySpec:
    """Instructions for an inference backend to activate pinned adapters."""

    patch_ids: tuple[str, ...]
    adapter_paths: tuple[str, ...]
    digests: tuple[str, ...]
    pinned_mib: int


def build_apply_spec(pins: PinCache, patch_ids: tuple[str, ...] | list[str]) -> ApplySpec:
    """Build apply spec for ids that are still pinned (evicted ids are skipped)."""
    paths: list[str] = []
    digests: list[str] = []
    ordered: list[str] = []
    for pid in patch_ids:
        slot = pins.get(pid)
        if slot is None:
            continue  # LRU may have evicted earlier pins in this segment
        patch: PatchRecord = slot.patch
        ordered.append(pid)
        digests.append(patch.digest)
        if patch.adapter_path:
            paths.append(patch.adapter_path)
    return ApplySpec(
        patch_ids=tuple(ordered),
        adapter_paths=tuple(paths),
        digests=tuple(digests),
        pinned_mib=pins.used_mib,
    )
