"""Campaign A′ constants — single source of truth for the blessed SKU.

Authoritative prose: docs/gates.md. Tests assert these numbers match the docs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CoreDtype(str, Enum):
    FP8 = "fp8"
    INT8 = "int8"


@dataclass(frozen=True)
class CampaignAPrime:
    """Locked Campaign A′ thresholds (16 GiB / ~7–8B)."""

    name: str = "A_prime"
    gpu_vram_mib: int = 16 * 1024
    max_resident_mib: int = 14_000
    max_core_mib: int = 9_000
    max_pins_mib: int = 1_024
    max_prefetch_wait_ms_p95: float = 20.0
    max_pin_digest_mismatch_rate: float = 0.0
    min_oracle_lift_pp: float = 15.0
    max_router_oracle_gap_pp: float = 5.0
    max_core_logic_gap_pp: float = 3.0
    min_core_logic_ratio: float = 0.90
    top_k: int = 4
    min_long_context_usable_tokens: int = 8_000
    n_standard_cell: int = 100
    n_long_context_cell: int = 30
    required_arms: tuple[str, ...] = ("core_only", "oracle_patches", "router")
    allowed_core_dtypes: tuple[CoreDtype, ...] = (CoreDtype.FP8, CoreDtype.INT8)


A_PRIME = CampaignAPrime()
