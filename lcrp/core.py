"""Campaign A′ core model wiring — path resolve + config probe (no ML deps).

Does not load weights. Fake-bank SegmentLoop keeps working without a local
checkout of the model.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from lcrp.campaign import CoreDtype
from lcrp.journal import Journal

ENV_CORE_PATH = "LCRP_CORE_PATH"

# Mid-range estimate for FP8 7B weights (~7.5–8.2 GiB); not measured VRAM.
_A_PRIME_APPROX_WEIGHTS_MIB = 7800


@dataclass(frozen=True)
class CoreSpec:
    """Blessed core SKU for Campaign A′ (identity + optional local layout)."""

    model_id: str
    local_path: str | None
    dtype: CoreDtype
    approx_weights_mib: int
    notes: str


@dataclass(frozen=True)
class CoreProbe:
    """Result of a lightweight on-disk config.json probe (stdlib only)."""

    ok: bool
    path: str
    model_type: str | None = None
    architectures: tuple[str, ...] | None = None
    quantization_summary: str | None = None
    detail: str | None = None


A_PRIME_CORE = CoreSpec(
    model_id="RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic",
    local_path="/mnt/extra/models/Qwen2.5-7B-Instruct-FP8-dynamic",
    dtype=CoreDtype.FP8,
    approx_weights_mib=_A_PRIME_APPROX_WEIGHTS_MIB,
    notes=(
        "Estimate for FP8 7B weights (~7.5–8.2 GiB on disk / weight footprint); "
        "not a measured resident VRAM number. Override path via LCRP_CORE_PATH."
    ),
)


def resolve_core_path(
    spec: CoreSpec,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve a core location for tooling.

    Order: ``LCRP_CORE_PATH`` env, then ``spec.local_path`` if that path
    exists on disk, else the Hugging Face ``model_id`` string. Never invents
    files or downloads weights.
    """
    environ = os.environ if env is None else env
    override = environ.get(ENV_CORE_PATH)
    if override:
        return override
    if spec.local_path and Path(spec.local_path).exists():
        return spec.local_path
    return spec.model_id


def _quantization_summary(qcfg: Any) -> str | None:
    if qcfg is None:
        return None
    if not isinstance(qcfg, dict):
        return str(qcfg)
    parts: list[str] = []
    for key in (
        "quant_method",
        "format",
        "bits",
        "weight_bits",
        "activation_scheme",
        "kv_cache_scheme",
    ):
        if key in qcfg and qcfg[key] is not None:
            parts.append(f"{key}={qcfg[key]}")
    if parts:
        return ", ".join(parts)
    # Compact fallback — avoid dumping huge nested objects.
    return json.dumps(qcfg, sort_keys=True)[:240]


def probe_core(path: str) -> CoreProbe:
    """Probe a local path for HF ``config.json`` metadata.

    Missing paths return ``ok=False``. No transformers / torch import.
    """
    root = Path(path)
    if not root.exists():
        return CoreProbe(ok=False, path=path, detail="path missing")

    config_file = root / "config.json" if root.is_dir() else root
    if not config_file.is_file():
        return CoreProbe(ok=False, path=path, detail="config.json missing")

    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return CoreProbe(ok=False, path=path, detail=f"config read error: {exc}")

    if not isinstance(data, dict):
        return CoreProbe(ok=False, path=path, detail="config.json root is not an object")

    model_type = data.get("model_type")
    arches = data.get("architectures")
    architectures: tuple[str, ...] | None
    if isinstance(arches, list):
        architectures = tuple(str(a) for a in arches)
    else:
        architectures = None

    return CoreProbe(
        ok=True,
        path=path,
        model_type=str(model_type) if model_type is not None else None,
        architectures=architectures,
        quantization_summary=_quantization_summary(data.get("quantization_config")),
    )


def boot_core(
    journal: Journal,
    spec: CoreSpec,
    *,
    core_dtype: CoreDtype | None = None,
    core_weights_mib: int | None = None,
    **extra: Any,
) -> None:
    """Journal a ``boot`` event with core identity — does not load weights."""
    dtype = core_dtype if core_dtype is not None else spec.dtype
    weights = (
        core_weights_mib if core_weights_mib is not None else spec.approx_weights_mib
    )
    journal.record(
        "boot",
        model_id=spec.model_id,
        core_dtype=dtype.value,
        core_weights_mib=weights,
        **extra,
    )
