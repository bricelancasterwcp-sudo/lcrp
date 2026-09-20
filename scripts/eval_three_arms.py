#!/usr/bin/env python3
"""Three-arm dry harness: core_only / oracle_patches journal path.

Default is CI-safe (no GPU). Pass --adapter once widget LoRA exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lcrp.bank import load_adapter_patch, make_toy_bank
from lcrp.campaign import CoreDtype
from lcrp.loop import SegmentLoop

DEFAULT_ADAPTER = "/mnt/extra/models/lcrp-patches/widget-api-lora"
EMB = (0.0, 1.0, 0.0, 0.0)


def run(adapter: Path | None) -> dict:
    if adapter and (adapter / "adapter_config.json").exists():
        patch = load_adapter_patch(
            patch_id="widget.api",
            adapter_path=adapter,
            domain="widget_api",
            embedding=EMB,
        )
        bank = {patch.patch_id: patch}
    else:
        bank = make_toy_bank()

    core = SegmentLoop(
        core_dtype=CoreDtype.FP8,
        core_weights_mib=8407,
        kv_mib=1500,
        overhead_mib=400,
        tau=10.0,
        bank=bank,
    ).run_segment("core_only", EMB)

    oracle = SegmentLoop(
        core_dtype=CoreDtype.FP8,
        core_weights_mib=8407,
        kv_mib=1500,
        overhead_mib=400,
        tau=0.05,
        bank=bank,
    ).run_segment("oracle", EMB)

    return {
        "adapter": str(adapter) if adapter else None,
        "core_only": {
            "refused": core.refused,
            "route_core_only": core.route.core_only,
            "applied": list(core.applied),
        },
        "oracle": {
            "refused": oracle.refused,
            "route_core_only": oracle.route.core_only,
            "applied": list(oracle.applied),
            "adapter_paths": list(oracle.apply_spec.adapter_paths)
            if oracle.apply_spec
            else [],
            "digests": list(oracle.apply_spec.digests) if oracle.apply_spec else [],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    args = ap.parse_args()
    path = Path(args.adapter)
    report = run(path if path.exists() else None)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
