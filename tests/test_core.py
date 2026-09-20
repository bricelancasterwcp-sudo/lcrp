"""Core wiring tests — no GPU, no network, weights not required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lcrp.campaign import CoreDtype
from lcrp.core import (
    A_PRIME_CORE,
    ENV_CORE_PATH,
    CoreSpec,
    boot_core,
    probe_core,
    resolve_core_path,
)
from lcrp.journal import Journal
from lcrp.loop import SegmentLoop


def test_a_prime_core_defaults() -> None:
    assert A_PRIME_CORE.model_id == "RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic"
    assert (
        A_PRIME_CORE.local_path
        == "/mnt/extra/models/Qwen2.5-7B-Instruct-FP8-dynamic"
    )
    assert A_PRIME_CORE.dtype is CoreDtype.FP8
    assert 7500 <= A_PRIME_CORE.approx_weights_mib <= 8200
    assert A_PRIME_CORE.notes


def test_resolve_falls_back_to_model_id_when_local_missing() -> None:
    # Use a local_path that cannot exist so this passes on machines that
    # already have the blessed /mnt/extra checkout stubbed or downloaded.
    spec = CoreSpec(
        model_id="RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic",
        local_path="/no/such/lcrp/core/local",
        dtype=CoreDtype.FP8,
        approx_weights_mib=7800,
        notes="test",
    )
    resolved = resolve_core_path(spec, env={})
    assert resolved == spec.model_id


def test_resolve_env_override(tmp_path: Path) -> None:
    override = str(tmp_path / "custom-core")
    resolved = resolve_core_path(
        A_PRIME_CORE, env={ENV_CORE_PATH: override}
    )
    assert resolved == override


def test_resolve_uses_existing_local_path(tmp_path: Path) -> None:
    local = tmp_path / "Qwen2.5-7B-Instruct-FP8-dynamic"
    local.mkdir()
    spec = CoreSpec(
        model_id="ignored/id",
        local_path=str(local),
        dtype=CoreDtype.FP8,
        approx_weights_mib=7800,
        notes="test",
    )
    assert resolve_core_path(spec, env={}) == str(local)


def test_probe_missing_path() -> None:
    result = probe_core("/no/such/lcrp/core/path")
    assert result.ok is False
    assert result.detail == "path missing"
    assert result.model_type is None


def test_probe_reads_config_json(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                },
            }
        ),
        encoding="utf-8",
    )
    result = probe_core(str(root))
    assert result.ok is True
    assert result.model_type == "qwen2"
    assert result.architectures == ("Qwen2ForCausalLM",)
    assert result.quantization_summary is not None
    assert "quant_method=fp8" in result.quantization_summary


def test_boot_core_journals_without_weights() -> None:
    journal = Journal()
    boot_core(journal, A_PRIME_CORE, campaign="A_prime")
    assert len(journal.events) == 1
    event = journal.events[0]
    assert event["kind"] == "boot"
    assert event["model_id"] == A_PRIME_CORE.model_id
    assert event["core_dtype"] == "fp8"
    assert event["core_weights_mib"] == A_PRIME_CORE.approx_weights_mib
    assert event["campaign"] == "A_prime"


def test_segment_loop_boot_includes_model_id() -> None:
    loop = SegmentLoop(
        core_dtype=CoreDtype.FP8,
        core_weights_mib=8_000,
        kv_mib=500,
        overhead_mib=200,
        tau=10.0,
    )
    boots = [e for e in loop.journal.events if e["kind"] == "boot"]
    assert len(boots) == 1
    assert boots[0]["model_id"] == A_PRIME_CORE.model_id
    assert boots[0]["core_weights_mib"] == 8_000


@pytest.mark.skipif(
    not Path("/mnt/extra/models/Qwen2.5-7B-Instruct-FP8-dynamic").exists(),
    reason="optional local FP8 core weights not present",
)
def test_optional_probe_mounted_extra_core() -> None:
    path = "/mnt/extra/models/Qwen2.5-7B-Instruct-FP8-dynamic"
    result = probe_core(path)
    assert result.ok is True
    assert result.model_type is not None
