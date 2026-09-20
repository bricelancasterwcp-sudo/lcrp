"""On-disk adapter digest + pin verification (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lcrp.apply import build_apply_spec
from lcrp.bank import (
    PinCache,
    digest_adapter_dir,
    load_adapter_patch,
    load_bank_from_manifest,
    verify_patch_digest,
)


def _fake_adapter(tmp: Path, name: str = "widget") -> Path:
    root = tmp / name
    root.mkdir()
    (root / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 8, "target_modules": ["q_proj"]}),
        encoding="utf-8",
    )
    (root / "adapter_model.safetensors").write_bytes(b"FAKE_ADAPTER_WEIGHTS_v1")
    return root


def test_digest_stable(tmp_path: Path) -> None:
    root = _fake_adapter(tmp_path)
    assert digest_adapter_dir(root) == digest_adapter_dir(root)
    assert len(digest_adapter_dir(root)) == 64


def test_digest_changes_when_weights_change(tmp_path: Path) -> None:
    root = _fake_adapter(tmp_path)
    before = digest_adapter_dir(root)
    (root / "adapter_model.safetensors").write_bytes(b"FAKE_ADAPTER_WEIGHTS_v2")
    assert digest_adapter_dir(root) != before


def test_pin_rejects_tampered_adapter(tmp_path: Path) -> None:
    root = _fake_adapter(tmp_path)
    patch = load_adapter_patch(
        patch_id="widget.api",
        adapter_path=root,
        domain="widget_api",
        embedding=(0.0, 1.0, 0.0, 0.0),
    )
    (root / "adapter_model.safetensors").write_bytes(b"TAMPERED")
    cache = PinCache(64, verify_digests=True)
    with pytest.raises(ValueError, match="digest"):
        cache.pin(patch, wait_ms=1.0)


def test_pin_and_apply_spec(tmp_path: Path) -> None:
    root = _fake_adapter(tmp_path)
    patch = load_adapter_patch(
        patch_id="widget.api",
        adapter_path=root,
        domain="widget_api",
        embedding=(0.0, 1.0, 0.0, 0.0),
    )
    verify_patch_digest(patch)
    cache = PinCache(64)
    assert cache.pin(patch, wait_ms=2.0) == []
    spec = build_apply_spec(cache, ["widget.api"])
    assert spec.adapter_paths == (str(root.resolve()),)
    assert spec.digests == (patch.digest,)
    assert spec.pinned_mib == patch.size_mib


def test_apply_spec_skips_evicted(tmp_path: Path) -> None:
    a = load_adapter_patch(
        patch_id="a",
        adapter_path=_fake_adapter(tmp_path, "a"),
        domain="widget_api",
        embedding=(1.0, 0.0, 0.0, 0.0),
        size_mib=40,
    )
    b = load_adapter_patch(
        patch_id="b",
        adapter_path=_fake_adapter(tmp_path, "b"),
        domain="widget_api",
        embedding=(0.0, 1.0, 0.0, 0.0),
        size_mib=40,
    )
    cache = PinCache(50)
    cache.pin(a, wait_ms=1.0)
    cache.pin(b, wait_ms=1.0)  # evicts a
    spec = build_apply_spec(cache, ["a", "b"])
    assert spec.patch_ids == ("b",)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    root = _fake_adapter(tmp_path)
    patch = load_adapter_patch(
        patch_id="widget.api",
        adapter_path=root,
        domain="widget_api",
        embedding=(0.0, 1.0, 0.0, 0.0),
    )
    man = tmp_path / "bank.json"
    man.write_text(
        json.dumps(
            {
                "patches": [
                    {
                        "patch_id": patch.patch_id,
                        "adapter_path": patch.adapter_path,
                        "domain": patch.domain,
                        "embedding": list(patch.embedding),
                        "digest": patch.digest,
                        "size_mib": patch.size_mib,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    bank = load_bank_from_manifest(man)
    assert bank["widget.api"].digest == patch.digest
