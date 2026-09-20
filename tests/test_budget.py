from lcrp.budget import ResidentBudget, admit
from lcrp.campaign import CoreDtype


def _ok(**over):
    base = dict(
        total_mib=16_384,
        core_weights_mib=8_000,
        pinned_patches_mib=512,
        kv_mib=2_000,
        overhead_mib=500,
    )
    base.update(over)
    return ResidentBudget(**base)


def test_admit_happy_path() -> None:
    result = admit(_ok(), core_dtype=CoreDtype.FP8)
    assert result.admitted and result.refusal is None


def test_refuse_unmeasured() -> None:
    result = admit(_ok(kv_mib=None), core_dtype=CoreDtype.FP8)
    assert not result.admitted and result.refusal is not None
    assert result.refusal.blocker == "unmeasured"


def test_refuse_pretend_measured() -> None:
    result = admit(_ok(), core_dtype=CoreDtype.FP8, pretend_measured=True)
    assert not result.admitted and result.refusal is not None
    assert result.refusal.blocker == "unmeasured"


def test_refuse_core_over_cap() -> None:
    result = admit(_ok(core_weights_mib=9_500), core_dtype=CoreDtype.INT8)
    assert not result.admitted and result.refusal is not None
    assert result.refusal.blocker == "core"


def test_refuse_pins_over_cap() -> None:
    result = admit(_ok(pinned_patches_mib=2_000), core_dtype=CoreDtype.FP8)
    assert not result.admitted and result.refusal is not None
    assert result.refusal.blocker == "pins"


def test_refuse_resident_over_cap() -> None:
    result = admit(
        _ok(core_weights_mib=9_000, pinned_patches_mib=1_024, kv_mib=4_000),
        core_dtype=CoreDtype.FP8,
    )
    assert not result.admitted and result.refusal is not None
    assert result.refusal.blocker == "resident"


def test_refuse_bad_dtype() -> None:
    result = admit(_ok(), core_dtype=None)
    assert not result.admitted and result.refusal is not None
    assert result.refusal.blocker == "core_dtype"


def test_free_none_when_unmeasured() -> None:
    assert _ok(overhead_mib=None).free_mib() is None
