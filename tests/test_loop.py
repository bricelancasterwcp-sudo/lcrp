from lcrp.campaign import CoreDtype
from lcrp.journal import missing_hot_path_events
from lcrp.loop import SegmentLoop


def _loop(**over) -> SegmentLoop:
    base = dict(
        core_dtype=CoreDtype.FP8,
        core_weights_mib=8_000,
        kv_mib=1_500,
        overhead_mib=400,
        tau=0.3,
        prefetch_wait_ms=4.0,
    )
    base.update(over)
    return SegmentLoop(**base)


def test_pin_path_journals_hot_events() -> None:
    loop = _loop()
    # Query aligned with code.python embedding
    result = loop.run_segment("s1", (0.0, 1.0, 0.0, 0.0))
    assert not result.refused
    assert not result.route.core_only
    assert result.applied
    assert missing_hot_path_events(loop.journal.events, core_only=False) == []


def test_core_only_path() -> None:
    loop = _loop(tau=10.0)  # impossible threshold → core-only
    result = loop.run_segment("s2", (0.0, 1.0, 0.0, 0.0))
    assert not result.refused
    assert result.route.core_only
    assert result.applied == ()
    assert missing_hot_path_events(loop.journal.events, core_only=True) == []


def test_refuse_when_core_over_cap() -> None:
    loop = _loop(core_weights_mib=9_500)
    result = loop.run_segment("s3", (0.0, 1.0, 0.0, 0.0))
    assert result.refused
    assert result.admission.refusal is not None
    assert result.admission.refusal.blocker == "core"
    kinds = [e["kind"] for e in loop.journal.events]
    assert "refuse" in kinds


def test_lru_evicts_under_pin_pressure() -> None:
    # Tiny pin budget forces eviction across domains
    from lcrp.campaign import A_PRIME, CampaignAPrime

    tiny = CampaignAPrime(max_pins_mib=100)  # one ~96 MiB code patch fits
    loop = SegmentLoop(
        core_dtype=CoreDtype.FP8,
        core_weights_mib=8_000,
        kv_mib=500,
        overhead_mib=200,
        tau=0.1,
        campaign=tiny,
    )
    r1 = loop.run_segment("a", (0.0, 1.0, 0.0, 0.0))  # code
    assert not r1.refused
    r2 = loop.run_segment("b", (1.0, 0.0, 0.0, 0.0))  # law
    assert not r2.refused
    # After second pin, first code patch may be evicted
    assert r2.evicted or set(r2.applied).isdisjoint(set(r1.applied))
