from pathlib import Path

from lcrp.campaign import A_PRIME

GATES = Path(__file__).resolve().parents[1] / "docs" / "gates.md"


def test_a_prime_numbers() -> None:
    text = GATES.read_text(encoding="utf-8")
    assert "14" in text and "000" in text
    assert "9" in text and "000" in text
    assert "1" in text and "024" in text
    assert A_PRIME.max_resident_mib == 14_000
    assert A_PRIME.max_core_mib == 9_000
    assert A_PRIME.max_pins_mib == 1_024
    assert A_PRIME.max_prefetch_wait_ms_p95 == 20.0
    assert A_PRIME.top_k == 4
    assert A_PRIME.min_oracle_lift_pp == 15.0
    assert A_PRIME.max_router_oracle_gap_pp == 5.0
    assert A_PRIME.n_standard_cell == 100
    assert A_PRIME.n_long_context_cell == 30
    assert A_PRIME.required_arms == ("core_only", "oracle_patches", "router")
