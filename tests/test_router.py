from lcrp.router import PatchScore, ScoreRouter


def test_core_only_below_tau() -> None:
    decision = ScoreRouter(tau=0.5).route(
        "seg-1",
        [PatchScore("a", 0.2), PatchScore("b", 0.1)],
    )
    assert decision.core_only
    assert decision.top_k == ()


def test_top_k_ordering_and_cap() -> None:
    scores = [
        PatchScore("d", 0.4),
        PatchScore("a", 0.9),
        PatchScore("c", 0.5),
        PatchScore("b", 0.8),
        PatchScore("e", 0.7),
    ]
    decision = ScoreRouter(tau=0.1).route("seg-2", scores)
    assert not decision.core_only
    assert [p.patch_id for p in decision.top_k] == ["a", "b", "e", "c"]
