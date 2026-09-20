from lcrp.journal import Journal, missing_hot_path_events


def test_missing_on_pin_path() -> None:
    j = Journal()
    j.record("boot")
    j.record("route", segment_id="s1")
    assert missing_hot_path_events(j.events, core_only=False) == [
        "prefetch",
        "pin",
        "apply",
    ]


def test_core_only_path_ok() -> None:
    j = Journal()
    j.record("boot")
    j.record("route", core_only=True)
    assert missing_hot_path_events(j.events, core_only=True) == []


def test_jsonl_roundtrip(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    j = Journal(path=path)
    j.record("boot", digest="abc")
    j.record("route", segment_id="s1")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert '"kind": "boot"' in lines[0]
