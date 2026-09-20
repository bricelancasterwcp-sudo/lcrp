from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "docs/architecture.md",
    "docs/bloomery-accounting.md",
    "docs/assay-eval-plan.md",
    "docs/sensorium-harness.md",
    "docs/borrowed-laws.md",
    "docs/gates.md",
    "docs/phase1.md",
]


def test_required_docs_exist() -> None:
    missing = [p for p in REQUIRED if not (ROOT / p).is_file()]
    assert not missing, f"missing required docs: {missing}"
