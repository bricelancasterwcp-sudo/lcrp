"""Append-only decision journal.

Design authority: docs/bloomery-accounting.md, docs/sensorium-harness.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

SCORED_PIN_PATH = ("boot", "route", "prefetch", "pin", "apply")
SCORED_CORE_ONLY_PATH = ("boot", "route")


@dataclass
class Journal:
    events: list[dict[str, Any]] = field(default_factory=list)
    path: Path | None = None

    def record(self, kind: str, **payload: Any) -> None:
        event = {"kind": kind, **payload}
        self.events.append(event)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, sort_keys=True) + "\n")


def missing_hot_path_events(
    events: Iterable[dict[str, Any]],
    *,
    core_only: bool,
) -> list[str]:
    """Return missing required kinds for G-journal on a scored run."""
    seen = {e.get("kind") for e in events}
    required = SCORED_CORE_ONLY_PATH if core_only else SCORED_PIN_PATH
    return [k for k in required if k not in seen]
