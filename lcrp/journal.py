"""Append-only decision journal (Phase 0 stub).

Design authority: docs/bloomery-accounting.md, docs/sensorium-harness.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Journal:
    """In-memory journal for prototypes; replace with durable JSONL later."""

    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, kind: str, **payload: Any) -> None:
        self.events.append({"kind": kind, **payload})
