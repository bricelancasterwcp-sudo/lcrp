"""Patch bank + pin cache for Phase 1 loops.

Supports synthetic toy patches and on-disk LoRA adapters with digest checks.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PatchRecord:
    patch_id: str
    size_mib: int
    digest: str
    domain: str
    # Fixed embedding used by the score router (dot with query).
    embedding: tuple[float, ...]
    # Optional on-disk LoRA/adapter directory (PEFT layout).
    adapter_path: str | None = None


def _synthetic_digest(patch_id: str, size_mib: int, domain: str) -> str:
    raw = f"{patch_id}:{size_mib}:{domain}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


_DIGEST_SKIP_NAMES = frozenset(
    {
        "MANIFEST.json",
        "bank_manifest.json",
        "train.jsonl",
        "eval.jsonl",
        "README.md",
        "three_arm_generate.json",
        "chat_template.jinja",
    }
)
_DIGEST_SKIP_DIR_PARTS = frozenset({"runs", "checkpoint", ".cache"})


def digest_adapter_dir(path: str | Path) -> str:
    """SHA-256 over adapter weight files (skips manifests, datasets, runs/)."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"adapter path missing: {root}")
    h = hashlib.sha256()
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.name not in _DIGEST_SKIP_NAMES
        and not any(part in _DIGEST_SKIP_DIR_PARTS for part in p.parts)
        and not p.name.endswith(".incomplete")
    )
    if not files:
        raise ValueError(f"adapter dir has no weight files: {root}")
    for f in files:
        rel = f.relative_to(root).as_posix().encode()
        h.update(rel)
        h.update(b"\0")
        h.update(f.read_bytes())
    return h.hexdigest()


def verify_patch_digest(patch: PatchRecord) -> None:
    """Raise ValueError if on-disk adapter digest does not match the record."""
    if not patch.adapter_path:
        return
    actual = digest_adapter_dir(patch.adapter_path)
    if actual != patch.digest and actual[:16] != patch.digest:
        # Allow full or truncated digests in records.
        if not (patch.digest and actual.startswith(patch.digest)):
            raise ValueError(
                f"digest mismatch for {patch.patch_id}: "
                f"record={patch.digest} actual={actual}"
            )


def make_toy_bank() -> dict[str, PatchRecord]:
    """Small deterministic synthetic bank spanning a few domains."""
    specs = [
        ("law.torts", 64, "law", (1.0, 0.0, 0.0, 0.1)),
        ("law.contracts", 64, "law", (0.9, 0.1, 0.0, 0.0)),
        ("code.python", 96, "code", (0.0, 1.0, 0.0, 0.0)),
        ("code.rust", 96, "code", (0.1, 0.9, 0.0, 0.0)),
        ("med.cardio", 80, "med", (0.0, 0.0, 1.0, 0.0)),
        ("med.neuro", 80, "med", (0.0, 0.1, 0.9, 0.0)),
        ("gen.trivia", 48, "gen", (0.2, 0.2, 0.2, 1.0)),
        ("gen.history", 48, "gen", (0.1, 0.1, 0.1, 0.9)),
    ]
    bank: dict[str, PatchRecord] = {}
    for patch_id, size_mib, domain, emb in specs:
        bank[patch_id] = PatchRecord(
            patch_id=patch_id,
            size_mib=size_mib,
            digest=_synthetic_digest(patch_id, size_mib, domain),
            domain=domain,
            embedding=emb,
        )
    return bank


def load_adapter_patch(
    *,
    patch_id: str,
    adapter_path: str | Path,
    domain: str,
    embedding: tuple[float, ...],
    size_mib: int | None = None,
) -> PatchRecord:
    """Build a PatchRecord from an on-disk PEFT adapter directory."""
    path = Path(adapter_path)
    digest = digest_adapter_dir(path)
    if size_mib is None:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        size_mib = max(1, (total + 1024 * 1024 - 1) // (1024 * 1024))
    return PatchRecord(
        patch_id=patch_id,
        size_mib=int(size_mib),
        digest=digest,
        domain=domain,
        embedding=embedding,
        adapter_path=str(path.resolve()),
    )


def load_bank_from_manifest(manifest_path: str | Path) -> dict[str, PatchRecord]:
    """Load patches listed in a JSON manifest (list or {patches: [...]})."""
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    items = data["patches"] if isinstance(data, dict) and "patches" in data else data
    bank: dict[str, PatchRecord] = {}
    for item in items:
        emb = tuple(float(x) for x in item["embedding"])
        patch = load_adapter_patch(
            patch_id=item["patch_id"],
            adapter_path=item["adapter_path"],
            domain=item["domain"],
            embedding=emb,
            size_mib=item.get("size_mib"),
        )
        if "digest" in item and item["digest"] not in (patch.digest, patch.digest[:16]):
            if not patch.digest.startswith(str(item["digest"])):
                raise ValueError(
                    f"manifest digest mismatch for {patch.patch_id}: "
                    f"manifest={item['digest']} actual={patch.digest}"
                )
        bank[patch.patch_id] = patch
    return bank


@dataclass
class PinSlot:
    patch: PatchRecord
    wait_ms: float


class PinCache:
    """Simple LRU pin cache charged in MiB, with optional digest verification."""

    def __init__(self, max_mib: int, *, verify_digests: bool = True) -> None:
        self.max_mib = max_mib
        self.verify_digests = verify_digests
        self._slots: OrderedDict[str, PinSlot] = OrderedDict()

    @property
    def used_mib(self) -> int:
        return sum(s.patch.size_mib for s in self._slots.values())

    def pinned_ids(self) -> tuple[str, ...]:
        return tuple(self._slots.keys())

    def get(self, patch_id: str) -> PinSlot | None:
        slot = self._slots.get(patch_id)
        if slot is not None:
            self._slots.move_to_end(patch_id)
        return slot

    def pin(self, patch: PatchRecord, *, wait_ms: float) -> list[str]:
        """Pin patch; return list of evicted patch ids."""
        if self.verify_digests:
            verify_patch_digest(patch)
        if patch.patch_id in self._slots:
            self._slots.move_to_end(patch.patch_id)
            self._slots[patch.patch_id] = PinSlot(patch=patch, wait_ms=wait_ms)
            return []
        evicted: list[str] = []
        while self._slots and self.used_mib + patch.size_mib > self.max_mib:
            eid, _ = self._slots.popitem(last=False)
            evicted.append(eid)
        if self.used_mib + patch.size_mib > self.max_mib:
            raise MemoryError(
                f"cannot pin {patch.patch_id}: "
                f"{patch.size_mib} MiB > cache max {self.max_mib}"
            )
        self._slots[patch.patch_id] = PinSlot(patch=patch, wait_ms=wait_ms)
        return evicted
