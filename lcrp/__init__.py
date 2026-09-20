"""LCRP — Logic-Core + Retrieved Parametric Patches."""

from lcrp.apply import ApplySpec, build_apply_spec
from lcrp.bank import (
    PatchRecord,
    PinCache,
    digest_adapter_dir,
    load_adapter_patch,
    load_bank_from_manifest,
    make_toy_bank,
    verify_patch_digest,
)
from lcrp.budget import AdmissionResult, RefusalArithmetic, ResidentBudget, admit
from lcrp.campaign import A_PRIME, CampaignAPrime, CoreDtype
from lcrp.core import (
    A_PRIME_CORE,
    ENV_CORE_PATH,
    CoreProbe,
    CoreSpec,
    boot_core,
    probe_core,
    resolve_core_path,
)
from lcrp.journal import Journal, missing_hot_path_events
from lcrp.loop import SegmentLoop, SegmentResult, score_patches
from lcrp.router import PatchScore, RouteDecision, ScoreRouter

__version__ = "0.1.3"

__all__ = [
    "A_PRIME",
    "A_PRIME_CORE",
    "AdmissionResult",
    "CampaignAPrime",
    "CoreDtype",
    "CoreProbe",
    "CoreSpec",
    "ENV_CORE_PATH",
    "Journal",
    "PatchRecord",
    "PatchScore",
    "PinCache",
    "RefusalArithmetic",
    "ResidentBudget",
    "RouteDecision",
    "ScoreRouter",
    "SegmentLoop",
    "SegmentResult",
    "ApplySpec",
    "build_apply_spec",
    "digest_adapter_dir",
    "load_adapter_patch",
    "load_bank_from_manifest",
    "verify_patch_digest",
    "admit",
    "boot_core",
    "make_toy_bank",
    "missing_hot_path_events",
    "probe_core",
    "resolve_core_path",
    "score_patches",
    "__version__",
]
