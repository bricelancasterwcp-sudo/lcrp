"""LCRP — Logic-Core + Retrieved Parametric Patches."""

from lcrp.bank import PatchRecord, PinCache, make_toy_bank
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

__version__ = "0.1.2"

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
    "admit",
    "boot_core",
    "make_toy_bank",
    "missing_hot_path_events",
    "probe_core",
    "resolve_core_path",
    "score_patches",
    "__version__",
]
