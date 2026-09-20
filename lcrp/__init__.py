"""LCRP — Logic-Core + Retrieved Parametric Patches."""

from lcrp.bank import PatchRecord, PinCache, make_toy_bank
from lcrp.budget import AdmissionResult, RefusalArithmetic, ResidentBudget, admit
from lcrp.campaign import A_PRIME, CampaignAPrime, CoreDtype
from lcrp.journal import Journal, missing_hot_path_events
from lcrp.loop import SegmentLoop, SegmentResult, score_patches
from lcrp.router import PatchScore, RouteDecision, ScoreRouter

__version__ = "0.1.1"

__all__ = [
    "A_PRIME",
    "AdmissionResult",
    "CampaignAPrime",
    "CoreDtype",
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
    "make_toy_bank",
    "missing_hot_path_events",
    "score_patches",
    "__version__",
]
