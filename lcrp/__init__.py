"""LCRP — Logic-Core + Retrieved Parametric Patches."""

from lcrp.budget import AdmissionResult, RefusalArithmetic, ResidentBudget, admit
from lcrp.campaign import A_PRIME, CampaignAPrime, CoreDtype
from lcrp.journal import Journal, missing_hot_path_events
from lcrp.router import PatchScore, RouteDecision, ScoreRouter

__version__ = "0.1.0"

__all__ = [
    "A_PRIME",
    "AdmissionResult",
    "CampaignAPrime",
    "CoreDtype",
    "Journal",
    "PatchScore",
    "RefusalArithmetic",
    "ResidentBudget",
    "RouteDecision",
    "ScoreRouter",
    "admit",
    "missing_hot_path_events",
    "__version__",
]
