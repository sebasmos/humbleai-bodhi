"""
EUDEAS: Epistemic Uncertainty-Driven Evaluation and Scoring

This module implements the EUDEAS framework for uncertainty-aware medical AI evaluation,
including PRECISE-U prompting and EVS (Epistemic Virtues Score) calculation.
"""

from .components import UEUDASComponents
from .template import PRECISEUTemplate
from .sampler_wrapper import EUDEASSamplerWrapper
from .scorer import EUDEASScorer, BASIC_WEIGHT, EVS_WEIGHT

__all__ = [
    "UEUDASComponents",
    "PRECISEUTemplate",
    "EUDEASSamplerWrapper",
    "EUDEASScorer",
    "BASIC_WEIGHT",
    "EVS_WEIGHT",
]
