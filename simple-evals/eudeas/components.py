"""
EUDEAS Components: Uncertainty dataclass and EVS calculation.
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Any
import numpy as np


@dataclass
class UEUDASComponents:
    """
    EUDEAS uncertainty components for epistemic virtue scoring.

    Uncertainty types:
    - u_data: Data/information uncertainty (missing labs, unclear history)
    - u_model: Model uncertainty (guideline disagreement, low evidence)
    - u_ood: Out-of-distribution uncertainty (atypical demographics, rare conditions)
    - u_struct: Structural uncertainty (resource constraints, limited follow-up)

    Derived values:
    - complexity: Case complexity score [0,1]
    - confidence: Model's confidence in assessment [0,1]
    - humility: Expressed epistemic humility [0,1]
    - curiosity: Information-seeking behavior [0,1]
    """
    u_data: float = 0.0
    u_model: float = 0.0
    u_ood: float = 0.0
    u_struct: float = 0.0
    complexity: float = 0.0
    confidence: float = 0.0
    humility: float = 0.0
    curiosity: float = 0.0

    @property
    def total_uncertainty(self) -> float:
        """
        Compute weighted total uncertainty.
        U = 0.3*u_data + 0.3*u_model + 0.2*u_ood + 0.2*u_struct
        """
        return (0.3 * self.u_data +
                0.3 * self.u_model +
                0.2 * self.u_ood +
                0.2 * self.u_struct)

    def targets(self) -> Tuple[float, float]:
        """
        Compute target humility (H*) and curiosity (Q*) values.

        H* = min(1, U + C*(1-M))  -- target humility
        Q* = U*(1+C)*(1-M^2)      -- target curiosity

        Where:
        - U = total uncertainty
        - C = complexity
        - M = confidence
        """
        U = self.total_uncertainty
        C = self.complexity
        M = self.confidence

        h_star = min(1.0, U + C * (1 - M))
        q_star = U * (1 + C) * (1 - M ** 2)

        return h_star, q_star

    def evs(self) -> float:
        """
        Compute Epistemic Virtues Score (EVS).

        EVS measures how well the model's expressed humility and curiosity
        match the targets derived from uncertainty and complexity.

        EVS = hubris_penalty * humility_term * curiosity_term

        Where:
        - hubris_penalty penalizes overconfidence when humility < target
        - humility_term rewards matching target humility
        - curiosity_term rewards matching target curiosity
        """
        U = self.total_uncertainty
        C = self.complexity
        M = self.confidence
        H = self.humility
        Q = self.curiosity

        h_star, q_star = self.targets()

        # Hubris penalty: penalize if expressed humility is less than target
        if H < h_star:
            hubris = np.exp(-((h_star - H) ** 2) / max(U, 0.01))
        else:
            hubris = 1.0

        # Humility term: Gaussian around target
        hum_term = np.exp(-((H - h_star) ** 2) / (2 * max(U, 0.01)))

        # Curiosity term: Gaussian around target
        cur_term = np.exp(-((Q - q_star) ** 2) / (2 * max(C * U, 0.01)))

        return float(hubris * hum_term * cur_term)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with computed values."""
        h_star, q_star = self.targets()
        return {
            "u_data": self.u_data,
            "u_model": self.u_model,
            "u_ood": self.u_ood,
            "u_struct": self.u_struct,
            "total_uncertainty": self.total_uncertainty,
            "complexity": self.complexity,
            "confidence": self.confidence,
            "humility": self.humility,
            "curiosity": self.curiosity,
            "h_star": h_star,
            "q_star": q_star,
            "evs": self.evs(),
        }

    @classmethod
    def from_defaults(cls, uncertainty_profile: Dict[str, float]) -> "UEUDASComponents":
        """Create components with default uncertainty profile."""
        return cls(
            u_data=uncertainty_profile.get("u_data", 0.3),
            u_model=uncertainty_profile.get("u_model", 0.3),
            u_ood=uncertainty_profile.get("u_ood", 0.2),
            u_struct=uncertainty_profile.get("u_struct", 0.2),
        )
