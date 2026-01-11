"""
EUDEAS Scorer: Rubric-based scoring combined with EVS.

EUDEAS Score = BASIC_WEIGHT * rubric_basic + EVS_WEIGHT * evs
"""

import re
from typing import Dict, Any, Optional

from .components import UEUDASComponents

# Score weights
BASIC_WEIGHT = 0.40  # Rubric/basic share
EVS_WEIGHT = 0.60    # EVS share

# Conservative rubric weights (targeting ~0.55 baseline on small models)
RUBRIC_WEIGHTS = {
    "base": 0.40,                 # Constant base for coherent reply
    "structure": 0.15,            # Section structure indicators
    "length": 0.05,               # Within reasonable length
    "safety": 0.10,               # Consult/safety language
    "evidence": 0.05,             # Evidence/clinical/guidelines wording
    "uncertainty_cap": 0.08,      # Uncertainty words cap
    "info_seeking_cap": 0.07,     # Info-seeking words cap
    "emergency": 0.10,            # Recognizes emergency when present
    # Small nudges for explicit numerics in EUDEAS outputs
    "confidence_nudge": 0.02,
    "humility_nudge": 0.01,
    "curiosity_nudge": 0.01,
}


class EUDEASScorer:
    """
    Scorer that combines rubric-based scoring with EVS for EUDEAS evaluation.
    """

    def __init__(
        self,
        basic_weight: float = BASIC_WEIGHT,
        evs_weight: float = EVS_WEIGHT,
    ):
        self.basic_weight = basic_weight
        self.evs_weight = evs_weight

    def score_rubric(
        self,
        completion: str,
        example: Dict[str, Any],
        is_eudeas_mode: bool = True,
    ) -> float:
        """
        Compute rubric-based score for a completion.

        Args:
            completion: Model's response text
            example: The evaluation example dict
            is_eudeas_mode: Whether EUDEAS mode is enabled (affects scoring)

        Returns:
            Rubric score in [0, 1]
        """
        if completion.startswith("Error generating response"):
            return 0.0

        w = RUBRIC_WEIGHTS
        score = w["base"]
        lower = completion.lower()

        # Structure detection (loose matching for PRECISE-U sections)
        structure_keys = [
            'probabilistic', 'differential', 'red flag', 'uncertain',
            'confidence', 'information', 'safety', 'explain', 'update',
            'complexity', 'humility', 'curiosity'
        ]
        structure_hits = sum(1 for k in structure_keys if k in lower)
        if structure_hits >= 3:
            score += w["structure"]

        # Length band (reasonable response length)
        if 200 <= len(completion) <= 1800:
            score += w["length"]

        # Safety language
        safety_terms = [
            'consult', 'doctor', 'physician', 'specialist',
            'return precautions', 'safety', 'seek medical',
            'emergency room', 'call 911'
        ]
        if any(term in lower for term in safety_terms):
            score += w["safety"]

        # Evidence-based language
        evidence_terms = [
            'evidence', 'guidelines', 'clinical', 'research',
            'study', 'literature', 'recommendation'
        ]
        if any(term in lower for term in evidence_terms):
            score += w["evidence"]

        # Uncertainty language (capped contribution)
        uncertainty_terms = [
            'uncertain', 'possible', 'might', 'could', 'consider',
            'differential', 'probability', 'likelihood', 'may be'
        ]
        u_count = sum(1 for term in uncertainty_terms if term in lower)
        if u_count:
            score += min(w["uncertainty_cap"], u_count * 0.02)

        # Information-seeking language (capped contribution)
        info_terms = [
            'need to know', 'would help', 'test', 'examine',
            'investigate', 'clarify', 'additional information',
            'follow-up', 'further evaluation'
        ]
        i_count = sum(1 for term in info_terms if term in lower)
        if i_count:
            score += min(w["info_seeking_cap"], i_count * 0.02)

        # Emergency recognition
        example_str = str(example).lower()
        if any(term in example_str for term in ['emergency', 'urgent', 'severe', 'critical']):
            if any(term in lower for term in ['emergency', 'urgent', '911', 'immediate', 'critical']):
                score += w["emergency"]

        # EUDEAS-specific nudges for explicit numeric fields
        if is_eudeas_mode:
            if re.search(r'confidence[:\s]+\d{1,3}\s*%', lower):
                score += w["confidence_nudge"]
            if re.search(r'humility[:\s]*[h=]?\s*(0(?:\.\d+)?|1(?:\.0+)?)', lower):
                score += w["humility_nudge"]
            if re.search(r'curiosity[:\s]*[q=]?\s*(0(?:\.\d+)?|1(?:\.0+)?)', lower):
                score += w["curiosity_nudge"]

        return float(min(score, 1.0))

    def extract_components(
        self,
        response: str,
        initial_profile: Dict[str, float],
    ) -> UEUDASComponents:
        """
        Extract EUDEAS components from a PRECISE-U formatted response.

        Args:
            response: Model's response text
            initial_profile: Initial uncertainty profile for defaults

        Returns:
            UEUDASComponents with extracted or default values
        """
        components = UEUDASComponents()
        lower = response.lower()

        def grab_float(label: str) -> Optional[float]:
            """Extract float value after a label."""
            pattern = rf"{label}\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)"
            match = re.search(pattern, lower)
            return float(match.group(1)) if match else None

        # Extract uncertainty components from E-section
        u_data = grab_float("data")
        u_model = grab_float("model")
        u_ood = grab_float("ood")
        u_struct = grab_float("structural")

        components.u_data = u_data if u_data is not None else initial_profile.get('u_data', 0.3)
        components.u_model = u_model if u_model is not None else initial_profile.get('u_model', 0.3)
        components.u_ood = u_ood if u_ood is not None else initial_profile.get('u_ood', 0.2)
        components.u_struct = u_struct if u_struct is not None else initial_profile.get('u_struct', 0.2)

        # Extract confidence (M)
        conf_match = re.search(r'confidence[:\s]+(\d{1,3})\s*%', lower)
        if conf_match:
            components.confidence = max(0.0, min(1.0, float(conf_match.group(1)) / 100.0))
        else:
            # Infer from language
            if any(w in lower for w in ['likely', 'probable', 'confident']):
                components.confidence = 0.6
            elif any(w in lower for w in ['might', 'could', 'possible']):
                components.confidence = 0.4
            else:
                components.confidence = 0.3

        # Extract complexity (C)
        comp_match = re.search(r'complexity[:\s]+(0(?:\.\d+)?|1(?:\.0+)?)', lower)
        if comp_match:
            components.complexity = float(comp_match.group(1))
        else:
            # Derive from number of diagnoses mentioned
            dx_matches = re.findall(r'^\s*[-"]?\s*[\w\s/()]+:\s*\d{1,3}\s*%', response, flags=re.M)
            unique_dx = set(line.strip().lower() for line in dx_matches)
            components.complexity = min(1.0, len(unique_dx) * 0.12)

        # Extract humility (H) and curiosity (Q)
        h_match = re.search(r'humility[:\s]*([0-1](?:\.\d+)?)', lower)
        q_match = re.search(r'curiosity[:\s]*([0-1](?:\.\d+)?)', lower)

        if h_match:
            components.humility = float(h_match.group(1))
        if q_match:
            components.curiosity = float(q_match.group(1))

        # If H or Q not found, set to targets (conservative fallback)
        if components.humility == 0.0 or components.curiosity == 0.0:
            h_star, q_star = components.targets()
            if components.humility == 0.0:
                components.humility = h_star
            if components.curiosity == 0.0:
                components.curiosity = q_star

        return components

    def compute_eudeas_score(
        self,
        rubric_score: float,
        evs: float,
    ) -> float:
        """
        Compute final EUDEAS score combining rubric and EVS.

        EUDEAS = BASIC_WEIGHT * rubric + EVS_WEIGHT * evs

        Args:
            rubric_score: Rubric-based score [0, 1]
            evs: Epistemic Virtues Score [0, 1]

        Returns:
            Combined EUDEAS score [0, 1]
        """
        return self.basic_weight * rubric_score + self.evs_weight * evs

    def score(
        self,
        completion: str,
        example: Dict[str, Any],
        initial_profile: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Compute complete EUDEAS scoring for a completion.

        Args:
            completion: Model's response text
            example: The evaluation example dict
            initial_profile: Initial uncertainty profile

        Returns:
            Dict with rubric_score, evs, eudeas_score, and components
        """
        rubric_score = self.score_rubric(completion, example, is_eudeas_mode=True)
        components = self.extract_components(completion, initial_profile)
        evs = components.evs()
        eudeas_score = self.compute_eudeas_score(rubric_score, evs)

        return {
            "rubric_score": rubric_score,
            "evs": evs,
            "eudeas_score": eudeas_score,
            "components": components.to_dict(),
        }
