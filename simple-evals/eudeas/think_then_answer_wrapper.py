"""
Think-Then-Answer Sampler Wrapper with Calibrated Confidence Steering.

Uses H* (target humility) and Q* (target curiosity) to calibrate response style.
"""

import json
from typing import Any, Dict, List, Union

# Handle imports for both package and standalone usage
try:
    from ..types import MessageList, SamplerBase, SamplerResponse
except ImportError:
    MessageList = List[Dict[str, Any]]
    from typing import Protocol

    class SamplerBase(Protocol):
        def __call__(self, message_list: Any) -> Any: ...

    class SamplerResponse:
        def __init__(self, response_text: str, actual_queried_message_list: Any, response_metadata: Dict[str, Any]):
            self.response_text = response_text
            self.actual_queried_message_list = actual_queried_message_list
            self.response_metadata = response_metadata

from .think_then_answer import ThinkThenAnswerTemplate


class ThinkThenAnswerWrapper(SamplerBase):
    """
    Wrapper that uses Think-Then-Answer prompting with Calibrated Confidence Steering.

    Modes:
    - single-pass: Basic internal reasoning prompt
    - two-pass: Analysis first, then calibrated response
    - two-pass-v2: More aggressive calibration with explicit behavioral instructions
    """

    def __init__(
        self,
        base_sampler: SamplerBase,
        two_pass: bool = False,
        calibration_version: int = 1,
    ):
        """
        Initialize the Think-Then-Answer wrapper.

        Args:
            base_sampler: The underlying sampler to wrap
            two_pass: If True, use two-pass mode (analysis then response)
            calibration_version: 1 for natural guidance, 2 for explicit behavioral instructions
        """
        self.base_sampler = base_sampler
        self.template = ThinkThenAnswerTemplate()
        self.two_pass = two_pass
        self.calibration_version = calibration_version

    def _pack_message(self, role: str, content: Any) -> Dict[str, Any]:
        """Create a message dict."""
        return {"role": str(role), "content": content}

    def _chat_to_text(self, messages: Union[str, List[Dict[str, Any]]]) -> str:
        """Convert chat messages to plain text."""
        if isinstance(messages, str):
            return messages.strip()

        if isinstance(messages, list):
            parts = []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(str(part.get("text", "")))
                elif isinstance(content, str):
                    parts.append(content)
            text = "\n".join([p for p in parts if p]).strip()
            return text if text else json.dumps(messages, ensure_ascii=False)

        return str(messages)

    def _analyze_uncertainty(self, text: str) -> Dict[str, float]:
        """
        Analyze input text to estimate uncertainty profile.
        """
        profile = {
            'u_data': 0.3,
            'u_model': 0.3,
            'u_ood': 0.2,
            'u_struct': 0.2,
        }

        lower = text.lower()

        # Data uncertainty indicators
        if any(word in lower for word in ['unknown', 'unclear', 'missing', 'limited', 'not provided', 'n/a']):
            profile['u_data'] = min(0.7, profile['u_data'] + 0.3)

        # Model uncertainty indicators
        if any(word in lower for word in ['rare', 'unusual', 'atypical', 'complex', 'controversial']):
            profile['u_model'] = min(0.7, profile['u_model'] + 0.2)
            profile['u_ood'] = min(0.7, profile['u_ood'] + 0.2)

        # Structural uncertainty indicators
        if any(word in lower for word in ['rural', 'limited resources', 'urgent', 'emergency', 'no follow-up']):
            profile['u_struct'] = min(0.7, profile['u_struct'] + 0.2)

        return profile

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        """
        Process messages with Think-Then-Answer enhancement.
        """
        # Extract case text
        case_text = self._chat_to_text(message_list)

        # Truncate if too long
        if len(case_text) > 2000:
            case_text = case_text[-2000:]

        # Analyze uncertainty
        uncertainty_profile = self._analyze_uncertainty(case_text)

        if self.two_pass:
            return self._two_pass_call(case_text, uncertainty_profile)
        else:
            return self._single_pass_call(case_text, uncertainty_profile)

    def _single_pass_call(self, case_text: str, uncertainty_profile: Dict[str, float]) -> SamplerResponse:
        """Single-pass: internal reasoning with clean output."""
        enhanced_prompt = self.template.render(case_text, uncertainty_profile)
        enhanced_messages = [self._pack_message("user", enhanced_prompt)]
        response = self.base_sampler(enhanced_messages)

        metadata = response.response_metadata.copy()
        metadata["think_then_answer"] = {
            "enabled": True,
            "mode": "single_pass",
            "uncertainty_profile": uncertainty_profile,
        }

        return SamplerResponse(
            response_text=response.response_text,
            actual_queried_message_list=enhanced_messages,
            response_metadata=metadata,
        )

    def _two_pass_call(self, case_text: str, uncertainty_profile: Dict[str, float]) -> SamplerResponse:
        """Two-pass: analysis first, then calibrated response."""
        # Pass 1: Get analysis
        if self.calibration_version == 7:
            analysis_prompt = self.template.render_v7_analysis(case_text, uncertainty_profile)
        elif self.calibration_version == 6:
            analysis_prompt = self.template.render_curious_humble_analysis(case_text, uncertainty_profile)
        elif self.calibration_version == 5:
            analysis_prompt = self.template.render_two_pass_analysis_v5(case_text, uncertainty_profile)
        elif self.calibration_version == 0:
            # v0 = simple analysis (original successful version)
            analysis_prompt = self.template.render_simple_analysis(case_text, uncertainty_profile)
        else:
            analysis_prompt = self.template.render_two_pass_analysis(case_text, uncertainty_profile)

        analysis_messages = [self._pack_message("user", analysis_prompt)]
        analysis_response = self.base_sampler(analysis_messages)
        analysis_text = analysis_response.response_text

        # Extract values and calculate calibration targets (for metadata)
        values = self.template.extract_analysis_values(analysis_text)
        U, h_star, q_star = self.template.calculate_targets(
            values['u_data'], values['u_model'], values['u_ood'], values['u_struct'],
            values['complexity'], values['confidence']
        )

        # Pass 2: Generate response
        if self.calibration_version == 7:
            response_prompt = self.template.render_v7_response(case_text, analysis_text)
        elif self.calibration_version == 6:
            response_prompt = self.template.render_curious_humble_response(case_text, analysis_text)
        elif self.calibration_version == 5:
            response_prompt = self.template.render_two_pass_response_v5(case_text, analysis_text)
        elif self.calibration_version == 0:
            response_prompt = self.template.render_simple_response(case_text, analysis_text)
        elif self.calibration_version == 4:
            response_prompt = self.template.render_two_pass_response_v4(case_text, analysis_text)
        elif self.calibration_version == 3:
            response_prompt = self.template.render_two_pass_response_v3(case_text, analysis_text)
        elif self.calibration_version == 2:
            response_prompt = self.template.render_two_pass_response_v2(case_text, analysis_text)
        else:
            response_prompt = self.template.render_two_pass_response(case_text, analysis_text)

        response_messages = [self._pack_message("user", response_prompt)]
        final_response = self.base_sampler(response_messages)

        # Build metadata
        metadata = final_response.response_metadata.copy()
        metadata["think_then_answer"] = {
            "enabled": True,
            "mode": f"two_pass_v{self.calibration_version}",
            "uncertainty_profile": uncertainty_profile,
            "extracted_values": values,
            "calibration": {
                "total_uncertainty": U,
                "h_star": h_star,
                "q_star": q_star,
            },
            "analysis": analysis_text,
        }

        return SamplerResponse(
            response_text=final_response.response_text,
            actual_queried_message_list=response_messages,
            response_metadata=metadata,
        )

    def get_base_sampler(self) -> SamplerBase:
        """Return the underlying base sampler."""
        return self.base_sampler
