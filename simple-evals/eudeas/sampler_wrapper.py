"""
EUDEAS Sampler Wrapper: Wraps any SamplerBase to add PRECISE-U prompting.
"""

import json
from typing import Any, Dict, List, Union

# Handle imports for both package and standalone usage
try:
    from ..types import MessageList, SamplerBase, SamplerResponse
except ImportError:
    # Fallback for standalone testing - define minimal types
    MessageList = List[Dict[str, Any]]
    from typing import Protocol

    class SamplerBase(Protocol):
        def __call__(self, message_list: Any) -> Any: ...

    class SamplerResponse:
        def __init__(self, response_text: str, actual_queried_message_list: Any, response_metadata: Dict[str, Any]):
            self.response_text = response_text
            self.actual_queried_message_list = actual_queried_message_list
            self.response_metadata = response_metadata

from .template import PRECISEUTemplate
from .scorer import EUDEASScorer
from .components import UEUDASComponents


class EUDEASSamplerWrapper(SamplerBase):
    """
    Wrapper that adds EUDEAS (PRECISE-U + EVS) functionality to any sampler.

    This wrapper:
    1. Analyzes the input for uncertainty indicators
    2. Renders a PRECISE-U structured prompt
    3. Calls the base sampler with the enhanced prompt
    4. Extracts EUDEAS components from the response
    5. Calculates EVS and EUDEAS scores
    6. Returns SamplerResponse with eudeas metadata
    """

    def __init__(
        self,
        base_sampler: SamplerBase,
        scorer: EUDEASScorer | None = None,
    ):
        """
        Initialize the EUDEAS wrapper.

        Args:
            base_sampler: The underlying sampler to wrap
            scorer: Optional custom scorer (uses default if None)
        """
        self.base_sampler = base_sampler
        self.template = PRECISEUTemplate()
        self.scorer = scorer or EUDEASScorer()

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
                    # Handle multimodal content
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
        Analyze input text to estimate initial uncertainty profile.

        Args:
            text: Input case/question text

        Returns:
            Dict with u_data, u_model, u_ood, u_struct estimates
        """
        profile = {
            'u_data': 0.3,
            'u_model': 0.3,
            'u_ood': 0.2,
            'u_struct': 0.2,
        }

        lower = text.lower()

        # Data uncertainty indicators
        data_indicators = [
            'unknown', 'unclear', 'missing', 'limited',
            'not provided', 'n/a', 'unavailable', 'pending'
        ]
        if any(word in lower for word in data_indicators):
            profile['u_data'] = min(0.7, profile['u_data'] + 0.3)

        # Model uncertainty indicators
        model_indicators = [
            'rare', 'unusual', 'atypical', 'complex',
            'controversial', 'debated', 'conflicting'
        ]
        if any(word in lower for word in model_indicators):
            profile['u_model'] = min(0.7, profile['u_model'] + 0.2)
            profile['u_ood'] = min(0.7, profile['u_ood'] + 0.2)

        # Structural uncertainty indicators
        struct_indicators = [
            'rural', 'limited resources', 'urgent', 'emergency',
            'no follow-up', 'remote', 'constrained'
        ]
        if any(word in lower for word in struct_indicators):
            profile['u_struct'] = min(0.7, profile['u_struct'] + 0.2)

        return profile

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        """
        Process messages with EUDEAS enhancement.

        Args:
            message_list: Input messages

        Returns:
            SamplerResponse with EUDEAS metadata
        """
        # Extract case text from messages
        case_text = self._chat_to_text(message_list)

        # Truncate if too long (keep last 1600 chars for context)
        if len(case_text) > 1600:
            case_text = case_text[-1600:]

        # Analyze uncertainty from input
        uncertainty_profile = self._analyze_uncertainty(case_text)

        # Render PRECISE-U template
        enhanced_prompt = self.template.render(case_text, uncertainty_profile)

        # Create enhanced message list
        enhanced_messages = [self._pack_message("user", enhanced_prompt)]

        # Call base sampler
        response = self.base_sampler(enhanced_messages)

        # Extract response text
        raw_text = response.response_text

        # Extract full EUDEAS structured output for metrics calculation
        full_eudeas_text = self.template.extract_full_eudeas(raw_text)

        # Extract natural response for grading (this is what the grader sees)
        natural_response = self.template.extract_response(raw_text)

        # Extract EUDEAS components from the structured output
        components = self.scorer.extract_components(full_eudeas_text, uncertainty_profile)

        # Calculate EVS
        evs = components.evs()

        # Build EUDEAS metadata
        eudeas_metadata = {
            "eudeas_enabled": True,
            "uncertainty_profile": uncertainty_profile,
            "components": components.to_dict(),
            "evs": evs,
            "raw_response_length": len(raw_text),
            "structured_response_length": len(full_eudeas_text),
            "natural_response_length": len(natural_response),
        }

        # Merge with existing metadata
        merged_metadata = response.response_metadata.copy()
        merged_metadata["eudeas"] = eudeas_metadata

        # Return the NATURAL response for grading, not the structured EUDEAS output
        return SamplerResponse(
            response_text=natural_response,
            actual_queried_message_list=enhanced_messages,
            response_metadata=merged_metadata,
        )

    def get_base_sampler(self) -> SamplerBase:
        """Return the underlying base sampler."""
        return self.base_sampler
