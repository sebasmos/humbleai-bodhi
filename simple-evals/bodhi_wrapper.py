"""
BODHI Wrapper - Integration with bodhi-llm package.

This wrapper integrates the bodhi-llm pip package into the simple-evals
evaluation framework for epistemic reasoning with curiosity and humility.

Usage:
    python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_consensus --use-bodhi

References:
    - PyPI: https://pypi.org/project/bodhi-llm/
    - PLOS Digital Health: doi:10.1371/journal.pdig.0001013
    - The Lancet: doi:10.1016/S0140-6736(25)01626-5
"""

from typing import Any, Dict, List
import time

from .types import MessageList, SamplerBase, SamplerResponse


class BODHIWrapper(SamplerBase):
    """
    Wrapper that integrates the bodhi-llm package for epistemic reasoning.

    This wrapper uses the published bodhi-llm package to implement the
    two-pass prompting strategy that embeds curiosity and humility.
    """

    def __init__(
        self,
        base_sampler: SamplerBase,
        domain: str = "medical",
        two_pass: bool = True,
    ):
        """
        Initialize BODHI wrapper.

        Args:
            base_sampler: The underlying sampler to wrap
            domain: Domain for BODHI prompts ("medical" or "general")
            two_pass: Whether to use two-pass mode (default: True)
        """
        self.base_sampler = base_sampler
        self.domain = domain
        self.two_pass = two_pass

        # Import bodhi-llm package
        try:
            from bodhi import BODHI, BODHIConfig
            self._bodhi_available = True
        except ImportError:
            raise ImportError(
                "bodhi-llm package not found. Install with: pip install bodhi-llm[all]"
            )

        # Create a chat function that uses the base_sampler
        self._chat_fn = self._create_chat_function()

        # Initialize BODHI with our chat function
        config = BODHIConfig(domain=domain)
        self._bodhi = BODHI(chat_function=self._chat_fn, config=config)

    def _create_chat_function(self):
        """Create a chat function adapter for BODHI that uses our base_sampler."""
        def chat_fn(messages: List[Dict[str, str]]) -> str:
            # Convert to MessageList format expected by base_sampler
            message_list = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in messages
            ]
            response = self.base_sampler(message_list)
            return response.response_text
        return chat_fn

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        """
        Process a message list through BODHI.

        Args:
            message_list: List of message dictionaries with 'role' and 'content'

        Returns:
            SamplerResponse with the BODHI-enhanced response
        """
        start_time = time.time()

        # Extract the case text from the message list
        case_text = self._extract_case_text(message_list)

        # Use BODHI to process the case
        bodhi_response = self._bodhi.complete(case_text, two_pass=self.two_pass)

        total_time = time.time() - start_time

        # Build metadata
        metadata = {
            "bodhi": {
                "enabled": True,
                "mode": "two_pass" if self.two_pass else "single_pass",
                "domain": self.domain,
                "analysis": bodhi_response.analysis,
                "timing": bodhi_response.metadata,
                "total_wrapper_time": total_time,
            }
        }

        # Return the response in the expected format
        return SamplerResponse(
            response_text=bodhi_response.content,
            actual_queried_message_list=message_list,
            response_metadata=metadata,
        )

    def _extract_case_text(self, message_list: MessageList) -> str:
        """Extract case text from message list."""
        parts = []
        for msg in message_list:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(str(part.get("text", "")))
        return "\n".join(p for p in parts if p).strip()
