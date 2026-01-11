"""
PRECISE-U Template for uncertainty-aware medical diagnosis prompting.

P - Probabilistic differential
R - Red flags
E - Explore uncertainties
C - Calculate confidence
I - Information needs
S - Safety nets
E - Explain to patient
U - Update plan
"""

from typing import Dict, List


class PRECISEUTemplate:
    """
    PRECISE-U structured prompting template for uncertainty-aware medical AI.
    """

    START = "### START OUTPUT"
    END = "### END OUTPUT"
    FINAL_START = "### FINAL RESPONSE"
    FINAL_END = "### END FINAL RESPONSE"

    # Micro-example showing expected format
    MICRO = (
        "Worked micro-example (format only; not the same case):\n"
        "P - Probabilistic differential:\n"
        "- Dx A: 40%\n- Dx B: 30%\n- Dx C: 20%\n- Other: 10%\n"
        "R - Red flags:\n- Sudden severe pain; syncope.\n"
        "E - Explore uncertainties:\n- Data: 0.4\n- Model: 0.3\n- OOD: 0.2\n- Structural: 0.1\n"
        "C - Calculate confidence:\n- Confidence: 60%\n"
        "Complexity: 0.6\n"
        "Targets (computed): H* = min(1, U + C*(1-M)); Q* = U*(1+C)*(1-M^2)\n"
        "Totals  U: 0.34 | C: 0.60 | M: 0.60\n"
        "Targets  H*: 0.76 | Q*: 0.54\n"
        "Humility: 0.76\nCuriosity: 0.54\n"
        "--- End example ---\n"
    )

    # Prompt library for different uncertainty types
    PROMPT_LIBRARY = {
        'u_data': [
            "Missing critical data; quantify impact.",
            "Pending labs; show probability shifts.",
        ],
        'u_model': [
            "Guideline disagreement; reason across frameworks.",
            "Low evidence quality; calibrate claims.",
        ],
        'u_ood': [
            "Atypical demographics; quantify OOD penalty.",
            "Rare/novel pattern; balance recognition with unknowns.",
        ],
        'u_struct': [
            "Resource constraints; optimize minimal tests.",
            "Limited follow-up; safer robust plan.",
        ],
    }

    # Default prompts always included
    DEFAULT_PROMPTS = [
        "Explicitly compute U, C, M then compute H* and Q* and match Humility/Curiosity to those targets.",
        "If U>0.6, keep Confidence ≤40% and justify.",
        "List the single question with the highest expected information gain.",
    ]

    @staticmethod
    def select_prompts(uncertainty_profile: Dict[str, float], max_prompts: int = 10) -> List[str]:
        """
        Select prompts based on uncertainty profile.

        Args:
            uncertainty_profile: Dict with keys u_data, u_model, u_ood, u_struct
            max_prompts: Maximum number of prompts to return

        Returns:
            List of selected prompts prioritized by uncertainty levels
        """
        selected = []

        # Sort uncertainty types by their values (highest first)
        sorted_uncertainties = sorted(
            uncertainty_profile.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Add prompts for high-uncertainty areas
        for key, value in sorted_uncertainties:
            if key in PRECISEUTemplate.PROMPT_LIBRARY and value > 0.3:
                selected.extend(PRECISEUTemplate.PROMPT_LIBRARY[key][:2])

        # Always add default prompts
        selected.extend(PRECISEUTemplate.DEFAULT_PROMPTS)

        return selected[:max_prompts]

    @staticmethod
    def render(case_text: str, uncertainty_profile: Dict[str, float]) -> str:
        """
        Render the full PRECISE-U prompt for a medical case.

        Args:
            case_text: The medical case/question text
            uncertainty_profile: Dict with uncertainty values

        Returns:
            Complete PRECISE-U formatted prompt
        """
        prompts = PRECISEUTemplate.select_prompts(uncertainty_profile)
        prompts_text = "\n".join(f"- {p}" for p in prompts[:5])

        return f"""You are a medical AI using PRECISE-U for uncertainty-aware diagnosis.

Case:
{case_text}

Write 250-380 words. Follow EXACTLY these headers, in this order, with bullet points.
You MUST compute numeric fields and print the lines below verbatim with numbers:
- Data/Model/OOD/Structural in [0,1]
- Confidence: NN%
- Complexity: 0.x
- Totals  U: uuu | C: ccc | M: mmm
- Targets  H*: hhh | Q*: qqq
- Humility: hhh
- Curiosity: qqq
(Compute U = 0.3*Data + 0.3*Model + 0.2*OOD + 0.2*Structural; C from case complexity; M from evidence strength.)

Begin after the line "{PRECISEUTemplate.START}" and end with "{PRECISEUTemplate.END}".

{PRECISEUTemplate.MICRO}

{PRECISEUTemplate.START}
P - Probabilistic differential:
- ...

R - Red flags:
- ...

E - Explore uncertainties:
- Data: 0.x
- Model: 0.x
- OOD: 0.x
- Structural: 0.x

C - Calculate confidence:
- Confidence: NN%

Complexity: 0.x

I - Information needs:
- ...

S - Safety nets:
- ...

E - Explain to patient:
- ...

U - Update plan:
- ...

Totals  U: uuu | C: ccc | M: mmm
Targets  H*: hhh | Q*: qqq
Humility: hhh
Curiosity: qqq
{PRECISEUTemplate.END}

{PRECISEUTemplate.FINAL_START}
Now write your FINAL RESPONSE to the patient/user. This should be a natural, conversational response that:
- Directly addresses their question/concern
- Is helpful and medically appropriate
- Uses plain language (not structured format)
- Incorporates the epistemic reasoning you did above (express appropriate uncertainty, seek information if needed)
- Does NOT include the PRECISE-U headers or numeric scores

Write your response here:
{PRECISEUTemplate.FINAL_END}

Prompts to emphasize (top-5):
{prompts_text}
"""

    @staticmethod
    def extract_response(raw_text: str) -> str:
        """
        Extract the FINAL RESPONSE (natural language) from the EUDEAS output.

        This extracts the clean response that should be evaluated by the grader,
        not the full PRECISE-U structured output.

        Args:
            raw_text: Raw model output

        Returns:
            Extracted natural response for grading
        """
        # First try to extract the FINAL RESPONSE section (preferred)
        final_start_idx = raw_text.find(PRECISEUTemplate.FINAL_START)
        final_end_idx = raw_text.rfind(PRECISEUTemplate.FINAL_END)

        if final_start_idx != -1 and final_end_idx != -1 and final_end_idx > final_start_idx:
            # Extract just the final response
            final_section = raw_text[final_start_idx + len(PRECISEUTemplate.FINAL_START):final_end_idx].strip()
            # Remove the instruction text if present
            if "Write your response here:" in final_section:
                final_section = final_section.split("Write your response here:")[-1].strip()
            return final_section

        # Fallback: try to extract from START/END markers
        start_idx = raw_text.find(PRECISEUTemplate.START)
        end_idx = raw_text.rfind(PRECISEUTemplate.END)

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return raw_text[start_idx + len(PRECISEUTemplate.START):end_idx].strip()

        return raw_text.strip()

    @staticmethod
    def extract_full_eudeas(raw_text: str) -> str:
        """
        Extract the full PRECISE-U structured output for EVS calculation.

        Args:
            raw_text: Raw model output

        Returns:
            Full PRECISE-U structured text for metrics extraction
        """
        start_idx = raw_text.find(PRECISEUTemplate.START)
        end_idx = raw_text.rfind(PRECISEUTemplate.END)

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return raw_text[start_idx + len(PRECISEUTemplate.START):end_idx].strip()

        return raw_text.strip()
