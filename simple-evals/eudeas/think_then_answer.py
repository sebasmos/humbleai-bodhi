"""
Think-Then-Answer Template: Uses epistemic reasoning as internal Chain-of-Thought.

Now with Calibrated Confidence Steering using H* (target humility) and Q* (target curiosity).
"""

from typing import Dict, Tuple
import re


class ThinkThenAnswerTemplate:
    """
    Think-Then-Answer prompting with Calibrated Confidence Steering.

    The model does epistemic reasoning internally, calculates calibration targets,
    then provides a response calibrated to appropriate confidence levels.
    """

    @staticmethod
    def calculate_targets(u_data: float, u_model: float, u_ood: float, u_struct: float,
                          complexity: float, confidence: float) -> Tuple[float, float, float]:
        """
        Calculate H* (target humility) and Q* (target curiosity) from UEUDAS formulas.

        Returns: (total_uncertainty, h_star, q_star)
        """
        # Total uncertainty
        U = 0.3 * u_data + 0.3 * u_model + 0.2 * u_ood + 0.2 * u_struct
        C = complexity
        M = confidence

        # Target humility: H* = min(1, U + C(1-M))
        h_star = min(1.0, U + C * (1 - M))

        # Target curiosity: Q* = U(1+C)(1-M²)
        q_star = U * (1 + C) * (1 - M ** 2)

        return U, h_star, q_star

    @staticmethod
    def get_confidence_guidance(h_star: float, q_star: float) -> str:
        """
        Generate natural language guidance based on calibration targets.
        """
        # Determine confidence level from h_star (inverse relationship)
        confidence_level = 1 - h_star

        if confidence_level >= 0.7:
            confidence_guidance = "Be direct and confident in your recommendations. The evidence is clear."
        elif confidence_level >= 0.5:
            confidence_guidance = "Be moderately confident but acknowledge some uncertainty. Use phrases like 'most likely' or 'typically'."
        elif confidence_level >= 0.3:
            confidence_guidance = "Express significant uncertainty. Use phrases like 'it's possible', 'could be', 'I would consider'. Present multiple possibilities."
        else:
            confidence_guidance = "Be very cautious and humble. Emphasize uncertainty strongly. Present this as preliminary thinking that requires professional evaluation."

        # Determine information-seeking level from q_star
        if q_star >= 0.5:
            info_guidance = "Actively ask clarifying questions - there are critical information gaps that could change the assessment significantly."
        elif q_star >= 0.3:
            info_guidance = "Consider asking 1-2 key questions that would help refine the assessment."
        elif q_star >= 0.15:
            info_guidance = "You may optionally ask for additional context if relevant, but you have enough to provide guidance."
        else:
            info_guidance = "You have sufficient information to provide a clear response without needing more details."

        return f"""CALIBRATION GUIDANCE (based on uncertainty analysis):
- Confidence level: {confidence_level:.0%}
- {confidence_guidance}
- {info_guidance}"""

    @staticmethod
    def render(case_text: str, uncertainty_profile: Dict[str, float]) -> str:
        """
        Render a Think-Then-Answer prompt (single-pass, for backwards compatibility).
        """
        hints = []
        if uncertainty_profile.get('u_data', 0) > 0.4:
            hints.append("Some information may be missing - consider what additional details would help.")
        if uncertainty_profile.get('u_model', 0) > 0.4:
            hints.append("This case may have multiple valid interpretations.")
        if uncertainty_profile.get('u_ood', 0) > 0.4:
            hints.append("This may be an unusual or atypical presentation.")
        if uncertainty_profile.get('u_struct', 0) > 0.4:
            hints.append("Consider practical constraints and safety-first approaches.")

        hints_text = "\n".join(f"- {h}" for h in hints) if hints else "- Standard case, proceed with appropriate clinical reasoning."

        return f"""You are a medical AI assistant. Before responding, think through these steps internally:

1. DIFFERENTIAL: What are the possible diagnoses/explanations? Rank by probability.
2. RED FLAGS: Are there any urgent or dangerous signs requiring immediate attention?
3. UNCERTAINTIES: What am I uncertain about? What information is missing?
4. CONFIDENCE: How confident am I? Should I express uncertainty or be direct?
5. INFORMATION NEEDS: What questions would help clarify the situation?
6. SAFETY: What precautions or follow-up recommendations are needed?

Context hints for this case:
{hints_text}

Now respond to the following. Your response should:
- Be natural and conversational (NOT structured with headers or bullet points unless helpful)
- Express appropriate confidence (be uncertain where warranted, direct where clear)
- Ask clarifying questions if critical information is missing
- Include safety recommendations when relevant
- Be helpful and actionable

---

{case_text}

---

Provide your response:"""

    @staticmethod
    def render_two_pass_analysis(case_text: str, uncertainty_profile: Dict[str, float]) -> str:
        """
        Render the first pass: internal analysis prompt with numeric uncertainty extraction.
        """
        return f"""Analyze this medical case. Provide a structured internal assessment.

Case:
{case_text}

Provide your analysis in EXACTLY this format (include the numeric values):

DIFFERENTIAL:
- [Diagnosis 1]: [probability]%
- [Diagnosis 2]: [probability]%
- [Diagnosis 3]: [probability]%

RED_FLAGS: [List any urgent concerns, or "None identified"]

UNCERTAINTIES:
- Data uncertainty: [0.0-1.0] (missing information, unclear history)
- Model uncertainty: [0.0-1.0] (multiple valid interpretations, guideline disagreements)
- OOD uncertainty: [0.0-1.0] (atypical presentation, rare condition)
- Structural uncertainty: [0.0-1.0] (resource constraints, follow-up limitations)

COMPLEXITY: [0.0-1.0] (how complex is this case?)
CONFIDENCE: [0-100]% (how confident are you in your top diagnosis?)

KEY_QUESTION: [Single most important question to ask if you could only ask one]
KEY_RECOMMENDATION: [Single most important action/advice]

Keep analysis concise but include all numeric values."""

    @staticmethod
    def extract_analysis_values(analysis: str) -> Dict[str, float]:
        """
        Extract numeric values from the analysis text.
        """
        values = {
            'u_data': 0.3,
            'u_model': 0.3,
            'u_ood': 0.2,
            'u_struct': 0.2,
            'complexity': 0.5,
            'confidence': 0.5,
        }

        lower = analysis.lower()

        # Extract uncertainty values
        patterns = [
            (r'data\s*uncertainty[:\s]*([0-9.]+)', 'u_data'),
            (r'model\s*uncertainty[:\s]*([0-9.]+)', 'u_model'),
            (r'ood\s*uncertainty[:\s]*([0-9.]+)', 'u_ood'),
            (r'structural\s*uncertainty[:\s]*([0-9.]+)', 'u_struct'),
            (r'complexity[:\s]*([0-9.]+)', 'complexity'),
        ]

        for pattern, key in patterns:
            match = re.search(pattern, lower)
            if match:
                try:
                    val = float(match.group(1))
                    if val <= 1.0:
                        values[key] = val
                except ValueError:
                    pass

        # Extract confidence (as percentage)
        conf_match = re.search(r'confidence[:\s]*([0-9]+)\s*%', lower)
        if conf_match:
            try:
                values['confidence'] = float(conf_match.group(1)) / 100.0
            except ValueError:
                pass

        return values

    @staticmethod
    def render_two_pass_response(case_text: str, analysis: str) -> str:
        """
        Render the second pass: generate calibrated response based on analysis.
        """
        # Extract values from analysis
        values = ThinkThenAnswerTemplate.extract_analysis_values(analysis)

        # Calculate calibration targets
        U, h_star, q_star = ThinkThenAnswerTemplate.calculate_targets(
            values['u_data'], values['u_model'], values['u_ood'], values['u_struct'],
            values['complexity'], values['confidence']
        )

        # Get natural language guidance
        calibration_guidance = ThinkThenAnswerTemplate.get_confidence_guidance(h_star, q_star)

        return f"""Based on your clinical analysis, now provide a response to the patient.

Your analysis summary:
{analysis}

---

{calibration_guidance}

---

Original case:
{case_text}

---

Now write your response. Remember:
- Match your confidence level to the calibration guidance above
- If guidance says to ask questions, do so
- If guidance says to be cautious, use hedging language
- If guidance says to be direct, be clear and actionable
- Always include appropriate safety recommendations

Your response to the patient:"""

    @staticmethod
    def render_two_pass_response_v2(case_text: str, analysis: str) -> str:
        """
        V2: More aggressive calibration with explicit behavioral instructions.
        """
        values = ThinkThenAnswerTemplate.extract_analysis_values(analysis)
        U, h_star, q_star = ThinkThenAnswerTemplate.calculate_targets(
            values['u_data'], values['u_model'], values['u_ood'], values['u_struct'],
            values['complexity'], values['confidence']
        )

        confidence_pct = int((1 - h_star) * 100)

        # Build specific behavioral instructions
        behaviors = []

        if h_star >= 0.6:
            behaviors.append("MUST express significant uncertainty - use phrases like 'it's possible', 'could be', 'would need to rule out'")
            behaviors.append("MUST present multiple possibilities, not just one answer")
        elif h_star >= 0.4:
            behaviors.append("Should acknowledge some uncertainty - use phrases like 'likely', 'most commonly', 'typically'")
        else:
            behaviors.append("Can be direct and confident in recommendations")

        if q_star >= 0.4:
            behaviors.append("MUST ask at least 2 clarifying questions before giving definitive advice")
        elif q_star >= 0.2:
            behaviors.append("Should ask 1 key clarifying question")

        # Check for red flags in analysis
        if 'emergency' in analysis.lower() or 'urgent' in analysis.lower() or 'red flag' in analysis.lower():
            behaviors.append("MUST include clear safety recommendations and when to seek emergency care")

        behaviors_text = "\n".join(f"- {b}" for b in behaviors)

        return f"""Based on your clinical analysis, provide a calibrated response.

ANALYSIS SUMMARY:
{analysis}

---

CALIBRATION TARGETS (computed from uncertainty analysis):
- Target confidence level: {confidence_pct}%
- Target humility (H*): {h_star:.2f}
- Target curiosity (Q*): {q_star:.2f}

REQUIRED BEHAVIORS:
{behaviors_text}

---

ORIGINAL CASE:
{case_text}

---

Write your response following the required behaviors above. Be natural and conversational, but ensure your confidence level and question-asking behavior matches the calibration targets.

Your response:"""

    @staticmethod
    def render_two_pass_response_v3(case_text: str, analysis: str) -> str:
        """
        V3: Minimal - just use the analysis insights without formula-based calibration.
        Focus on the KEY_QUESTION and KEY_RECOMMENDATION from analysis.
        """
        # Extract key question and recommendation from analysis
        key_question = ""
        key_rec = ""
        red_flags = ""

        for line in analysis.split('\n'):
            lower_line = line.lower().strip()
            if lower_line.startswith('key_question:') or lower_line.startswith('key question:'):
                key_question = line.split(':', 1)[-1].strip()
            elif lower_line.startswith('key_recommendation:') or lower_line.startswith('key recommendation:'):
                key_rec = line.split(':', 1)[-1].strip()
            elif 'red_flag' in lower_line or 'red flag' in lower_line:
                red_flags = line.split(':', 1)[-1].strip() if ':' in line else line.strip()

        # Build focused guidance
        guidance_parts = []
        if key_question and key_question.lower() != 'none' and 'n/a' not in key_question.lower():
            guidance_parts.append(f"Consider asking: {key_question}")
        if red_flags and 'none' not in red_flags.lower():
            guidance_parts.append(f"Address safety: {red_flags}")
        if key_rec:
            guidance_parts.append(f"Key point: {key_rec}")

        guidance = "\n".join(f"- {g}" for g in guidance_parts) if guidance_parts else "Provide clear, helpful guidance."

        return f"""Based on your analysis, provide a helpful response to the patient.

Your analysis identified:
{guidance}

Case:
{case_text}

Write a natural, helpful response. Be conversational and actionable. Include safety recommendations if relevant.

Your response:"""

    @staticmethod
    def render_two_pass_response_v4(case_text: str, analysis: str) -> str:
        """
        V4: Ultra-minimal - just pass analysis context, let model respond naturally.
        """
        return f"""You analyzed this case and found:

{analysis}

Now respond naturally to the patient. Be helpful, clear, and include any important safety information.

Case:
{case_text}

Your response:"""

    @staticmethod
    def render_simple_analysis(case_text: str, uncertainty_profile: Dict[str, float]) -> str:
        """
        Simple analysis prompt - less structured, focuses on key clinical thinking.
        """
        return f"""Analyze this medical case. Provide a brief internal assessment (this will not be shown to the patient).

Case:
{case_text}

Provide your analysis in this format:
DIFFERENTIAL: [Top 3 possibilities with rough probabilities]
RED FLAGS: [Any urgent concerns, or "None identified"]
MISSING INFO: [Critical information gaps, or "Sufficient information"]
CONFIDENCE: [High/Medium/Low and why]
KEY RECOMMENDATION: [Single most important point]

Keep your analysis under 150 words."""

    @staticmethod
    def render_simple_response(case_text: str, analysis: str) -> str:
        """
        Simple response prompt - matches the original successful two-pass.
        """
        return f"""Based on your clinical analysis, now provide a response to the patient.

Your analysis:
{analysis}

Original case:
{case_text}

Guidelines for your response:
- Write naturally and conversationally
- Express appropriate uncertainty based on your confidence level
- If you identified missing info, ask about it
- If you identified red flags, address them clearly
- Be helpful and actionable

Provide your response to the patient:"""

    @staticmethod
    def render_two_pass_analysis_v5(case_text: str, uncertainty_profile: Dict[str, float]) -> str:
        """
        V5: Enhanced analysis with HealthBench-specific focus.
        """
        return f"""You are a medical AI assistant. Analyze this case carefully.

Case:
{case_text}

Think through:
1. What is most likely going on? What are the alternatives?
2. Are there any red flags or urgent concerns?
3. What information is missing that would help?
4. How confident are you?

Provide your analysis:
ASSESSMENT: [Your main assessment in 1-2 sentences]
DIFFERENTIAL: [Top possibilities]
URGENCY: [Is this urgent? What should happen if symptoms worsen?]
GAPS: [What would you want to know?]
CONFIDENCE: [How sure are you? High/Medium/Low]

Keep analysis under 200 words."""

    @staticmethod
    def render_two_pass_response_v5(case_text: str, analysis: str) -> str:
        """
        V5: Response that incorporates HealthBench rubric elements naturally.
        """
        # Check if analysis mentions urgency or red flags
        has_urgency = any(word in analysis.lower() for word in ['urgent', 'emergency', 'immediate', 'red flag', 'worsen'])
        has_gaps = any(word in analysis.lower() for word in ['missing', 'need to know', 'would help', 'gaps', 'unclear'])

        # Determine confidence from analysis
        low_conf = any(word in analysis.lower() for word in ['low confidence', 'uncertain', 'unclear', 'multiple possibilities'])

        # Build dynamic guidance
        guidance = []
        if has_gaps:
            guidance.append("Ask about the information gaps you identified")
        if has_urgency:
            guidance.append("Include clear guidance on when to seek urgent care")
        if low_conf:
            guidance.append("Express appropriate uncertainty - don't be overconfident")
        else:
            guidance.append("You can be relatively direct since confidence is reasonable")

        guidance_text = "\n".join(f"- {g}" for g in guidance)

        return f"""Based on your analysis, respond to the patient.

Your analysis:
{analysis}

Case:
{case_text}

Response guidance:
{guidance_text}

Write a natural, helpful response:"""

    @staticmethod
    def render_curious_humble_analysis(case_text: str, uncertainty_profile: Dict[str, float]) -> str:
        """
        V6: Analysis focused on curiosity (what to ask) and humility (what's uncertain).
        """
        return f"""You are a thoughtful medical AI. Analyze this case with intellectual humility.

Case:
{case_text}

Think carefully and provide:

1. WHAT I THINK: Your best assessment (be honest about confidence)
2. WHAT I'M UNSURE ABOUT: Key uncertainties that affect your assessment
3. WHAT I NEED TO KNOW: Questions that would significantly help (be specific)
4. RED FLAGS: Any urgent warning signs (or "None")
5. SAFE ADVICE: What can you confidently recommend regardless of uncertainty?

Be genuinely curious and humble - don't pretend to know more than you do."""

    @staticmethod
    def render_curious_humble_response(case_text: str, analysis: str) -> str:
        """
        V6: Response that naturally embeds curiosity and humility from analysis.
        """
        # Extract what the model is unsure about and wants to know
        lines = analysis.split('\n')
        unsure_about = ""
        need_to_know = ""
        red_flags = ""
        safe_advice = ""

        current_section = None
        for line in lines:
            lower = line.lower().strip()
            if 'unsure about' in lower or "i'm unsure" in lower:
                current_section = 'unsure'
            elif 'need to know' in lower or 'questions' in lower:
                current_section = 'need'
            elif 'red flag' in lower:
                current_section = 'flags'
            elif 'safe advice' in lower or 'confidently recommend' in lower:
                current_section = 'safe'
            elif current_section and line.strip():
                if current_section == 'unsure':
                    unsure_about += line + " "
                elif current_section == 'need':
                    need_to_know += line + " "
                elif current_section == 'flags':
                    red_flags += line + " "
                elif current_section == 'safe':
                    safe_advice += line + " "

        # Build natural guidance based on what we found
        response_hints = []

        if need_to_know.strip() and 'none' not in need_to_know.lower():
            response_hints.append(f"You identified important questions - ASK them naturally in your response")

        if unsure_about.strip() and 'none' not in unsure_about.lower():
            response_hints.append(f"You noted uncertainty - express this honestly (e.g., 'it could be...', 'I'd want to rule out...')")

        if red_flags.strip() and 'none' not in red_flags.lower():
            response_hints.append(f"Include clear guidance on when to seek immediate care")

        if safe_advice.strip():
            response_hints.append(f"Include the safe recommendations you identified")

        hints_text = "\n".join(f"- {h}" for h in response_hints) if response_hints else "- Respond helpfully based on your analysis"

        return f"""Now respond to the patient naturally.

Your analysis:
{analysis}

Case:
{case_text}

In your response:
{hints_text}

Write conversationally - be genuinely helpful, curious, and appropriately humble:"""

    @staticmethod
    def render_v7_analysis(case_text: str, uncertainty_profile: Dict[str, float]) -> str:
        """
        V7: Simpler analysis focused on actionable outputs.
        """
        return f"""Analyze this medical case briefly.

Case:
{case_text}

Provide:
ASSESSMENT: [1-2 sentence summary of what you think is going on]
CONFIDENCE: [High/Medium/Low - be honest]
MUST ASK: [1-2 critical questions you'd want answered, or "None needed"]
MUST WARN: [Any urgent safety concerns, or "None"]

Keep it under 100 words."""

    @staticmethod
    def render_v7_response(case_text: str, analysis: str) -> str:
        """
        V7: Simple response prompt that naturally incorporates analysis.
        """
        # Check if there are questions to ask
        has_questions = 'must ask:' in analysis.lower() and 'none' not in analysis.lower().split('must ask:')[1][:50]
        has_warnings = 'must warn:' in analysis.lower() and 'none' not in analysis.lower().split('must warn:')[1][:50]
        is_low_conf = 'low' in analysis.lower() and 'confidence' in analysis.lower()

        instructions = []
        if has_questions:
            instructions.append("Ask the question(s) you identified")
        if has_warnings:
            instructions.append("Include the safety warning")
        if is_low_conf:
            instructions.append("Express your uncertainty honestly")
        if not instructions:
            instructions.append("Provide clear, helpful guidance")

        inst_text = " and ".join(instructions) + "."

        return f"""Respond to the patient based on your analysis.

Analysis:
{analysis}

Case:
{case_text}

{inst_text} Be natural and helpful.

Your response:"""
