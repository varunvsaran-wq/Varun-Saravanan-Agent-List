"""
Safety module: red-flag detection, safety prompts, and guardrails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schemas import EmergencyOutput, SymptomInput


# ── Red-flag rules ────────────────────────────────────────────────────────────

@dataclass
class RedFlag:
    name: str
    patterns: list[str]  # regex patterns (case-insensitive)
    reason: str
    action: str


RED_FLAGS: list[RedFlag] = [
    RedFlag(
        name="chest_pain",
        patterns=[
            r"chest\s*(pain|pressure|tightness|heaviness|squeezing)",
            r"angina",
        ],
        reason="Chest pain or pressure may indicate a cardiac emergency.",
        action="Call 911 or go to the nearest emergency department immediately.",
    ),
    RedFlag(
        name="severe_breathing",
        patterns=[
            r"(severe|extreme|can'?t)\s*(shortness\s*of\s*breath|breathe|breathing)",
            r"blue\s*(lips|face|fingertips|skin)",
            r"cyanosis",
        ],
        reason="Severe difficulty breathing or cyanosis requires urgent evaluation.",
        action="Call 911 or go to the nearest emergency department immediately.",
    ),
    RedFlag(
        name="stroke_signs",
        patterns=[
            r"(one[- ]?sided?|unilateral)\s*(weakness|numbness|paralysis)",
            r"facial\s*droop",
            r"(trouble|difficulty|unable)\s*(speaking|speech|talking)",
            r"sudden\s*severe\s*headache",
            r"worst\s*headache\s*(of|in)\s*(my|their)?\s*life",
        ],
        reason="These symptoms may indicate a stroke (time-critical emergency).",
        action=(
            "Call 911 immediately. Note the time symptoms started. "
            "Do NOT drive yourself."
        ),
    ),
    RedFlag(
        name="severe_abdominal",
        patterns=[
            r"(severe|acute)\s*abdominal\s*pain.*(rigid|board[- ]?like|faint)",
            r"rigid\s*abdomen",
        ],
        reason="Severe abdominal pain with rigidity or fainting may indicate a surgical emergency.",
        action="Call 911 or go to the nearest emergency department immediately.",
    ),
    RedFlag(
        name="altered_mental_status",
        patterns=[
            r"(confusion|disoriented|altered\s*mental)",
            r"seizure",
            r"neck\s*stiffness.*(fever|headache)",
            r"(fever|headache).*neck\s*stiffness",
        ],
        reason="Confusion, seizures, or neck stiffness with fever may indicate meningitis or another neurological emergency.",
        action="Call 911 or go to the nearest emergency department immediately.",
    ),
    RedFlag(
        name="uncontrolled_bleeding",
        patterns=[
            r"(uncontrolled|won'?t\s*stop|profuse|massive)\s*bleeding",
            r"hemorrhag",
        ],
        reason="Uncontrolled bleeding requires immediate intervention.",
        action="Apply direct pressure and call 911 immediately.",
    ),
    RedFlag(
        name="suicidal_self_harm",
        patterns=[
            r"suicid(al|e|ing)",
            r"(want|plan|going)\s*to\s*(kill|end|hurt)\s*(myself|themselves|my\s*life)",
            r"self[- ]?harm",
        ],
        reason="Expression of suicidal intent or self-harm.",
        action=(
            "If you or someone you know is in immediate danger, call 911. "
            "National Suicide Prevention Lifeline: 988 (call or text). "
            "Crisis Text Line: text HOME to 741741."
        ),
    ),
    RedFlag(
        name="anaphylaxis",
        patterns=[
            r"(swelling|swell).*(face|throat|tongue|lips)",
            r"(throat|airway)\s*(closing|swelling|tight)",
            r"(severe|anaphyla)\s*allergic\s*reaction",
            r"anaphylaxis",
            r"wheezing.*(hives|rash|swelling)",
        ],
        reason="Signs of anaphylaxis (severe allergic reaction).",
        action=(
            "Use epinephrine auto-injector (EpiPen) if available. "
            "Call 911 immediately."
        ),
    ),
    RedFlag(
        name="infant_emergency",
        patterns=[
            r"(infant|newborn|baby).*(fever|not\s*feeding|lethargic|limp|blue)",
            r"(fever|temp).*(infant|newborn|baby)",
        ],
        reason="Infants with fever, poor feeding, or lethargy need urgent evaluation.",
        action="Go to the nearest pediatric emergency department or call 911.",
    ),
]


def check_red_flags(symptom_input: SymptomInput) -> EmergencyOutput:
    """Scan the symptom text (and vitals) for red-flag patterns."""
    text = symptom_input.symptoms_text.lower()
    triggered: list[RedFlag] = []

    for flag in RED_FLAGS:
        for pattern in flag.patterns:
            if re.search(pattern, text, re.IGNORECASE):
                triggered.append(flag)
                break  # one match per flag is enough

    # Vital-based red flags
    if symptom_input.heart_rate_bpm is not None:
        if symptom_input.heart_rate_bpm > 150 or symptom_input.heart_rate_bpm < 40:
            triggered.append(
                RedFlag(
                    name="extreme_hr",
                    patterns=[],
                    reason=f"Heart rate of {symptom_input.heart_rate_bpm} bpm is outside safe range.",
                    action="Call 911 or go to the nearest emergency department immediately.",
                )
            )
    if symptom_input.temperature_f is not None:
        if symptom_input.temperature_f >= 104.0:
            triggered.append(
                RedFlag(
                    name="high_fever",
                    patterns=[],
                    reason=f"Temperature of {symptom_input.temperature_f}\u00b0F is dangerously high.",
                    action="Call 911 or go to the nearest emergency department immediately.",
                )
            )

    if not triggered:
        return EmergencyOutput(is_emergency=False, reasons=[], immediate_action="")

    reasons = list({f.reason for f in triggered})
    actions = list({f.action for f in triggered})
    combined_action = " | ".join(actions) if len(actions) > 1 else actions[0]
    return EmergencyOutput(
        is_emergency=True,
        reasons=reasons,
        immediate_action=combined_action,
    )


# ── System-level safety prompt (injected into LLM calls) ─────────────────────

SAFETY_SYSTEM_PROMPT = """\
You are a medical-symptom educational triage assistant. You are NOT a doctor, \
NOT a clinician, and you do NOT provide medical diagnoses or treatment plans.

ABSOLUTE RULES — NEVER VIOLATE:
1. Begin every response by acknowledging: "This is informational only and is \
not medical advice. It is not a substitute for professional medical evaluation."
2. NEVER claim to diagnose. Use phrases like "possible conditions that could \
explain these symptoms" or "heuristic likelihood estimates for educational \
triage only."
3. NEVER recommend prescription medication changes. You may suggest general \
self-care (hydration, rest) ONLY when low-risk, and always with caveats.
4. If ANY emergency red flag is present, you MUST instruct the user to seek \
immediate emergency care (call 911) BEFORE any differential discussion.
5. Percentages you assign are heuristic educational estimates, NOT clinical \
probabilities. They MUST sum to 100 across the top 5. State this clearly.
6. When uncertain, flatten the distribution and label confidence "low". \
When the knowledge base lacks coverage, say so explicitly.
7. Ground all reasoning in the retrieved knowledge-base chunks. Cite chunk IDs. \
Do NOT fabricate medical facts.
8. Ask clarifying questions when input is insufficient.
9. Do NOT request unnecessary personal data. Do NOT store user data.
10. Be calm, clear, non-alarmist, and non-judgmental.\
"""

DIFFERENTIAL_PROMPT_TEMPLATE = """\
Given the following patient-reported symptoms and context, plus the retrieved \
knowledge-base excerpts below, produce a differential triage assessment.

=== PATIENT INPUT ===
{patient_input}

=== RETRIEVED KNOWLEDGE BASE EXCERPTS ===
{kb_excerpts}

=== INSTRUCTIONS ===
Return a JSON object matching this schema EXACTLY (no extra keys):
{{
  "emergency": {{
    "is_emergency": <bool>,
    "reasons": [<string>, ...],
    "immediate_action": "<string>"
  }},
  "differential": [
    {{
      "condition": "<string>",
      "percent": <number 0-100>,
      "confidence": "low" | "medium" | "high",
      "supporting_features": ["<string>", ...],
      "missing_or_contradicting_features": ["<string>", ...],
      "rationale": "<string — cite KB chunk IDs>",
      "citations": ["<chunk_id>", ...]
    }}
    // ... up to 5 entries, percentages summing to 100
  ],
  "most_important_questions": ["<string>", ...],  // 3-8 questions
  "next_steps": {{
    "self_care": ["<string>", ...],
    "see_a_clinician_if": ["<string>", ...],
    "suggested_clinician_type": "<string>"
  }},
  "disclaimer": "INFORMATIONAL ONLY — NOT MEDICAL ADVICE. This output is not a \
substitute for professional medical evaluation. Heuristic likelihood estimates \
are for educational triage only, not clinical probabilities. Always consult a \
qualified healthcare professional for diagnosis and treatment."
}}

RULES:
- Percentages MUST sum to exactly 100 across the top 5 differential entries.
- If uncertainty is high, flatten the distribution (e.g., 25/20/20/18/17) and \
mark confidence "low" on all entries.
- If KB coverage is insufficient, include fewer conditions and note the gap.
- Justify each percentage by symptom-match strength, base-rate info from KB \
(if available), and exclusions/contradictions.
- Cite KB chunk IDs in "citations" and in "rationale".
- If emergency red flags were detected upstream, keep is_emergency=true and \
repeat the immediate_action. The differential is secondary.
- Ask 3-8 targeted clarifying questions in "most_important_questions".
- NEVER recommend prescription changes.
- Output valid JSON only — no markdown fences, no commentary outside the JSON.\
"""
