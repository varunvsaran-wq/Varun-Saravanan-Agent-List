"""
Pydantic schemas for the Medical Symptom Differential + Triage Agent.
Defines structured input/output types used across all modules.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Input models ──────────────────────────────────────────────────────────────


class AgeRange(str, Enum):
    CHILD = "child"
    TEEN = "teen"
    ADULT = "adult"
    OLDER_ADULT = "older_adult"


class Severity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class SymptomInput(BaseModel):
    """User-provided symptom description plus optional structured fields."""

    symptoms_text: str = Field(
        ..., description="Free-text description of symptoms."
    )
    age_range: Optional[AgeRange] = None
    sex_at_birth: Optional[str] = Field(
        None, description="Optional: male / female / intersex"
    )
    pregnancy_possible: Optional[bool] = None
    duration: Optional[str] = Field(
        None, description="e.g. '3 days', '2 hours', '1 week'"
    )
    severity: Optional[Severity] = None
    temperature_f: Optional[float] = None
    heart_rate_bpm: Optional[int] = None
    existing_conditions: Optional[list[str]] = None
    current_medications: Optional[list[str]] = None
    allergies: Optional[list[str]] = None


# ── Output models (match required JSON schema exactly) ────────────────────────


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EmergencyOutput(BaseModel):
    is_emergency: bool
    reasons: list[str] = Field(default_factory=list)
    immediate_action: str = ""


class DifferentialEntry(BaseModel):
    condition: str
    percent: float = Field(ge=0, le=100)
    confidence: Confidence
    supporting_features: list[str]
    missing_or_contradicting_features: list[str]
    rationale: str
    citations: list[str] = Field(
        default_factory=list,
        description="KB chunk IDs that ground this entry.",
    )


class NextSteps(BaseModel):
    self_care: list[str] = Field(default_factory=list)
    see_a_clinician_if: list[str] = Field(default_factory=list)
    suggested_clinician_type: str = ""


DISCLAIMER = (
    "INFORMATIONAL ONLY — NOT MEDICAL ADVICE. This output is not a substitute "
    "for professional medical evaluation. Heuristic likelihood estimates are "
    "for educational triage only, not clinical probabilities. Always consult a "
    "qualified healthcare professional for diagnosis and treatment."
)


class TriageOutput(BaseModel):
    """Top-level output schema returned by the agent."""

    emergency: EmergencyOutput
    differential: list[DifferentialEntry] = Field(
        default_factory=list, max_length=5
    )
    most_important_questions: list[str] = Field(default_factory=list)
    next_steps: NextSteps = Field(default_factory=NextSteps)
    disclaimer: str = DISCLAIMER

    def to_human_readable(self) -> str:
        """Render the structured output as a clear, calm, user-facing summary."""
        lines: list[str] = []

        # Emergency banner
        if self.emergency.is_emergency:
            lines.append("=" * 60)
            lines.append("  *** URGENT — POSSIBLE EMERGENCY ***")
            lines.append("=" * 60)
            for r in self.emergency.reasons:
                lines.append(f"  - {r}")
            lines.append(f"\n  >> {self.emergency.immediate_action}")
            lines.append("=" * 60)
            lines.append("")

        # Differential
        if self.differential:
            lines.append("--- Possible Conditions (heuristic estimates) ---")
            lines.append(
                "(These are educational estimates, NOT clinical probabilities.)\n"
            )
            for i, d in enumerate(self.differential, 1):
                lines.append(
                    f"  {i}. {d.condition}  —  ~{d.percent:.0f}%  "
                    f"[confidence: {d.confidence.value}]"
                )
                lines.append(f"     Supporting: {', '.join(d.supporting_features)}")
                if d.missing_or_contradicting_features:
                    lines.append(
                        f"     Missing/Contradicting: "
                        f"{', '.join(d.missing_or_contradicting_features)}"
                    )
                lines.append(f"     Rationale: {d.rationale}")
                if d.citations:
                    lines.append(f"     Sources: {', '.join(d.citations)}")
                lines.append("")

        # Questions
        if self.most_important_questions:
            lines.append("--- Clarifying Questions ---")
            for q in self.most_important_questions:
                lines.append(f"  - {q}")
            lines.append("")

        # Next steps
        ns = self.next_steps
        if ns.self_care or ns.see_a_clinician_if:
            lines.append("--- What To Do Next ---")
            if ns.self_care:
                lines.append("  Self-care (low-risk, general guidance):")
                for s in ns.self_care:
                    lines.append(f"    - {s}")
            if ns.see_a_clinician_if:
                lines.append("  See a clinician if:")
                for s in ns.see_a_clinician_if:
                    lines.append(f"    - {s}")
            if ns.suggested_clinician_type:
                lines.append(
                    f"  Suggested clinician type: {ns.suggested_clinician_type}"
                )
            lines.append("")

        # Disclaimer
        lines.append(f"[Disclaimer] {self.disclaimer}")
        return "\n".join(lines)
