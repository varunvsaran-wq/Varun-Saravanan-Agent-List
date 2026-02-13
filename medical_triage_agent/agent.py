"""
Medical Symptom Differential + Triage Agent — Google ADK implementation.

Uses a SequentialAgent pipeline:
  Step 1  ─  Parse & normalise symptoms
  Step 2  ─  Red-flag triage (rule-based)
  Step 3  ─  Retrieve relevant KB chunks (local RAG)
  Step 4  ─  LLM: generate differential + heuristic percentages
  Step 5  ─  Validate & emit structured JSON + human-readable summary

No web browsing, no external APIs — only the LLM and local retrieval.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .rag import MedicalKBRetriever
from .safety import check_red_flags, SAFETY_SYSTEM_PROMPT, DIFFERENTIAL_PROMPT_TEMPLATE
from .schemas import (
    SymptomInput,
    TriageOutput,
    EmergencyOutput,
    DifferentialEntry,
    NextSteps,
    Confidence,
    DISCLAIMER,
)



# ── Configuration ─────────────────────────────────────────────────────────────

KB_DIR = os.environ.get("MEDICAL_KB_DIR", str(Path(__file__).parent / "medical_kb"))
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "8"))
MODEL_ID = os.environ.get("MODEL_ID", "gemini-2.0-flash")

# ── Initialise retriever (module-level so it indexes once) ────────────────────

retriever = MedicalKBRetriever(kb_dir=KB_DIR)

# ── Tool functions exposed to LlmAgent ────────────────────────────────────────


def retrieve_knowledge(query: str, k: int = RAG_TOP_K) -> list[dict]:
    """Search the local medical knowledge base and return relevant excerpts.

    Args:
        query: The search query (symptoms, condition name, etc.).
        k: Number of chunks to retrieve (default 8).

    Returns:
        A list of dicts with keys: id, title, text, score.
    """
    results = retriever.retrieve(query, k=k)
    return [
        {"id": r.id, "title": r.title, "text": r.text, "score": round(r.score, 4)}
        for r in results
    ]


def check_emergency_flags(
    symptoms_text: str,
    temperature_f: float | None = None,
    heart_rate_bpm: int | None = None,
) -> dict:
    """Run rule-based red-flag checks on the symptom text and vitals.

    Args:
        symptoms_text: Free-text symptom description.
        temperature_f: Optional temperature in Fahrenheit.
        heart_rate_bpm: Optional heart rate in beats per minute.

    Returns:
        A dict with is_emergency, reasons, and immediate_action.
    """
    inp = SymptomInput(
        symptoms_text=symptoms_text,
        temperature_f=temperature_f,
        heart_rate_bpm=heart_rate_bpm,
    )
    result = check_red_flags(inp)
    return result.model_dump()


# ── Build the ADK agent ──────────────────────────────────────────────────────

TRIAGE_AGENT_INSTRUCTION = f"""\
{SAFETY_SYSTEM_PROMPT}

You are a medical-symptom educational triage assistant. When a user describes
symptoms, you MUST call BOTH tools before writing ANY text response.

CRITICAL RULES — FOLLOW EXACTLY:
1. Do NOT write ANY text to the user until you have called BOTH tools below.
2. Do NOT echo, repeat, or display tool results as JSON. Tool results are
   internal data for YOUR reasoning only — never show them to the user.
3. Do NOT say "I will now gather information" or "let me proceed" — just
   call the tools silently and then write your final response.
4. You MUST call BOTH tools for every query. No exceptions.

TOOL CALLS (do these FIRST, before any text output):

  Tool call 1: `check_emergency_flags`
    - Pass the user's symptom text and any vitals they mention.

  Tool call 2: `retrieve_knowledge`
    - Pass a search query based on the user's symptoms.
    - You may call this a second time with a different query if needed.

After BOTH tool calls return, write your response following these rules:

RESPONSE FORMAT:
  Write a calm, clear, human-readable response. NO JSON. NO code blocks.
  NO XML tags. NO raw data. Structure it like this:

  1. Start with: "This is informational only and is not medical advice. It is
     not a substitute for professional medical evaluation."

  2. If emergency flags were triggered, show an urgent warning with
     instructions BEFORE anything else.

  3. List possible conditions with heuristic likelihood estimates in plain
     language. For example:
       "**Viral pharyngitis (~35%)** — Your sore throat and cough are
       consistent with a viral infection. The absence of strep makes this
       more likely."
     Include up to 5 conditions. Percentages must sum to 100.
     For each, mention supporting and missing/contradicting features.

  4. Ask 3-8 clarifying questions to help narrow down the possibilities.

  5. Provide next steps: self-care suggestions and when to see a clinician.

  6. End with the disclaimer: "{DISCLAIMER}"

ADDITIONAL RULES:
  - Percentages are heuristic educational estimates, NOT clinical probabilities.
  - If uncertainty is high, flatten the distribution and note low confidence.
  - If the knowledge base lacks coverage, say so explicitly.
  - NEVER recommend prescription medication changes.
  - NEVER promise to do something "next" or "later" — everything goes in
    this one response.
  - Keep the tone calm, empathetic, and non-alarmist.
"""


# The main triage LLM agent with tool access
triage_llm_agent = LlmAgent(
    name="MedicalTriageLLM",
    model=MODEL_ID,
    instruction=TRIAGE_AGENT_INSTRUCTION,
    tools=[retrieve_knowledge, check_emergency_flags],
)

# Wrap in a SequentialAgent (single-step sequence; extensible for future steps)
root_agent = SequentialAgent(
    name="MedicalTriageAgent",
    description=(
        "An educational medical-symptom differential and triage agent. "
        "Provides heuristic likelihood estimates for possible conditions "
        "based on user-reported symptoms and a local medical knowledge base. "
        "NOT a substitute for professional medical evaluation."
    ),
    sub_agents=[triage_llm_agent],
)


# ── Helper: parse structured output from LLM response ────────────────────────


def parse_agent_output(raw_text: str) -> tuple[TriageOutput | None, str]:
    """Extract JSON and summary from the agent's tagged response.

    Returns (parsed_output_or_None, human_summary).
    """
    # Try <JSON>...</JSON> tags first
    json_str = ""
    summary = raw_text

    import re

    json_match = re.search(r"<JSON>\s*(.*?)\s*</JSON>", raw_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Fallback: find first { ... } block
        brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if brace_match:
            json_str = brace_match.group(0)

    summary_match = re.search(r"<SUMMARY>\s*(.*?)\s*</SUMMARY>", raw_text, re.DOTALL)
    if summary_match:
        summary = summary_match.group(1)

    if json_str:
        try:
            data = json.loads(json_str)
            output = TriageOutput(**data)
            return output, summary
        except (json.JSONDecodeError, Exception) as e:
            print(f"[Agent] Warning: Could not parse JSON output: {e}")

    return None, summary


# ── Convenience runner for CLI / programmatic use ─────────────────────────────


async def run_triage(symptom_text: str, **kwargs: Any) -> tuple[TriageOutput | None, str]:
    """Run the triage agent on the given symptom input.

    Returns (structured_output, human_readable_summary).
    """
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="medical_triage", user_id="user"
    )

    runner = Runner(
        agent=root_agent,
        app_name="medical_triage",
        session_service=session_service,
    )

    # Build user message
    parts = [f"My symptoms: {symptom_text}"]
    field_labels = {
        "age_range": "Age range",
        "sex_at_birth": "Sex at birth",
        "pregnancy_possible": "Pregnancy possible",
        "duration": "Duration",
        "severity": "Severity",
        "temperature_f": "Temperature (F)",
        "heart_rate_bpm": "Heart rate (bpm)",
        "existing_conditions": "Existing conditions",
        "current_medications": "Current medications",
        "allergies": "Allergies",
    }
    for key, label in field_labels.items():
        if key in kwargs and kwargs[key] is not None:
            val = kwargs[key]
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            parts.append(f"{label}: {val}")

    user_msg = "\n".join(parts)

    content = types.Content(
        role="user", parts=[types.Part.from_text(text=user_msg)]
    )

    raw_text = ""
    async for event in runner.run_async(
        user_id="user", session_id=session.id, new_message=content
    ):
        if event.is_final_response():
            for part in event.content.parts:
                if part.text:
                    raw_text += part.text

    return parse_agent_output(raw_text)


# ── CLI entry point ──────────────────────────────────────────────────────────


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Medical Triage Agent (CLI)")
    parser.add_argument("symptoms", nargs="?", help="Symptom description (free text)")
    parser.add_argument("--age-range", choices=["child", "teen", "adult", "older_adult"])
    parser.add_argument("--sex", dest="sex_at_birth")
    parser.add_argument("--pregnant", dest="pregnancy_possible", action="store_true", default=None)
    parser.add_argument("--duration")
    parser.add_argument("--severity", choices=["mild", "moderate", "severe"])
    parser.add_argument("--temp", type=float, dest="temperature_f")
    parser.add_argument("--hr", type=int, dest="heart_rate_bpm")
    parser.add_argument("--conditions", nargs="*", dest="existing_conditions")
    parser.add_argument("--medications", nargs="*", dest="current_medications")
    parser.add_argument("--allergies", nargs="*")
    parser.add_argument("--json-only", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    if not args.symptoms:
        print("Enter your symptoms (free text). Press Enter twice to submit.")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        args.symptoms = " ".join(lines)

    kwargs = {
        k: v
        for k, v in vars(args).items()
        if k not in ("symptoms", "json_only") and v is not None
    }

    structured, summary = await run_triage(args.symptoms, **kwargs)

    if args.json_only and structured:
        print(structured.model_dump_json(indent=2))
    else:
        if structured:
            print("\n" + "=" * 60)
            print("  STRUCTURED OUTPUT (JSON)")
            print("=" * 60)
            print(structured.model_dump_json(indent=2))
            print("\n" + "=" * 60)
            print("  HUMAN-READABLE SUMMARY")
            print("=" * 60)
            print(structured.to_human_readable())
        else:
            print("\n[Agent could not produce structured output. Raw summary below.]\n")
            print(summary)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
