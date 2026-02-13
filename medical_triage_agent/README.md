# Medical Symptom Differential + Triage Agent

An educational medical-symptom triage agent built with **Google Agent Development Kit (ADK)**. It provides heuristic differential assessments grounded in a local medical knowledge base — no external APIs, no web browsing.

> **DISCLAIMER:** This agent is for **educational and informational purposes only**. It is **NOT** a substitute for professional medical evaluation, diagnosis, or treatment. Always consult a qualified healthcare professional.

---

## Architecture

```
User Input
    │
    ▼
┌──────────────────────────────────────────────┐
│  SequentialAgent (MedicalTriageAgent)         │
│                                              │
│  └─ LlmAgent (MedicalTriageLLM)             │
│       │                                      │
│       ├─ Tool: check_emergency_flags()       │
│       │   └─ Rule-based red-flag detection   │
│       │     (safety.py)                      │
│       │                                      │
│       ├─ Tool: retrieve_knowledge()          │
│       │   └─ Local RAG over medical_kb/      │
│       │     (rag.py — embeddings + cosine)   │
│       │                                      │
│       └─ LLM generates differential +       │
│          structured JSON + human summary     │
└──────────────────────────────────────────────┘
    │
    ▼
Structured JSON (TriageOutput) + Human-Readable Summary
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- A Google AI API key (`GOOGLE_API_KEY`) for Gemini models

### 2. Install

```bash
cd medical_triage_agent
pip install -r requirements.txt

# Optional: for better retrieval quality
pip install sentence-transformers
```

### 3. Set API Key

```bash
# Linux / macOS
export GOOGLE_API_KEY="your-api-key-here"

# Windows (PowerShell)
$env:GOOGLE_API_KEY="your-api-key-here"

# Windows (cmd)
set GOOGLE_API_KEY=your-api-key-here
```

### 4. Run the Agent (CLI)

```bash
# Interactive mode
python agent.py

# One-shot with arguments
python agent.py "sore throat, fever, body aches for 3 days" \
    --age-range adult --severity moderate --duration "3 days"

# JSON-only output
python agent.py "runny nose, sneezing, mild cough" --json-only
```

### 5. Run with ADK Dev UI

```bash
adk web
# Then open http://localhost:8000 and select "MedicalTriageAgent"
```

For the ADK dev UI, ensure `agent.py` exposes `root_agent` at module level (it does).

### 6. Run Evaluation

```bash
# Quick: red-flag unit tests only (no LLM, no API key needed)
python eval.py --quick

# Full integration tests (requires API key + LLM)
python eval.py

# Single test case
python eval.py --case 3
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | (required) | Google AI API key |
| `MODEL_ID` | `gemini-2.0-flash` | Gemini model to use |
| `MEDICAL_KB_DIR` | `./medical_kb/` | Path to knowledge base files |
| `RAG_TOP_K` | `8` | Number of KB chunks to retrieve |

## Project Structure

```
medical_triage_agent/
├── agent.py              # ADK agent definition + CLI entry point
├── rag.py                # Local RAG retrieval (embeddings + cosine similarity)
├── schemas.py            # Pydantic input/output schemas
├── safety.py             # Red-flag rules + safety system prompts
├── eval.py               # 10-case evaluation harness
├── requirements.txt
├── __init__.py
└── medical_kb/           # Knowledge base (plain text / markdown)
    ├── common_conditions.md
    ├── emergencies.md
    ├── respiratory.md
    ├── abdominal_genitourinary.md
    └── mental_health.md
```

## Output Schema

The agent outputs both:

1. **Structured JSON** matching `TriageOutput` (see `schemas.py`)
2. **Human-readable summary** with emergency banners, differential table, clarifying questions, and next steps

See `schemas.py` for the full Pydantic model definition.

## Safety Features

- Rule-based red-flag detection runs **before** LLM (no LLM hallucination can suppress emergencies)
- Emergency override: if red flags detected, emergency banner is always shown first
- System prompt enforces: no diagnosis claims, no prescription changes, mandatory disclaimer
- Percentages explicitly labeled as "heuristic educational estimates, not clinical probabilities"
- All reasoning grounded in retrieved KB chunks with citations
- Clarifying questions asked when input is insufficient

## Extending the Knowledge Base

Add `.md` or `.txt` files to `medical_kb/`. Use markdown headings (`##`) to define sections — each heading-delimited section becomes a retrievable chunk. Include:

- Condition name and description
- Typical symptoms and base rates
- Key differentiators from similar conditions
- Red flags and complications
- Self-care guidance and "see a clinician if" criteria

The RAG module will automatically index new files on startup.
