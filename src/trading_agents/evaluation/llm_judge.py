"""
LLM-as-a-Judge evaluators.

Uses the same Gemini flash model as the pipeline to evaluate two dimensions
that require semantic understanding:
  - reasoning_quality  : Are the report's claims backed by specific numbers?
  - hallucination_risk : Does the report cite figures that match the raw data?

Each judge function returns (score: float, rationale: str) where:
  - score is normalized to [0.0, 1.0]
  - rationale is a 1-2 sentence explanation for terminal display

Judge prompts are structured to return a JSON object for reliable parsing.
"""
from __future__ import annotations

import json
import re
import warnings

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from trading_agents.config import GEMINI_MODEL, GOOGLE_API_KEY

# ── Shared judge LLM (low temperature for deterministic scoring) ──────────────
def _get_judge_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,  # Fully deterministic for scoring
    )


def _parse_judge_response(content: str) -> tuple[float, str]:
    """
    Extract (score, rationale) from the judge's JSON response.
    Falls back gracefully if the response is malformed.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", content).strip().strip("`").strip()
    try:
        data = json.loads(cleaned)
        raw_score = float(data.get("score", 5))
        rationale = str(data.get("rationale", "No rationale provided."))
        # Normalize score from 0–10 to 0.0–1.0
        normalized = round(max(0.0, min(10.0, raw_score)) / 10.0, 3)
        return normalized, rationale
    except (json.JSONDecodeError, ValueError, TypeError):
        # Try to find a number anywhere in the response as a fallback
        numbers = re.findall(r"\b([0-9](?:\.[0-9]+)?|10)\b", content)
        if numbers:
            raw = float(numbers[0])
            normalized = round(max(0.0, min(10.0, raw)) / 10.0, 3)
        else:
            normalized = 0.5  # Unknown — default to middle
        return normalized, "Score extracted from unstructured judge response."


# ── Judge 1: Reasoning Quality ────────────────────────────────────────────────
_REASONING_SYSTEM = """You are a rigorous financial research quality evaluator.
Your job is to assess whether an equity research report's investment thesis is
properly backed by specific, quantitative data from the underlying financial data provided.

Scoring rubric (0–10):
  10: Every major claim cites specific numbers (e.g., "P/E of 24.3x", "revenue grew 18% YoY")
   8: Most claims are quantified; minor conclusions are qualitative but acceptable
   6: Mix of quantified and vague claims; thesis is plausible but lightly supported
   4: Mostly vague/qualitative assertions; few hard numbers cited
   2: Thesis is largely unsupported; no meaningful quantitative backing
   0: Report is entirely generic with no data references whatsoever

You MUST respond with valid JSON only, no markdown:
{"score": <0-10>, "rationale": "<1-2 sentence explanation>"}"""

_REASONING_USER_TEMPLATE = """--- RAW FINANCIAL DATA (source of truth) ---
{raw_data_summary}

--- EQUITY RESEARCH REPORT TO EVALUATE ---
{report}

Evaluate the reasoning quality. Respond with JSON only."""


def judge_reasoning_quality(report: str, raw_data: dict) -> tuple[float, str]:
    """
    LLM judge: score how well the report's claims are backed by specific
    quantitative data from raw_data.

    Returns (score: float [0.0–1.0], rationale: str)
    """
    # Summarise raw_data to the most relevant fields (avoid token overload)
    raw_summary = _summarise_raw_data(raw_data)

    user_content = _REASONING_USER_TEMPLATE.format(
        raw_data_summary=raw_summary,
        report=report[:6000],  # Truncate very long reports
    )

    try:
        llm = _get_judge_llm()
        response = llm.invoke([
            SystemMessage(content=_REASONING_SYSTEM),
            HumanMessage(content=user_content),
        ])
        content = response.content if isinstance(response.content, str) else str(response.content)
        return _parse_judge_response(content)
    except Exception as exc:
        warnings.warn(f"Reasoning quality judge failed: {exc}", stacklevel=2)
        return 0.5, f"Judge unavailable: {exc}"


# ── Judge 2: Hallucination Risk ───────────────────────────────────────────────
_HALLUCINATION_SYSTEM = """You are a financial fact-checking evaluator.
Your job is to assess whether specific financial figures cited in an equity research report
are consistent with (or at least plausible given) the raw financial data provided.

Scoring rubric (0–10):
  10: All cited figures match or are directly derivable from the raw data
   8: Most figures match; 1-2 minor discrepancies or rounding differences
   6: Some figures match; a few seem invented or are not verifiable
   4: Many figures appear to be fabricated or contradict the data
   2: Almost no figures can be verified against the raw data
   0: Report invents financial data wholesale; completely ungrounded

A HIGH score means LOW hallucination risk (good). A LOW score means HIGH risk (bad).

You MUST respond with valid JSON only, no markdown:
{"score": <0-10>, "rationale": "<1-2 sentence explanation>"}"""

_HALLUCINATION_USER_TEMPLATE = """--- RAW FINANCIAL DATA (source of truth) ---
{raw_data_summary}

--- EQUITY RESEARCH REPORT TO FACT-CHECK ---
{report}

Evaluate grounding of the cited figures. Respond with JSON only."""


def judge_hallucination_risk(report: str, raw_data: dict) -> tuple[float, str]:
    """
    LLM judge: score how well the specific financial figures in the report
    are grounded in the raw financial data.

    Returns (score: float [0.0–1.0], rationale: str)
    A score of 1.0 means no hallucination detected; 0.0 means severe hallucination.
    """
    raw_summary = _summarise_raw_data(raw_data)

    user_content = _HALLUCINATION_USER_TEMPLATE.format(
        raw_data_summary=raw_summary,
        report=report[:6000],
    )

    try:
        llm = _get_judge_llm()
        response = llm.invoke([
            SystemMessage(content=_HALLUCINATION_SYSTEM),
            HumanMessage(content=user_content),
        ])
        content = response.content if isinstance(response.content, str) else str(response.content)
        return _parse_judge_response(content)
    except Exception as exc:
        warnings.warn(f"Hallucination risk judge failed: {exc}", stacklevel=2)
        return 0.5, f"Judge unavailable: {exc}"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _summarise_raw_data(raw_data: dict) -> str:
    """
    Flatten raw_data dict to a concise string for judge prompts.
    Keeps valuation metrics and financials (most relevant for fact-checking);
    truncates to avoid token overload.
    """
    priority_keys = ["valuation_metrics", "financials", "earnings", "stock_info"]
    parts = []
    for key in priority_keys:
        if key in raw_data:
            value = str(raw_data[key])
            parts.append(f"[{key.upper()}]\n{value[:1500]}")
    # Append remaining keys at lower priority
    for key, value in raw_data.items():
        if key not in priority_keys:
            parts.append(f"[{key.upper()}]\n{str(value)[:500]}")
    return "\n\n".join(parts)[:8000]
