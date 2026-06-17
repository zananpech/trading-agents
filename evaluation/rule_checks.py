"""
Rule-based evaluation checks — fast, deterministic, no LLM cost.

These checks verify structural and format requirements that don't require
semantic understanding:
  - Are all required report sections present?
  - Is the BUY/HOLD/SELL verdict stated clearly with confidence and horizon?
  - Does the fundamental analysis cover all 7 required dimensions?
"""
from __future__ import annotations

import re


# ── Required sections in the final report (report_writer output) ─────────────
_REPORT_SECTIONS = [
    "executive summary",
    "company overview",
    "valuation analysis",
    "financial health",
    "earnings quality",
    "news",          # "News & Market Sentiment"
    "key risks",
    "investment thesis",
]

# ── Required sections in the fundamental analysis (fundamental_analyst output) ─
_ANALYSIS_DIMENSIONS = [
    "valuation",
    "financial health",
    "profitab",          # "Profitability & Margins"
    "growth",
    "cash flow",
    "earnings quality",
    "news sentiment",
]
_ANALYSIS_STANCES = ["bullish", "neutral", "bearish"]


def check_report_structure(report: str) -> float:
    """
    Check that all 8 required sections are present in the final report.

    Returns a float in [0.0, 1.0] — fraction of sections found.
    """
    lower = report.lower()
    found = sum(1 for section in _REPORT_SECTIONS if section in lower)
    return round(found / len(_REPORT_SECTIONS), 3)


def check_verdict_clarity(report: str) -> float:
    """
    Check that the verdict is stated clearly with all required elements:
      1. A VERDICT: BUY / HOLD / SELL line
      2. A Confidence: X% value
      3. An Investment Horizon: ... value

    Returns a float in [0.0, 1.0] — fraction of elements present.
    """
    lower = report.lower()
    checks = [
        # Verdict keyword
        bool(re.search(r"\bverdict\s*:\s*(buy|hold|sell)\b", lower)),
        # Confidence percentage
        bool(re.search(r"\bconfidence\s*:\s*\d+\s*%", lower)),
        # Investment horizon
        bool(re.search(r"\binvestment\s+horizon\s*:", lower)),
    ]
    return round(sum(checks) / len(checks), 3)


def check_analysis_structure(analysis: str) -> float:
    """
    Check that the fundamental analyst's output covers all 7 required
    analysis dimensions plus a final stance (BULLISH / NEUTRAL / BEARISH).

    Returns a float in [0.0, 1.0] — fraction of expected elements found.
    """
    lower = analysis.lower()
    dimension_hits = sum(1 for dim in _ANALYSIS_DIMENSIONS if dim in lower)
    stance_hit = int(any(stance in lower for stance in _ANALYSIS_STANCES))
    total_checks = len(_ANALYSIS_DIMENSIONS) + 1  # 7 dimensions + stance
    found = dimension_hits + stance_hit
    return round(found / total_checks, 3)
