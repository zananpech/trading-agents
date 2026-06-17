"""
Evaluation score names (constants) and result dataclasses.

Score semantics — all values are in [0.0, 1.0]:
  SCORE_STRUCTURE     : Fraction of required report sections present
  SCORE_VERDICT       : Fraction of required verdict elements present (BUY/HOLD/SELL, confidence %, horizon)
  SCORE_REASONING     : LLM judge score for data-backed reasoning quality
  SCORE_HALLUCINATION : LLM judge score for grounding (1.0 = no hallucination detected)

Agent-level scores for fundamental_analyst:
  SCORE_ANALYSIS_STRUCTURE : Fraction of required analysis sections present
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ── Final report scores (report_writer output) ────────────────────────────────
SCORE_STRUCTURE = "report_structure_completeness"
SCORE_VERDICT = "verdict_clarity"
SCORE_REASONING = "reasoning_quality"
SCORE_HALLUCINATION = "hallucination_risk"

# ── Agent-level score (fundamental_analyst output) ────────────────────────────
SCORE_ANALYSIS_STRUCTURE = "analysis_structure_completeness"


@dataclass
class ScoreDimension:
    """A single scored dimension with its value, label, and optional rationale."""
    name: str
    label: str
    value: float          # 0.0 – 1.0
    method: str           # "rule-based" | "llm-judge"
    rationale: str = ""   # Populated by LLM judge; empty for rule-based


@dataclass
class EvalResult:
    """
    Container for all evaluation scores from a single pipeline run.

    Dimensions
    ----------
    analysis_structure  : fundamental_analyst output structure check
    report_structure    : report_writer output section completeness
    verdict_clarity     : BUY/HOLD/SELL + confidence + horizon present
    reasoning_quality   : LLM-judge — data-backed reasoning
    hallucination_risk  : LLM-judge — grounding vs raw financial data
    """
    ticker: str
    analysis_structure: ScoreDimension = field(default_factory=lambda: ScoreDimension(
        name=SCORE_ANALYSIS_STRUCTURE,
        label="Analysis Structure",
        value=0.0,
        method="rule-based",
    ))
    report_structure: ScoreDimension = field(default_factory=lambda: ScoreDimension(
        name=SCORE_STRUCTURE,
        label="Report Structure",
        value=0.0,
        method="rule-based",
    ))
    verdict_clarity: ScoreDimension = field(default_factory=lambda: ScoreDimension(
        name=SCORE_VERDICT,
        label="Verdict Clarity",
        value=0.0,
        method="rule-based",
    ))
    reasoning_quality: ScoreDimension = field(default_factory=lambda: ScoreDimension(
        name=SCORE_REASONING,
        label="Reasoning Quality",
        value=0.0,
        method="llm-judge",
    ))
    hallucination_risk: ScoreDimension = field(default_factory=lambda: ScoreDimension(
        name=SCORE_HALLUCINATION,
        label="Hallucination Risk",
        value=0.0,
        method="llm-judge",
    ))

    @property
    def overall(self) -> float:
        """Weighted average across all dimensions."""
        weights = {
            "report_structure": 0.20,
            "verdict_clarity": 0.20,
            "analysis_structure": 0.15,
            "reasoning_quality": 0.25,
            "hallucination_risk": 0.20,
        }
        total = (
            self.report_structure.value * weights["report_structure"]
            + self.verdict_clarity.value * weights["verdict_clarity"]
            + self.analysis_structure.value * weights["analysis_structure"]
            + self.reasoning_quality.value * weights["reasoning_quality"]
            + self.hallucination_risk.value * weights["hallucination_risk"]
        )
        return round(total, 3)

    @property
    def dimensions(self) -> list[ScoreDimension]:
        """All dimensions in display order."""
        return [
            self.analysis_structure,
            self.report_structure,
            self.verdict_clarity,
            self.reasoning_quality,
            self.hallucination_risk,
        ]
