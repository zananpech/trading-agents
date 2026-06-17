"""
Evaluation Orchestrator.

The single public entry point for evaluation:
    evaluate_pipeline_run(state, trace_id) -> EvalResult

Runs all rule-based checks and LLM judge calls, uploads scores to Langfuse,
and returns a structured EvalResult for terminal display.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from trading_agents.evaluation.llm_judge import judge_hallucination_risk, judge_reasoning_quality
from trading_agents.evaluation.rule_checks import (
    check_analysis_structure,
    check_report_structure,
    check_verdict_clarity,
)
from trading_agents.evaluation.scores import (
    SCORE_ANALYSIS_STRUCTURE,
    SCORE_HALLUCINATION,
    SCORE_REASONING,
    SCORE_STRUCTURE,
    SCORE_VERDICT,
    EvalResult,
    ScoreDimension,
)

if TYPE_CHECKING:
    from trading_agents.state import AgentState


def evaluate_pipeline_run(
    state: "AgentState",
    trace_id: str | None = None,
) -> EvalResult:
    """
    Run all evaluations for a completed pipeline run.

    Parameters
    ----------
    state : AgentState
        The final pipeline state containing ticker, raw_data,
        fundamental_analysis, and final_report.
    trace_id : str | None
        The Langfuse trace ID to attach scores to. If None or Langfuse
        is disabled, scores are computed but not uploaded.

    Returns
    -------
    EvalResult
        All dimension scores + overall score for terminal display.
    """
    ticker = state.get("ticker", "UNKNOWN")
    raw_data = state.get("raw_data", {})
    analysis = state.get("fundamental_analysis", "")
    report = state.get("final_report", "")

    result = EvalResult(ticker=ticker)

    # ── 1. Rule-based: analysis structure (fundamental_analyst) ──────────────
    analysis_score = check_analysis_structure(analysis)
    result.analysis_structure = ScoreDimension(
        name=SCORE_ANALYSIS_STRUCTURE,
        label="Analysis Structure",
        value=analysis_score,
        method="rule-based",
        rationale="",
    )

    # ── 2. Rule-based: report structure (report_writer) ───────────────────────
    report_score = check_report_structure(report)
    result.report_structure = ScoreDimension(
        name=SCORE_STRUCTURE,
        label="Report Structure",
        value=report_score,
        method="rule-based",
        rationale="",
    )

    # ── 3. Rule-based: verdict clarity ────────────────────────────────────────
    verdict_score = check_verdict_clarity(report)
    result.verdict_clarity = ScoreDimension(
        name=SCORE_VERDICT,
        label="Verdict Clarity",
        value=verdict_score,
        method="rule-based",
        rationale="",
    )

    # ── 4. LLM judge: reasoning quality ──────────────────────────────────────
    reasoning_score, reasoning_rationale = judge_reasoning_quality(report, raw_data)
    result.reasoning_quality = ScoreDimension(
        name=SCORE_REASONING,
        label="Reasoning Quality",
        value=reasoning_score,
        method="llm-judge",
        rationale=reasoning_rationale,
    )

    # ── 5. LLM judge: hallucination risk ─────────────────────────────────────
    hallucination_score, hallucination_rationale = judge_hallucination_risk(report, raw_data)
    result.hallucination_risk = ScoreDimension(
        name=SCORE_HALLUCINATION,
        label="Hallucination Risk",
        value=hallucination_score,
        method="llm-judge",
        rationale=hallucination_rationale,
    )

    # ── 6. Upload scores to Langfuse (if enabled + trace_id available) ────────
    if trace_id:
        _upload_scores_to_langfuse(result, trace_id)

    return result


def _upload_scores_to_langfuse(result: EvalResult, trace_id: str) -> None:
    """Upload all dimension scores to Langfuse attached to the given trace."""
    from trading_agents.config import LANGFUSE_ENABLED
    if not LANGFUSE_ENABLED:
        return

    try:
        from langfuse import get_client
        client = get_client()

        for dim in result.dimensions:
            client.create_score(
                trace_id=trace_id,
                name=dim.name,
                value=dim.value,
                comment=dim.rationale or None,
            )

        # Also upload the overall weighted score
        client.create_score(
            trace_id=trace_id,
            name="overall_quality",
            value=result.overall,
            comment=f"Weighted average across {len(result.dimensions)} dimensions",
        )
    except Exception as exc:
        warnings.warn(
            f"Failed to upload eval scores to Langfuse: {exc}",
            stacklevel=2,
        )
