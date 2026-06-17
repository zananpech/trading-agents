"""
Trading Agents — Evaluation Package

Provides automatic quality scoring for every pipeline run using:
  - Rule-based checks (fast, no LLM cost)
  - LLM-as-a-judge (Gemini flash, same model as pipeline)

Scores are attached to Langfuse traces and printed to the terminal.
"""
from trading_agents.evaluation.evaluator import evaluate_pipeline_run
from trading_agents.evaluation.scores import EvalResult

__all__ = ["evaluate_pipeline_run", "EvalResult"]
