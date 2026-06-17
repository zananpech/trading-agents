"""
Trading Agents — Evaluation Package

Provides automatic quality scoring for every pipeline run using:
  - Rule-based checks (fast, no LLM cost)
  - LLM-as-a-judge (Gemini flash, same model as pipeline)

Scores are attached to Langfuse traces and printed to the terminal.
"""
from evaluation.evaluator import evaluate_pipeline_run
from evaluation.scores import EvalResult

__all__ = ["evaluate_pipeline_run", "EvalResult"]
