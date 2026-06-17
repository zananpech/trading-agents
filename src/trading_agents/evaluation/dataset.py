"""
Langfuse Dataset Manager for regression testing.

Manages a Langfuse Dataset named (by default) "trading-agents-eval".
Each dataset item represents one ticker run with its raw_data as input.

Usage
-----
  # Add the current run's data to the dataset:
  from trading_agents.evaluation.dataset import upsert_dataset_item
  upsert_dataset_item(ticker, raw_data)

  # CLI: replay the whole dataset and re-score (see eval_cli.py):
  from trading_agents.evaluation.dataset import run_dataset_eval
  run_dataset_eval()
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any


def upsert_dataset_item(
    ticker: str,
    raw_data: dict[str, Any],
    dataset_name: str | None = None,
) -> None:
    """
    Add or update a dataset item for the given ticker.

    Creates the Langfuse dataset if it doesn't exist yet. The item's
    external_id is the ticker symbol, so re-running the same ticker
    updates the existing item rather than creating a duplicate.

    Parameters
    ----------
    ticker : str
        The stock ticker (e.g. "AAPL").
    raw_data : dict
        The raw financial data fetched by the pipeline for this ticker.
    dataset_name : str | None
        Dataset name override. Defaults to config.EVAL_DATASET_NAME.
    """
    from trading_agents.config import EVAL_DATASET_NAME, LANGFUSE_ENABLED
    if not LANGFUSE_ENABLED:
        return

    name = dataset_name or EVAL_DATASET_NAME

    try:
        from langfuse import get_client
        client = get_client()

        # Ensure the dataset exists (get_or_create pattern)
        try:
            client.get_dataset(name)
        except Exception:
            client.create_dataset(
                name=name,
                description=(
                    "Benchmark dataset for trading-agents evaluation. "
                    "Each item is a ticker with its fetched financial data."
                ),
            )

        # Upsert the item (external_id = ticker for deduplication)
        client.create_dataset_item(
            dataset_name=name,
            input={
                "ticker": ticker,
                "raw_data": raw_data,
                "added_at": datetime.now(timezone.utc).isoformat(),
            },
            # Use ticker as external_id so same ticker = update, not duplicate
            id=ticker.upper(),
        )
    except Exception as exc:
        warnings.warn(
            f"Failed to upsert dataset item for {ticker}: {exc}",
            stacklevel=2,
        )


def run_dataset_eval(
    dataset_name: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """
    Replay all items in the benchmark dataset and score each one.

    For each dataset item:
      1. Re-runs the full pipeline for that ticker
      2. Evaluates the output
      3. Logs scores back to Langfuse as a dataset run

    Parameters
    ----------
    dataset_name : str | None
        Dataset name to replay. Defaults to config.EVAL_DATASET_NAME.
    run_name : str | None
        Name for this evaluation run (e.g., "prompt-v2-experiment").
        Defaults to a timestamp-based name.

    Returns
    -------
    dict
        Summary of scores across all tickers.
    """
    from trading_agents.config import EVAL_DATASET_NAME, LANGFUSE_ENABLED
    if not LANGFUSE_ENABLED:
        raise RuntimeError(
            "Langfuse is not configured. Set LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY in your .env to use dataset evaluation."
        )

    name = dataset_name or EVAL_DATASET_NAME
    run_label = run_name or f"eval-run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    from langfuse import get_client
    from trading_agents.evaluation.evaluator import evaluate_pipeline_run
    from trading_agents.graph import pipeline
    from langchain_core.runnables import RunnableConfig
    from trading_agents.observability import get_callback_handler, get_trace_id, flush
    from trading_agents.state import AgentState

    client = get_client()
    dataset = client.get_dataset(name)
    items = dataset.items

    if not items:
        print(f"Dataset '{name}' is empty. Run some tickers first to populate it.")
        return {}

    results: dict[str, Any] = {}

    for item in items:
        ticker = item.input.get("ticker", "UNKNOWN")
        raw_data = item.input.get("raw_data", {})

        print(f"\n  🔄 Evaluating {ticker}...")

        # Build initial state from stored data (skip data fetch — use cached)
        initial_state: AgentState = {
            "ticker": ticker,
            "raw_data": raw_data,
            "fundamental_analysis": "",
            "final_report": "",
        }

        handler = get_callback_handler(ticker)
        run_config: RunnableConfig = {"callbacks": [handler]} if handler else {}

        final_state: AgentState | None = None
        for event in pipeline.stream(initial_state, config=run_config):
            node_name = list(event.keys())[0]
            final_state = event[node_name]

        if handler:
            flush()

        if not final_state:
            print(f"  ⚠️  Pipeline produced no output for {ticker}. Skipping.")
            continue

        trace_id = get_trace_id(handler) if handler else None
        eval_result = evaluate_pipeline_run(state=final_state, trace_id=trace_id)

        # Log to Langfuse dataset run
        if trace_id:
            try:
                item.link(
                    trace_or_observation=trace_id,
                    run_name=run_label,
                )
            except Exception as exc:
                warnings.warn(f"Failed to link dataset item for {ticker}: {exc}", stacklevel=2)

        results[ticker] = {
            "overall": eval_result.overall,
            "analysis_structure": eval_result.analysis_structure.value,
            "report_structure": eval_result.report_structure.value,
            "verdict_clarity": eval_result.verdict_clarity.value,
            "reasoning_quality": eval_result.reasoning_quality.value,
            "hallucination_risk": eval_result.hallucination_risk.value,
        }
        print(f"  ✅ {ticker}: overall={eval_result.overall:.2f}")

    return results
