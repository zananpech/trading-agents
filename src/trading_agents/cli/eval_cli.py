#!/usr/bin/env python
"""
Trading Agents — Evaluation CLI

Standalone tool for running regression evaluations against a Langfuse Dataset.

Usage
-----
  # Run the full benchmark dataset and score all tickers:
  uv run trading-agents-eval

  # Run against a specific named dataset:
  uv run trading-agents-eval --dataset my-custom-dataset

  # Add tickers to the benchmark dataset (runs pipeline + adds to dataset):
  uv run trading-agents-eval --add AAPL MSFT TSLA

  # Name this evaluation run (shows in Langfuse UI for comparison):
  uv run trading-agents-eval --run-name prompt-v2-experiment
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

THEME = Theme(
    {
        "header": "bold cyan",
        "success": "bold green",
        "error": "bold red",
        "info": "dim cyan",
        "warn": "bold yellow",
    }
)

console = Console(theme=THEME, width=120)


def _score_color(value: float) -> str:
    if value >= 0.85:
        return "bold green"
    if value >= 0.65:
        return "bold yellow"
    return "bold red"


def print_summary_table(results: dict) -> None:
    """Print a summary table of all ticker scores from a dataset eval run."""
    if not results:
        return

    console.print()
    console.print(Rule(title="📊  DATASET EVALUATION SUMMARY", style="cyan"))
    console.print()

    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim cyan",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Ticker", style="bold yellow", min_width=8)
    table.add_column("Analysis", justify="center", min_width=10)
    table.add_column("Structure", justify="center", min_width=10)
    table.add_column("Verdict", justify="center", min_width=10)
    table.add_column("Reasoning", justify="center", min_width=10)
    table.add_column("Hallucination", justify="center", min_width=13)
    table.add_column("Overall", justify="center", min_width=10)

    for ticker, scores in sorted(results.items()):
        def fmt(v: float) -> str:
            return f"[{_score_color(v)}]{v:.2f}[/{_score_color(v)}]"

        table.add_row(
            ticker,
            fmt(scores.get("analysis_structure", 0)),
            fmt(scores.get("report_structure", 0)),
            fmt(scores.get("verdict_clarity", 0)),
            fmt(scores.get("reasoning_quality", 0)),
            fmt(scores.get("hallucination_risk", 0)),
            fmt(scores.get("overall", 0)),
        )

    console.print(table)

    # Aggregate stats
    if results:
        avg_overall = sum(s["overall"] for s in results.values()) / len(results)
        color = _score_color(avg_overall)
        console.print()
        console.print(
            f"  Mean overall quality across {len(results)} tickers: "
            f"[{color}]{avg_overall:.3f}[/{color}]"
        )
    console.print()


def cmd_add(tickers: list[str], dataset_name: str) -> None:
    """Run the pipeline for each ticker and add results to the dataset."""
    from trading_agents.graph import pipeline
    from langchain_core.runnables import RunnableConfig
    from trading_agents.observability import flush, get_callback_handler, get_trace_id
    from trading_agents.evaluation.dataset import upsert_dataset_item
    from trading_agents.state import AgentState

    console.print(
        f"\n  Adding {len(tickers)} ticker(s) to dataset [bold cyan]'{dataset_name}'[/bold cyan]...\n"
    )

    for ticker in tickers:
        ticker = ticker.upper()
        console.print(f"  🔄  Running pipeline for [bold yellow]{ticker}[/bold yellow]...")

        initial_state: AgentState = {
            "ticker": ticker,
            "raw_data": {},
            "fundamental_analysis": "",
            "final_report": "",
        }

        handler = get_callback_handler(ticker)
        run_config: RunnableConfig = {"callbacks": [handler]} if handler else {}

        final_state: AgentState | None = None
        try:
            for event in pipeline.stream(initial_state, config=run_config):
                node_name = list(event.keys())[0]
                final_state = event[node_name]
        except Exception as exc:
            console.print(f"  [error]❌ Pipeline failed for {ticker}: {exc}[/error]")
            continue

        if handler:
            flush()

        if final_state:
            upsert_dataset_item(
                ticker=final_state.get("ticker", ticker),
                raw_data=final_state.get("raw_data", {}),
                dataset_name=dataset_name,
            )
            console.print(f"  [success]✅ {ticker} added to dataset.[/success]")
        else:
            console.print(f"  [warn]⚠️  Pipeline produced no output for {ticker}. Skipped.[/warn]")


def cmd_run(dataset_name: str, run_name: str | None) -> None:
    """Replay all items in the dataset and score them."""
    from trading_agents.evaluation.dataset import run_dataset_eval

    console.print(
        f"\n  Running evaluation on dataset [bold cyan]'{dataset_name}'[/bold cyan]"
        + (f" (run: [bold]{run_name}[/bold])" if run_name else "")
        + "...\n"
    )
    results = run_dataset_eval(dataset_name=dataset_name, run_name=run_name)
    print_summary_table(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trading-agents-eval",
        description="Trading Agents — Langfuse Evaluation CLI",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Langfuse dataset name to use (default: from EVAL_DATASET_NAME env or 'trading-agents-eval')",
    )
    parser.add_argument(
        "--add",
        nargs="+",
        metavar="TICKER",
        help="Run the pipeline for these tickers and add them to the benchmark dataset",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        dest="run_name",
        help="Name for this evaluation run in Langfuse (useful for comparing experiments)",
    )

    args = parser.parse_args()

    # Resolve dataset name
    from trading_agents.config import EVAL_DATASET_NAME
    dataset_name = args.dataset or EVAL_DATASET_NAME

    console.print()
    console.print(
        "  🤖 [bold cyan]Trading Agents[/bold cyan] · Evaluation CLI",
        style="dim",
    )
    console.print()

    try:
        if args.add:
            cmd_add(tickers=args.add, dataset_name=dataset_name)
        else:
            cmd_run(dataset_name=dataset_name, run_name=args.run_name)
    except EnvironmentError as e:
        console.print(f"\n[error]❌ Configuration Error:[/error]\n{e}")
        sys.exit(1)
    except RuntimeError as e:
        console.print(f"\n[error]❌ Error:[/error]\n{e}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Evaluation cancelled.[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
