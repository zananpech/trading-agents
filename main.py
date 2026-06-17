#!/usr/bin/env python
"""
Trading Agents — Multi-Agent Stock Research System
CLI entry point.

Usage:
    uv run main.py AAPL
    uv run main.py MSFT --save
    uv run main.py TSLA --save --output-dir my_reports
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ── Rich theme ────────────────────────────────────────────────────────────────
THEME = Theme(
    {
        "header": "bold cyan",
        "ticker": "bold yellow",
        "step": "dim white",
        "success": "bold green",
        "error": "bold red",
        "info": "dim cyan",
        "verdict.buy": "bold green on dark_green",
        "verdict.hold": "bold yellow on dark_goldenrod",
        "verdict.sell": "bold red on dark_red",
    }
)

console = Console(theme=THEME, width=120)


def print_banner() -> None:
    banner = Text()
    banner.append("  🤖 TRADING AGENTS", style="bold cyan")
    banner.append("  |  Multi-Agent Stock Research System", style="dim white")
    console.print(
        Panel(
            banner,
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print(
        "  Powered by [bold]Google Gemini 3.1 flash[/bold] · "
        "[bold]LangGraph[/bold] · "
        "[bold]Yahoo Finance[/bold]",
        style="dim white",
    )
    console.print()


def print_step(icon: str, title: str, subtitle: str = "") -> None:
    text = Text()
    text.append(f" {icon}  ", style="cyan")
    text.append(title, style="bold white")
    if subtitle:
        text.append(f"  {subtitle}", style="dim white")
    console.print(text)


def detect_verdict(report: str) -> str:
    """Extract BUY/HOLD/SELL from the report for a highlighted summary line."""
    upper = report.upper()
    if "VERDICT: BUY" in upper or "**BUY**" in upper:
        return "BUY"
    if "VERDICT: SELL" in upper or "**SELL**" in upper:
        return "SELL"
    return "HOLD"


def _score_bar(value: float, width: int = 16) -> str:
    """Render a simple unicode progress bar for a score in [0.0, 1.0]."""
    filled = round(value * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def _score_color(value: float) -> str:
    """Pick a rich color based on score value."""
    if value >= 0.85:
        return "bold green"
    if value >= 0.65:
        return "bold yellow"
    return "bold red"


def print_eval_results(eval_result) -> None:
    """Render the evaluation results as a rich table in the terminal."""

    console.print()
    console.print(Rule(title="🎯  EVALUATION RESULTS", style="cyan"))
    console.print()

    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim cyan",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Dimension", style="white", min_width=26)
    table.add_column("Method", style="dim white", min_width=12)
    table.add_column("Score", justify="center", min_width=20)
    table.add_column("Value", justify="right", min_width=6)

    for dim in eval_result.dimensions:
        bar = _score_bar(dim.value)
        color = _score_color(dim.value)
        method_label = "[dim cyan]llm-judge[/dim cyan]" if dim.method == "llm-judge" else "[dim white]rule-based[/dim white]"
        table.add_row(
            dim.label,
            method_label,
            f"[{color}]{bar}[/{color}]",
            f"[{color}]{dim.value:.2f}[/{color}]",
        )

    # Separator + overall
    overall_color = _score_color(eval_result.overall)
    table.add_section()
    table.add_row(
        "[bold white]Overall Quality[/bold white]",
        "[dim white]weighted avg[/dim white]",
        f"[{overall_color}]{_score_bar(eval_result.overall)}[/{overall_color}]",
        f"[{overall_color}]{eval_result.overall:.2f}[/{overall_color}]",
    )

    console.print(table)

    # Print LLM judge rationales
    llm_dims = [d for d in eval_result.dimensions if d.method == "llm-judge" and d.rationale]
    if llm_dims:
        console.print()
        for dim in llm_dims:
            console.print(
                f"  [dim cyan]{dim.label}:[/dim cyan] [dim white]{dim.rationale}[/dim white]"
            )

    console.print()


def save_report(ticker: str, report: str, output_dir: str) -> Path:
    """Save the report to a markdown file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    file_path = out / f"{ticker.upper()}_{date_str}.md"
    file_path.write_text(report, encoding="utf-8")
    return file_path


def run_analysis(ticker: str, save: bool, output_dir: str) -> None:
    """Run the full multi-agent pipeline for the given ticker."""
    # Late import so config validation happens after arg parsing
    from graph import pipeline
    from langchain_core.runnables import RunnableConfig
    from observability import flush, get_callback_handler, get_trace_id
    from state import AgentState

    print_banner()
    console.print(Rule(style="cyan"))
    console.print(
        f"  Analyzing [ticker]{ticker.upper()}[/ticker]",
        style="bold",
    )
    console.print(Rule(style="cyan"))
    console.print()

    # ── Langfuse tracing (optional) ───────────────────────────────────────────
    langfuse_handler = get_callback_handler(ticker)
    if langfuse_handler:
        from config import LANGFUSE_HOST
        console.print(
            f"  [dim]🔭 Langfuse tracing active → [link={LANGFUSE_HOST}]{LANGFUSE_HOST}[/link][/dim]"
        )
        console.print()
    run_config: RunnableConfig = {"callbacks": [langfuse_handler]} if langfuse_handler else {}

    initial_state: AgentState = {
        "ticker": ticker.upper(),
        "raw_data": {},
        "fundamental_analysis": "",
        "final_report": "",
    }

    steps = [
        ("📡", "Fetching market data from Yahoo Finance..."),
        ("🧠", "Fundamental Analyst Agent running..."),
        ("✍️ ", "Report Writer Agent composing final report..."),
    ]

    final_state: AgentState | None = None
    eval_result = None  # Will be populated after pipeline + eval run

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        # We run the full pipeline — steps are shown sequentially
        task = progress.add_task(steps[0][1], total=None)

        # Manually stream node by node using the stream API
        for i, event in enumerate(pipeline.stream(initial_state, config=run_config)):
            node_name = list(event.keys())[0]

            step_idx = {"fetch_data": 0, "fundamental_analyst": 1, "report_writer": 2}.get(node_name, i)
            if step_idx + 1 < len(steps):
                progress.update(task, description=steps[step_idx + 1][1])
            else:
                progress.update(task, description="✅  Finalizing report...")

            final_state = event[node_name]

    # Flush Langfuse spans before rendering (ensures traces reach the server)
    if langfuse_handler:
        flush()

    # ── Evaluation ────────────────────────────────────────────────────────────
    if final_state:
        from evaluation.evaluator import evaluate_pipeline_run
        from evaluation.dataset import upsert_dataset_item

        trace_id = get_trace_id(langfuse_handler)
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as eval_progress:
            eval_progress.add_task("🎯  Running evaluations...", total=None)
            eval_result = evaluate_pipeline_run(state=final_state, trace_id=trace_id)

        # Save to Langfuse Dataset for regression testing
        upsert_dataset_item(
            ticker=final_state.get("ticker", ticker),
            raw_data=final_state.get("raw_data", {}),
        )

    console.print()
    console.print(Rule(title="📋  RESEARCH REPORT", style="cyan"))
    console.print()

    if final_state and final_state.get("final_report"):
        report = final_state["final_report"]

        # Render the report as rich markdown
        console.print(Markdown(report))

        # Highlight the verdict
        verdict = detect_verdict(report)
        verdict_style = {
            "BUY": "verdict.buy",
            "HOLD": "verdict.hold",
            "SELL": "verdict.sell",
        }.get(verdict, "verdict.hold")

        console.print()
        console.print(Rule(style="cyan"))
        console.print(
            Panel(
                Text(f"  FINAL VERDICT: {verdict}  ", style=verdict_style, justify="center"),
                border_style="cyan",
                padding=(0, 4),
            )
        )

        # Print evaluation results
        if eval_result is not None:
            print_eval_results(eval_result)

        # Save report if requested
        if save:
            path = save_report(ticker, report, output_dir)
            console.print(
                f"\n  [success]✅ Report saved →[/success] [info]{path}[/info]\n"
            )
    else:
        console.print("[error]❌ Pipeline produced no report. Check your API key and try again.[/error]")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trading-agents",
        description="Multi-Agent Stock Research System powered by Google Gemini & LangGraph",
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol to analyze (e.g., AAPL, MSFT, TSLA)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save the final report to a markdown file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save reports in (default: ./reports)",
    )

    args = parser.parse_args()
    ticker = args.ticker.strip().upper()

    if not ticker.isalpha():
        console.print(f"[error]❌ Invalid ticker: '{ticker}'. Ticker must contain only letters.[/error]")
        sys.exit(1)

    try:
        run_analysis(ticker=ticker, save=args.save, output_dir=args.output_dir)
    except EnvironmentError as e:
        console.print(f"\n[error]❌ Configuration Error:[/error]\n{e}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Analysis cancelled.[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
