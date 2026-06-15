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
    from trading_agents.graph import pipeline
    from trading_agents.state import AgentState

    print_banner()
    console.print(Rule(style="cyan"))
    console.print(
        f"  Analyzing [ticker]{ticker.upper()}[/ticker]",
        style="bold",
    )
    console.print(Rule(style="cyan"))
    console.print()

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
        for i, event in enumerate(pipeline.stream(initial_state)):
            node_name = list(event.keys())[0]

            step_idx = {"fetch_data": 0, "fundamental_analyst": 1, "report_writer": 2}.get(node_name, i)
            if step_idx + 1 < len(steps):
                progress.update(task, description=steps[step_idx + 1][1])
            else:
                progress.update(task, description="✅  Finalizing report...")

            final_state = event[node_name]

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
