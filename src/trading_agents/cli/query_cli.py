#!/usr/bin/env python
"""
CLI command to query RAG filings context directly and generate a strict answer.
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

from trading_agents.rag.retrieval import generate_rag_answer, get_rag_context

console = Console(width=120)


def print_banner() -> None:
    banner = Text()
    banner.append("  [QUERY] RAG Q&A GENERATION", style="bold yellow")
    banner.append("  |  Trading Agents Direct Document Query", style="dim white")
    console.print(
        Panel(
            banner,
            border_style="yellow",
            padding=(0, 2),
        )
    )
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trading-agents-query",
        description="Directly queries quarterly/annual filings for a ticker and generates a grounded response.",
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol to query (e.g., AAPL)",
    )
    parser.add_argument(
        "query",
        type=str,
        help="Query string / question to ask the RAG system",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        default=False,
        help="Show the retrieved report chunks used as context",
    )

    args = parser.parse_args()
    ticker = args.ticker.strip().upper()
    query = args.query.strip()

    if not ticker.isalpha():
        console.print(f"[bold red]Error: Invalid ticker '{ticker}'. Ticker must contain only letters.[/bold red]")
        sys.exit(1)

    print_banner()
    
    console.print(f"  [bold cyan]Ticker:[/bold cyan] [white]{ticker}[/white]")
    console.print(f"  [bold cyan]Query:[/bold cyan] [white]{query}[/white]")
    console.print()

    try:
        with Progress(
            SpinnerColumn(style="yellow"),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Retrieving reports and generating answer...", total=None)
            answer = generate_rag_answer(ticker, query)

        console.print(
            Panel(
                answer,
                title=f"🤖 Gemini Answer for {ticker}",
                title_align="left",
                border_style="green",
                padding=(1, 2),
            )
        )
        console.print()

        if args.show_sources:
            # Fetch context to display
            context = get_rag_context(ticker, query=query, limit=5)
            console.print(
                Panel(
                    context,
                    title="📚 Retrieved Report Sources",
                    title_align="left",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
            console.print()

    except Exception as e:
        console.print(f"[bold red]Execution error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
