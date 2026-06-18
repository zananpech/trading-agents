"""
CLI command to run the document ingestion pipeline.
"""
from __future__ import annotations

import argparse
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

from trading_agents.config import RAG_REPORTS_DIR
from trading_agents.rag.ingestion import ingest_directory, ingest_document

console = Console(width=120)


def print_banner() -> None:
    banner = Text()
    banner.append("  [INGEST] RAG INGESTION PIPELINE", style="bold green")
    banner.append("  |  Trading Agents Knowledge Base Ingestor", style="dim white")
    console.print(
        Panel(
            banner,
            border_style="green",
            padding=(0, 2),
        )
    )
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trading-agents-ingest",
        description="Ingests PDF and HTML quarterly reports into local ChromaDB with table & chart parsing.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=RAG_REPORTS_DIR,
        help=f"Directory containing reports to ingest (default: {RAG_REPORTS_DIR})",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a single PDF/HTML file to ingest",
    )

    args = parser.parse_args()

    print_banner()

    try:
        if args.file:
            filepath = args.file
            if not os.path.exists(filepath):
                console.print(f"[bold red]Error: File not found at {filepath}[/bold red]")
                sys.exit(1)

            console.print(f"  [bold cyan]Processing single file:[/bold cyan] [white]{filepath}[/white]")
            with Progress(
                SpinnerColumn(style="green"),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Ingesting document...", total=None)
                chunks_added = ingest_document(filepath)

            if chunks_added > 0:
                console.print(f"  [bold green]Ingested {chunks_added} chunks successfully.[/bold green]\n")
            else:
                console.print(f"  [bold red]Document ingestion failed. Check logs.[/bold red]\n")

        else:
            directory = args.dir
            if not os.path.exists(directory):
                console.print(f"  [yellow]Directory {directory} not found. Creating it...[/yellow]")
                os.makedirs(directory, exist_ok=True)
                console.print(f"  [white]Created reports directory at: {directory}[/white]")
                console.print(f"  [white]Please place PDF/HTML report files (e.g. AAPL_10Q.pdf) inside it and run ingestion again.[/white]\n")
                sys.exit(0)

            console.print(f"  [bold cyan]Scanning directory:[/bold cyan] [white]{directory}[/white]")
            with Progress(
                SpinnerColumn(style="green"),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Ingesting directory files...", total=None)
                chunks_added = ingest_directory(directory)

            if chunks_added > 0:
                console.print(f"  [bold green]Ingestion completed! Added {chunks_added} chunks to ChromaDB.[/bold green]\n")
            else:
                console.print(f"  [yellow]Ingestion completed. 0 chunks added. Place reports in '{directory}' folder.[/yellow]\n")

    except KeyboardInterrupt:
        console.print("\n[dim]Ingestion cancelled.[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
