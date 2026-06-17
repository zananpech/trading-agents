# Trading Agents — Documentation

> Architecture and usage reference for the `trading-agents` project.

## Architecture Overview

```
src/trading_agents/
├── cli/           # Entry points: main.py (analyze) and eval_cli.py (evaluate)
├── agents/        # LangGraph node functions (fundamental_analyst, report_writer)
├── tools/         # yfinance data-fetching LangChain tools
├── evaluation/    # Rule-based and LLM-as-a-judge evaluation framework
├── config.py      # Environment variables and constants
├── graph.py       # LangGraph pipeline definition
├── observability.py # Langfuse tracing integration
└── state.py       # Shared AgentState TypedDict
```

## Quick Start

```bash
# Analyze a stock
uv run trading-agents AAPL

# Save report to file
uv run trading-agents MSFT --save

# Run regression evaluation
uv run trading-agents-eval
```
