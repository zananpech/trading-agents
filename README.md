# Trading Agents 🤖

> **Multi-agent stock fundamental research system** — powered by Google Gemini 3.1 Flash Lite, LangGraph, and Yahoo Finance.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-orange)
![Gemini](https://img.shields.io/badge/Gemini-3.1_Flash_Lite-4285F4?logo=google&logoColor=white)
![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9)
![License](https://img.shields.io/badge/license-MIT-green)

Given a stock ticker, a team of AI agents collaborates to produce a comprehensive, investor-grade equity research report with a clear **BUY / HOLD / SELL** verdict — all from the command line.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 3.1 Flash Lite |
| Agent Orchestration | LangGraph + LangChain |
| Market Data | Yahoo Finance (`yfinance`) — free, no API key |
| Package Manager | `uv` |
| CLI Output | `rich` (colored, formatted terminal output) |
| Language | Python 3.11+ |

---

## Architecture

```
User (CLI: uv run main.py AAPL)
    │
    ▼
LangGraph Pipeline
    │
    ├── 📡 Data Fetcher Node  (yfinance — free)
    │       ├── Company overview & description
    │       ├── Valuation ratios (P/E, Forward P/E, EV/EBITDA, P/B, P/S, PEG)
    │       ├── Financial statements (revenue, margins, debt, FCF)
    │       ├── Earnings history (EPS actual vs. estimate, surprise %)
    │       └── Recent news headlines (last 7 days)
    │
    ├── 🧠 Fundamental Analyst Agent  (Gemini 3.1 Flash Lite)
    │       └── Analyzes 7 dimensions: valuation, health, margins,
    │           growth, cash flow, earnings quality, sentiment
    │
    └── ✍️  Report Writer Agent  (Gemini 3.1 Flash Lite)
            └── Synthesizes analysis into a polished 8-section
                investor-grade report with BUY/HOLD/SELL verdict
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) package manager — install with:
  ```bash
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- A [Google Gemini API key](https://aistudio.google.com/app/apikey) — free tier available

### 2. Clone & install dependencies

```bash
git clone https://github.com/your-username/trading-agents.git
cd trading-agents
uv sync
```

### 3. Configure your API key

```bash
copy .env.example .env
```

Edit `.env` and set your key:
```env
GOOGLE_API_KEY=your-actual-gemini-key-here
```

---

## Usage

```bash
# Analyze a stock (prints report to terminal)
uv run main.py AAPL

# Analyze and save the report as a markdown file
uv run main.py MSFT --save

# Save to a custom directory
uv run main.py TSLA --save --output-dir my_research
```

Reports are saved to `reports/<TICKER>_<DATE>.md` by default.

---

## Report Sections

Each run produces a full **8-section equity research report**:

| # | Section | What it covers |
|---|---|---|
| 1 | **Executive Summary** | 3-4 sentence overview + **BUY / HOLD / SELL** + confidence % |
| 2 | **Company Overview** | Sector, business model, competitive position |
| 3 | **Valuation Analysis** | P/E, EV/EBITDA, P/B, P/S vs. fair value |
| 4 | **Financial Health** | Balance sheet, debt-to-equity, current ratio, cash |
| 5 | **Earnings Quality & Growth** | EPS history, estimate beats/misses, growth rate |
| 6 | **News & Market Sentiment** | Recent news tone — positive, negative, or neutral |
| 7 | **Key Risks** | Top 3–5 investment risks |
| 8 | **Investment Thesis** | Final conviction statement + recommendation |

---

## Project Structure

```
trading-agents/
├── main.py                          # CLI entry point
├── pyproject.toml                   # uv project config & dependencies
├── .env                             # Your API key (gitignored)
├── .env.example                     # API key template
├── reports/                         # Saved reports (auto-created, gitignored)
└── trading_agents/
    ├── config.py                    # Loads env vars, model constants
    ├── state.py                     # LangGraph AgentState TypedDict
    ├── graph.py                     # Pipeline: fetch_data → analyst → writer
    ├── agents/
    │   ├── fundamental_analyst.py   # Gemini: analyzes 7 fundamental dimensions
    │   └── report_writer.py         # Gemini: writes polished investor report
    └── tools/
        └── data_fetcher.py          # 5 yfinance @tool functions
```

---

## Extending the System

The LangGraph pipeline is easy to extend. Some ideas:

- **Technical Analyst agent** — RSI, MACD, Bollinger Bands via `ta` library
- **Risk Manager agent** — enforce position sizing, stop-loss rules
- **Bull vs. Bear debate** — two agents argue before the final verdict
- **Watchlist mode** — analyze multiple tickers in one run
- **SEC EDGAR integration** — parse 10-K / 10-Q filings
- **Web dashboard** — FastAPI + React frontend to browse saved reports

---

## Disclaimer

This system is for **research and educational purposes only**. It does not constitute financial advice. Always do your own due diligence before making investment decisions.
