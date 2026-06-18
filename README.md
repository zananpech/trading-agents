# Trading Agents 🤖

> **Multi-agent stock fundamental research system** — powered by Google Gemini 3.1 Flash Lite, LangGraph, and Yahoo Finance.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-orange)
![Gemini](https://img.shields.io/badge/Gemini-3.1_Flash_Lite-4285F4?logo=google&logoColor=white)
![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9)
![Langfuse](https://img.shields.io/badge/Langfuse-observability-8B5CF6)
![License](https://img.shields.io/badge/license-MIT-green)

Given a stock ticker, a team of AI agents collaborates to produce a comprehensive, investor-grade equity research report with a clear **BUY / HOLD / SELL** verdict — all from the command line.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 3.1 Flash Lite |
| Agent Orchestration | LangGraph + LangChain |
| Market Data | Yahoo Finance (`yfinance`) — free, no API key |
| Knowledge Base / RAG | ChromaDB + `pymupdf4llm` (table-preserving Markdown) |
| Multimodal Chart Parsing | Google Gemini 2.5 Flash Vision |
| Observability | Langfuse (self-hosted via Docker Compose) |
| Package Manager | `uv` |
| CLI Output | `rich` (colored, formatted terminal output) |
| Language | Python 3.11+ |

---

## Architecture

```
User (CLI: uv run trading-agents AAPL)
    │
    ▼
LangGraph Pipeline
    │
    ├── 📡 Data Fetcher Node  (yfinance — free + Local RAG)
    │       ├── Company overview & description
    │       ├── Valuation ratios (P/E, Forward P/E, EV/EBITDA, P/B, P/S, PEG)
    │       ├── Financial statements (revenue, margins, debt, FCF)
    │       ├── Earnings history (EPS actual vs. estimate, surprise %)
    │       ├── Recent news headlines (last 7 days)
    │       └── 📥 RAG context from quarterly/annual reports (local ChromaDB)
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
  ```powershell
  # PowerShell (Windows)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- A [Google Gemini API key](https://aistudio.google.com/app/apikey) — free tier available
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — only required for Langfuse observability (optional)

### 2. Clone & install dependencies

```bash
git clone https://github.com/zananpech/trading-agents.git
cd trading-agents
uv sync
```

### 3. Configure your API key

```bash
# Git Bash / macOS / Linux
cp .env.example .env

# Windows CMD / PowerShell
copy .env.example .env
```

> **If `.env` already exists**, just open it and add your key — do not overwrite it.

Edit `.env` and set your key:
```env
GOOGLE_API_KEY=your-actual-gemini-key-here
```

---

## Usage

```bash
# Analyze a stock (prints report to terminal)
uv run trading-agents AAPL

# Analyze and save the report as a markdown file
uv run trading-agents MSFT --save

# Save to a custom directory
uv run trading-agents TSLA --save --output-dir my_research
```

Reports are saved to `reports/<TICKER>_<DATE>.md` by default.

---

## Knowledge Base & RAG Ingestion

You can drop PDF or HTML quarterly/annual reports (e.g. 10-Q, 10-K) into the local `data/reports/` directory, ingest them into a local vector database, and have the agents automatically use this context when analyzing a stock.

This pipeline:
* Parses PDF reports into high-quality Markdown, keeping financial tables intact (via `pymupdf4llm`).
* Automatically extracts charts, plots, and visual graphics to `data/reports/images/`.
* Transcribes visual data into detailed text descriptions using `gemini-2.5-flash` during ingestion.
* Chunks and stores context into a local ChromaDB instance (`.chroma_db/`).

### Ingestion Usage

1. Place PDF or HTML reports in `data/reports/`. Name files starting with the ticker symbol (e.g., `AAPL_10Q.pdf`, `TSLA_2025_10K.html`).
2. Run the ingestion command:
   ```bash
   uv run trading-agents-ingest
   ```
3. Run the analysis as usual:
   ```bash
   uv run trading-agents AAPL
   ```
   The retrieved context (including parsed financial tables and transcribed charts) will automatically be injected into the fundamental analyst agent's input context.

To ingest a single report file:
```bash
uv run trading-agents-ingest --file data/reports/AAPL_10Q.pdf
```

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
├── src/
│   └── trading_agents/
│       ├── __init__.py
│       ├── cli/
│       │   ├── main.py              # CLI entry point (trading-agents)
│       │   ├── eval_cli.py          # Eval CLI (trading-agents-eval)
│       │   └── ingest_cli.py        # Ingestion CLI (trading-agents-ingest)
│       ├── agents/                  # LangGraph node functions
│       ├── tools/                   # yfinance data fetching tools
│       ├── rag/                     # RAG ingestion & retrieval logic
│       │   ├── __init__.py
│       │   ├── ingestion.py         # Table and chart/image parser
│       │   └── retrieval.py         # ChromaDB query filtering
│       ├── evaluation/              # Rule-based & LLM judges
│       ├── config.py                # Environment variables & constants
│       ├── graph.py                 # LangGraph pipeline definition
│       ├── observability.py         # Langfuse integration
│       └── state.py                 # Shared AgentState TypedDict
├── tests/                           # Pytest test suite
├── docs/                            # Documentation
├── pyproject.toml                   # uv project config & dependencies
├── uv.lock                          # Locked dependency versions
├── docker-compose.yml               # Self-hosted Langfuse stack
├── .env                             # Your API keys (gitignored)
└── reports/                         # Saved reports (auto-created)
```

---

## Observability with Langfuse

The system ships with full [Langfuse](https://langfuse.com) observability — trace every LLM call, measure latency, count tokens, and score report quality.

### 1. Start the self-hosted Langfuse stack

**Step 1** — Copy the secrets template:
```bash
# Git Bash / macOS / Linux
cp docker-compose.env.example docker-compose.env

# Windows CMD / PowerShell
copy docker-compose.env.example docker-compose.env
```

**Step 2** — Edit `docker-compose.env` and fill in your passwords. Three keys **must be 64-character hex strings** — generate them with:
```bash
uv run python -c "import secrets; print(secrets.token_hex(32))"
```
Run that command three times and paste each result into `NEXTAUTH_SECRET`, `SALT`, and `ENCRYPTION_KEY`.

**Step 3** — Start all services:
```bash
docker compose up -d
```

Open **http://localhost:3000** — the Langfuse UI will load.

### 2. Create a project and get API keys

1. Edit `docker-compose.env` and set `LANGFUSE_INIT_USER_PASSWORD` to your desired admin password
2. Open **http://localhost:3000** and log in with `admin@local.dev` and the password you set
3. Create a new project → **Settings** → **API Keys** → **Create new key**
4. Copy the **public key** and **secret key**

### 3. Add keys to your `.env`

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

### 4. Run the pipeline — traces appear automatically

```bash
uv run trading-agents AAPL
```

The CLI will show: `🔭 Langfuse tracing active → http://localhost:3000`

### What you see in the dashboard

| Trace level | What's captured |
|---|---|
| **Root trace** | Ticker, model, run date, pipeline name |
| **fetch_data span** | yfinance tool calls and raw outputs |
| **fundamental_analyst span** | Gemini prompt + analysis output + token usage |
| **report_writer span** | Gemini prompt + final report + latency |

You can also add **manual scores** (e.g., report quality 1–5) on any trace directly in the UI.

> **Note:** Langfuse is fully optional — if the keys are not set, the pipeline runs exactly as before with zero overhead.

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
