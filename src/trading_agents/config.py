"""
Configuration and constants for the Trading Agents system.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Settings ────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL: str = "gemini-3.1-flash-lite"
LLM_TEMPERATURE: float = 0.2  # Low temperature for analytical consistency

# ── Langfuse Observability (optional) ────────────────────────────────────────
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_ENABLED: bool = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

# ── Evaluation Settings ───────────────────────────────────────────────────────
# Toggle to False to disable Langfuse score uploads while keeping local eval
LANGFUSE_EVAL_ENABLED: bool = LANGFUSE_ENABLED
# Name of the Langfuse Dataset used for regression testing
EVAL_DATASET_NAME: str = os.getenv("EVAL_DATASET_NAME", "trading-agents-eval")

# ── Report Settings ──────────────────────────────────────────────────────────
REPORTS_DIR: str = "reports"

# ── Data Settings ────────────────────────────────────────────────────────────
NEWS_LOOKBACK_DAYS: int = 7
PRICE_HISTORY_PERIOD: str = "1y"  # yfinance period string

# ── Validation ───────────────────────────────────────────────────────────────
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY is not set.\n"
        "Copy .env.example to .env and add your key from:\n"
        "https://aistudio.google.com/app/apikey"
    )
