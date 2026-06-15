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
