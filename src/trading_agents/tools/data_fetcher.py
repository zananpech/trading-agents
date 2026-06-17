"""
Data fetching tools using yfinance.

All functions return clean, human-readable strings so the LLM agents
can consume them directly without additional parsing.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf
from langchain_core.tools import tool


def _safe(val: Any, fmt: str = "") -> str:
    """Return a formatted value, or 'N/A' if None/NaN."""
    if val is None:
        return "N/A"
    try:
        if fmt == "$":
            return f"${val:,.2f}"
        if fmt == "%":
            return f"{val:.2f}%"
        if fmt == "B":
            return f"${val / 1e9:.2f}B"
        if fmt == "M":
            return f"${val / 1e6:.2f}M"
        return str(val)
    except (TypeError, ValueError):
        return "N/A"


@tool
def get_stock_info(ticker: str) -> str:
    """
    Fetch company overview: name, sector, industry, market cap,
    description, employees, website, country, currency.
    """
    t = yf.Ticker(ticker.upper())
    info = t.info

    return json.dumps(
        {
            "ticker": ticker.upper(),
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "country": info.get("country", "N/A"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "employees": info.get("fullTimeEmployees", "N/A"),
            "website": info.get("website", "N/A"),
            "market_cap": _safe(info.get("marketCap"), "B"),
            "enterprise_value": _safe(info.get("enterpriseValue"), "B"),
            "description": (info.get("longBusinessSummary") or "N/A")[:800],
        },
        indent=2,
    )


@tool
def get_valuation_metrics(ticker: str) -> str:
    """
    Fetch key valuation ratios: P/E, Forward P/E, P/B, P/S,
    EV/EBITDA, EV/Revenue, PEG ratio.
    """
    t = yf.Ticker(ticker.upper())
    info = t.info

    return json.dumps(
        {
            "ticker": ticker.upper(),
            "pe_ratio_ttm": _safe(info.get("trailingPE")),
            "pe_ratio_forward": _safe(info.get("forwardPE")),
            "price_to_book": _safe(info.get("priceToBook")),
            "price_to_sales_ttm": _safe(info.get("priceToSalesTrailing12Months")),
            "ev_to_ebitda": _safe(info.get("enterpriseToEbitda")),
            "ev_to_revenue": _safe(info.get("enterpriseToRevenue")),
            "peg_ratio": _safe(info.get("trailingPegRatio")),
            "beta": _safe(info.get("beta")),
            "52_week_high": _safe(info.get("fiftyTwoWeekHigh"), "$"),
            "52_week_low": _safe(info.get("fiftyTwoWeekLow"), "$"),
            "current_price": _safe(info.get("currentPrice") or info.get("regularMarketPrice"), "$"),
            "analyst_target_price": _safe(info.get("targetMeanPrice"), "$"),
            "analyst_recommendation": info.get("recommendationKey", "N/A"),
            "number_of_analysts": _safe(info.get("numberOfAnalystOpinions")),
        },
        indent=2,
    )


@tool
def get_financials(ticker: str) -> str:
    """
    Fetch income statement & balance sheet highlights:
    revenue, gross profit, operating income, net income,
    EBITDA, total debt, cash, free cash flow, current ratio.
    """
    t = yf.Ticker(ticker.upper())
    info = t.info

    return json.dumps(
        {
            "ticker": ticker.upper(),
            # Income Statement
            "revenue_ttm": _safe(info.get("totalRevenue"), "B"),
            "revenue_growth_yoy": _safe(info.get("revenueGrowth"), "%") if info.get("revenueGrowth") else "N/A",
            "gross_profit_ttm": _safe(info.get("grossProfits"), "B"),
            "gross_margin": _safe(info.get("grossMargins", 0) * 100, "%") if info.get("grossMargins") else "N/A",
            "operating_income_ttm": _safe(info.get("operatingCashflow"), "B"),
            "operating_margin": _safe(info.get("operatingMargins", 0) * 100, "%") if info.get("operatingMargins") else "N/A",
            "net_income_ttm": _safe(info.get("netIncomeToCommon"), "B"),
            "profit_margin": _safe(info.get("profitMargins", 0) * 100, "%") if info.get("profitMargins") else "N/A",
            "ebitda": _safe(info.get("ebitda"), "B"),
            # Balance Sheet
            "total_cash": _safe(info.get("totalCash"), "B"),
            "total_debt": _safe(info.get("totalDebt"), "B"),
            "debt_to_equity": _safe(info.get("debtToEquity")),
            "current_ratio": _safe(info.get("currentRatio")),
            "book_value_per_share": _safe(info.get("bookValue"), "$"),
            # Cash Flow
            "free_cash_flow": _safe(info.get("freeCashflow"), "B"),
            "operating_cash_flow": _safe(info.get("operatingCashflow"), "B"),
        },
        indent=2,
    )


@tool
def get_earnings(ticker: str) -> str:
    """
    Fetch quarterly earnings history: EPS (actual vs estimated),
    surprise percentage, and revenue beats/misses.
    """
    t = yf.Ticker(ticker.upper())

    try:
        hist = t.earnings_dates
        if hist is None or hist.empty:
            return json.dumps({"ticker": ticker.upper(), "earnings": [], "note": "No earnings data available"})

        # Take last 8 quarters
        hist = hist.head(8).reset_index()
        earnings = []
        for _, row in hist.iterrows():
            earnings.append(
                {
                    "date": str(row.get("Earnings Date", "N/A"))[:10],
                    "eps_estimate": _safe(row.get("EPS Estimate")),
                    "eps_actual": _safe(row.get("Reported EPS")),
                    "surprise_pct": _safe(row.get("Surprise(%)"), "%") if row.get("Surprise(%)") else "N/A",
                }
            )

        info = t.info
        return json.dumps(
            {
                "ticker": ticker.upper(),
                "eps_ttm": _safe(info.get("trailingEps"), "$"),
                "eps_forward": _safe(info.get("forwardEps"), "$"),
                "eps_growth_qoq": _safe(info.get("earningsQuarterlyGrowth", 0) * 100, "%") if info.get("earningsQuarterlyGrowth") else "N/A",
                "eps_growth_annual": _safe(info.get("earningsGrowth", 0) * 100, "%") if info.get("earningsGrowth") else "N/A",
                "quarterly_earnings": earnings,
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"ticker": ticker.upper(), "error": str(e)})


@tool
def get_news(ticker: str) -> str:
    """
    Fetch recent news headlines for the ticker (last 7 days).
    Returns titles, publishers, and links.
    """
    t = yf.Ticker(ticker.upper())
    try:
        news_items = t.news or []
        cutoff = datetime.now() - timedelta(days=7)
        recent = []
        for item in news_items[:15]:
            pub_time = datetime.fromtimestamp(item.get("providerPublishTime", 0))
            if pub_time >= cutoff:
                recent.append(
                    {
                        "title": item.get("title", "N/A"),
                        "publisher": item.get("publisher", "N/A"),
                        "published": pub_time.strftime("%Y-%m-%d %H:%M"),
                        "link": item.get("link", ""),
                    }
                )

        return json.dumps(
            {
                "ticker": ticker.upper(),
                "articles_found": len(recent),
                "news": recent,
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"ticker": ticker.upper(), "error": str(e)})
