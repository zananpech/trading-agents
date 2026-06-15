"""
Shared state schema for the LangGraph pipeline.

Every node reads from and writes to this TypedDict.
"""
from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    The shared state that flows through every node in the LangGraph pipeline.

    Fields
    ------
    ticker : str
        The stock ticker being analyzed (e.g. "AAPL").
    raw_data : dict[str, str]
        Raw JSON strings returned by the yfinance data tools.
        Keys: "stock_info", "valuation_metrics", "financials", "earnings", "news".
    fundamental_analysis : str
        Markdown analysis produced by the Fundamental Analyst node.
    final_report : str
        Polished markdown report produced by the Report Writer node.
    """

    ticker: str
    raw_data: dict[str, Any]
    fundamental_analysis: str
    final_report: str
