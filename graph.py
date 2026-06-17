"""
LangGraph pipeline definition.

Graph topology:
    fetch_data → fundamental_analyst → report_writer → END
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents import fundamental_analyst_node, report_writer_node
from state import AgentState
from tools import (
    get_earnings,
    get_financials,
    get_news,
    get_stock_info,
    get_valuation_metrics,
)


def fetch_data_node(state: AgentState) -> AgentState:
    """
    Data Fetcher node: calls all yfinance tools and stores results in state.
    Runs synchronously — all 5 tools are called in sequence.
    """
    ticker = state["ticker"]

    raw_data: dict[str, str] = {
        "stock_info": get_stock_info.invoke({"ticker": ticker}),
        "valuation_metrics": get_valuation_metrics.invoke({"ticker": ticker}),
        "financials": get_financials.invoke({"ticker": ticker}),
        "earnings": get_earnings.invoke({"ticker": ticker}),
        "news": get_news.invoke({"ticker": ticker}),
    }

    return {**state, "raw_data": raw_data}


def build_graph() -> CompiledStateGraph:
    """
    Construct and compile the multi-agent LangGraph pipeline.

    Returns a compiled runnable graph.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("fetch_data", fetch_data_node)
    graph.add_node("fundamental_analyst", fundamental_analyst_node)
    graph.add_node("report_writer", report_writer_node)

    # ── Define edges (linear pipeline) ───────────────────────────────────────
    graph.set_entry_point("fetch_data")
    graph.add_edge("fetch_data", "fundamental_analyst")
    graph.add_edge("fundamental_analyst", "report_writer")
    graph.add_edge("report_writer", END)

    return graph.compile()


# Singleton compiled graph — import and call .invoke() on this
pipeline = build_graph()
