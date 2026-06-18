"""
Fundamental Analyst Agent

Receives raw market data and produces a structured fundamental analysis
using Google Gemini 3.1 flash Pro. This is a LangGraph node function.
"""
from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from trading_agents.config import GEMINI_MODEL, GOOGLE_API_KEY, LLM_TEMPERATURE
from trading_agents.state import AgentState


def _extract_text(content) -> str:
    """
    Safely extract a plain string from an LLM response content.
    Newer versions of langchain-google-genai may return a list of
    content blocks instead of a bare string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)

SYSTEM_PROMPT = """You are a senior equity research analyst at a top-tier investment bank.
Your job is to produce rigorous, data-driven fundamental analysis of publicly traded stocks.

You will receive structured financial data for a company and must analyze:

1. **Valuation** — Is the stock cheap, fair, or expensive vs. peers and history?
   Examine P/E, Forward P/E, P/B, P/S, EV/EBITDA, EV/Revenue, PEG ratio.

2. **Financial Health** — Is the balance sheet strong?
   Examine total debt, debt-to-equity, current ratio, cash position.

3. **Profitability & Margins** — Is the business profitable and improving?
   Examine gross margin, operating margin, net margin, EBITDA.

4. **Growth** — Is revenue and earnings growing?
   Examine revenue growth YoY, EPS growth, and quarterly earnings trends.

5. **Cash Flow Quality** — Is the business generating real cash?
   Examine free cash flow, operating cash flow.

6. **Earnings Quality** — Does the company consistently beat estimates?
   Look at EPS surprise % across recent quarters.

7. **News Sentiment** — What is the market narrative right now?
   Summarize recent news tone: positive, negative, or neutral.

Be specific and quantitative. Reference actual numbers. Note any red flags.
Structure your output as a detailed analysis report using markdown headers.
End your analysis with a clear BULLISH / NEUTRAL / BEARISH stance and your reasoning.
"""


def fundamental_analyst_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Fundamental Analyst Agent.
    Reads raw data from state and writes analysis back to state.
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
    )

    ticker = state["ticker"]
    raw_data = state["raw_data"]

    user_prompt = f"""Analyze {ticker} using the following financial data:

--- COMPANY OVERVIEW ---
{raw_data.get('stock_info', 'Not available')}

--- VALUATION METRICS ---
{raw_data.get('valuation_metrics', 'Not available')}

--- FINANCIAL STATEMENTS ---
{raw_data.get('financials', 'Not available')}

--- EARNINGS HISTORY ---
{raw_data.get('earnings', 'Not available')}

--- RECENT NEWS ---
{raw_data.get('news', 'Not available')}

--- QUARTERLY REPORTS & RAG CONTEXT ---
{raw_data.get('rag_context', 'No recent quarterly report filings found in context.')}

Provide a comprehensive fundamental analysis covering all 7 dimensions listed in your instructions.
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    analysis = _extract_text(response.content)

    return {**state, "fundamental_analysis": analysis}
