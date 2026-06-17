"""
Report Writer Agent

Takes the Fundamental Analyst's output and synthesizes it into a
polished, investor-grade research report using Google Gemini 3.1 flash Pro.
This is a LangGraph node function.
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

SYSTEM_PROMPT = """You are a professional equity research report writer at a prestigious investment bank.
Your job is to transform raw fundamental analysis into a polished, investor-grade research report.

The report MUST follow this exact structure:

---
# [COMPANY NAME] ([TICKER]) — Equity Research Report
**Date:** [Today's Date]
**Analyst:** AI Fundamental Research System

---

## Executive Summary
A concise 3-4 sentence summary of the investment case.
Then on a new line, clearly state:
**VERDICT: [BUY / HOLD / SELL]** | **Confidence: [X]%** | **Investment Horizon: [Short/Medium/Long-term]**

---

## Company Overview
Brief description of what the company does, its sector, and competitive position.

---

## Valuation Analysis
Detailed analysis of whether the stock is over/under/fairly valued.
Include a comparison table of key multiples if possible.

---

## Financial Health
Analysis of balance sheet strength, debt levels, and liquidity.

---

## Earnings Quality & Growth
Analysis of revenue/EPS trends, consistency of beating estimates.

---

## News & Market Sentiment
Summary of recent news themes and market sentiment.

---

## Key Risks
A bulleted list of the 3-5 most important risks to the investment thesis.

---

## Investment Thesis
Final paragraph: why an investor should (or should not) own this stock.
Restate the BUY/HOLD/SELL recommendation with conviction.

---

Use markdown formatting throughout. Be specific with numbers.
Do NOT hedge excessively — take a clear stance backed by data.
"""


def report_writer_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Report Writer Agent.
    Synthesizes fundamental analysis into a final investor report.
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
    )

    ticker = state["ticker"]
    analysis = state["fundamental_analysis"]

    user_prompt = f"""Using the following fundamental analysis for {ticker}, write a complete,
polished equity research report following the exact structure specified.

--- FUNDAMENTAL ANALYSIS ---
{analysis}

Remember to:
1. Include specific numbers from the analysis
2. Give a clear BUY/HOLD/SELL verdict with confidence percentage
3. List concrete risks
4. Make it read like a professional Wall Street research report
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    report = _extract_text(response.content)

    return {**state, "final_report": report}
