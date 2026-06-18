"""
RAG retrieval module for querying local ChromaDB for a ticker's report context.
"""
from __future__ import annotations

import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from trading_agents.config import GOOGLE_API_KEY, CHROMA_DB_PATH

def get_rag_context(ticker: str, limit: int = 5) -> str:
    """
    Retrieves relevant text context (including parsed tables and chart summaries)
    from ChromaDB for the specified ticker.
    """
    if not os.path.exists(CHROMA_DB_PATH):
        return "No recent quarterly report filings found in context (ChromaDB directory does not exist)."

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Load ChromaDB
        db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        
        # Query matching ticker in metadata
        ticker_upper = ticker.upper()
        
        # Retrieve context
        # In newer langchain, search_kwargs has 'filter'
        results = db.similarity_search(
            query=f"financial results, earnings, balance sheet, income statement, valuation, or guidance for {ticker_upper}",
            k=limit,
            filter={"ticker": ticker_upper}
        )
        
        if not results:
            return f"No recent quarterly report filings found in context for ticker: {ticker_upper}."
            
        # Consolidate retrieved contexts
        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown file")
            chunk_content = doc.page_content.strip()
            # Construct a clean section for the agent
            context_parts.append(
                f"--- Chunk {i} from {source} ---\n"
                f"{chunk_content}"
            )
            
        return "\n\n".join(context_parts)
        
    except Exception as e:
        print(f"Error querying ChromaDB for {ticker}: {e}")
        return f"Error retrieving RAG context for ticker: {ticker}. Details: {e}"
