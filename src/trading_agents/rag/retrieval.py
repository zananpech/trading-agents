"""
RAG retrieval module for querying local ChromaDB for a ticker's report context.
"""
from __future__ import annotations

import os
import json
from google import genai
from google.genai import types
from trading_agents.tools.retry import retry_with_backoff
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from trading_agents.config import GOOGLE_API_KEY, CHROMA_DB_PATH

@retry_with_backoff()
def get_rag_context(ticker: str, query: str | None = None, limit: int = 5) -> str:
    """
    Retrieves relevant text context (including parsed tables and chart summaries)
    from ChromaDB for the specified ticker. Uses hybrid search (Vector + BM25)
    fused with Reciprocal Rank Fusion (RRF) and reranked using Gemini.
    """
    if not os.path.exists(CHROMA_DB_PATH):
        return "No recent quarterly report filings found in context (ChromaDB directory does not exist)."

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Load ChromaDB
        db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        
        ticker_upper = ticker.upper()
        
        # 1. Fetch all documents for this ticker to perform BM25 search locally
        db_docs = db.get(where={"ticker": ticker_upper})
        if not db_docs or not db_docs.get("documents"):
            return f"No recent quarterly report filings found in context for ticker: {ticker_upper}."
            
        all_docs = []
        for text_content, metadata in zip(db_docs["documents"], db_docs["metadatas"]):
            all_docs.append(Document(page_content=text_content, metadata=metadata))
            
        # 2. Setup BM25 Retriever
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 10
        
        if query is None:
            query = f"financial results, earnings, balance sheet, income statement, valuation, or guidance for {ticker_upper}"
        
        # Retrieve candidates
        bm25_results = bm25_retriever.invoke(query)
        vector_results = db.similarity_search(
            query=query,
            k=10,
            filter={"ticker": ticker_upper}
        )
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        def add_rrf_scores(results_list):
            for rank, doc in enumerate(results_list, 1):
                source = doc.metadata.get("source", "unknown")
                chunk_idx = doc.metadata.get("chunk_index", -1)
                key = (doc.page_content, source, chunk_idx)
                if key not in rrf_scores:
                    rrf_scores[key] = {"doc": doc, "score": 0.0}
                rrf_scores[key]["score"] += 1.0 / (60.0 + rank)
                
        add_rrf_scores(vector_results)
        add_rrf_scores(bm25_results)
        
        fused = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        top_fused = [item["doc"] for item in fused[:10]]
        
        # 4. Gemini-based Reranking
        final_docs = top_fused
        if len(top_fused) > 1:
            try:
                client = genai.Client(api_key=GOOGLE_API_KEY)
                
                candidates_str = ""
                for idx, doc in enumerate(top_fused):
                    candidates_str += f"--- Candidate [{idx}] ---\n{doc.page_content}\n\n"
                
                prompt = (
                    f"You are a financial analyst assistant tasked with ranking candidate document chunks by relevance.\n"
                    f"Given the search query, examine each candidate chunk and determine how relevant it is to answering the query. "
                    f"Order the candidate indexes from most relevant to least relevant.\n\n"
                    f"Search Query: {query}\n\n"
                    f"{candidates_str}"
                    f"Return a JSON object containing a key 'ranked_indices' which is a list of integers corresponding to the indices of the most relevant candidate chunks (from most to least relevant). "
                    f"Do not output anything else."
                )
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "ranked_indices": types.Schema(
                                    type=types.Type.ARRAY,
                                    items=types.Schema(type=types.Type.INTEGER)
                                )
                            },
                            required=["ranked_indices"]
                        ),
                        temperature=0.0
                    )
                )
                
                res_data = json.loads(response.text)
                ranked_indices = res_data.get("ranked_indices", [])
                
                seen = set()
                reranked_docs = []
                for idx in ranked_indices:
                    if 0 <= idx < len(top_fused) and idx not in seen:
                        reranked_docs.append(top_fused[idx])
                        seen.add(idx)
                        
                for idx, doc in enumerate(top_fused):
                    if idx not in seen:
                        reranked_docs.append(doc)
                        seen.add(idx)
                        
                final_docs = reranked_docs
            except Exception as rerank_err:
                print(f"Warning: Gemini reranking failed: {rerank_err}. Using RRF fusion fallback.")
                final_docs = top_fused
                
        # Limit to the requested size
        results = final_docs[:limit]
        
        # Consolidate retrieved contexts
        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown file")
            chunk_content = doc.page_content.strip()
            
            # Format header path from metadata keys (Header 1, Header 2, Header 3)
            headers = []
            for h_key in ["Header 1", "Header 2", "Header 3"]:
                h_val = doc.metadata.get(h_key)
                if h_val:
                    headers.append(h_val)
            header_path = " > ".join(headers)
            
            chunk_idx = doc.metadata.get("chunk_index")
            tot_chunks = doc.metadata.get("total_chunks")
            
            meta_str = ""
            if chunk_idx is not None and tot_chunks is not None:
                meta_str += f" [Chunk {chunk_idx + 1}/{tot_chunks}]"
            if header_path:
                meta_str += f" [Section: {header_path}]"
                
            # Construct a clean section for the agent
            context_parts.append(
                f"--- Chunk {i} from {source}{meta_str} ---\n"
                f"{chunk_content}"
            )
            
        return "\n\n".join(context_parts)
        
    except Exception as e:
        print(f"Error querying ChromaDB for {ticker}: {e}")
        return f"Error retrieving RAG context for ticker: {ticker}. Details: {e}"


@retry_with_backoff()
def generate_rag_answer(ticker: str, query: str) -> str:
    """
    Answers a specific user query about a ticker by retrieving context
    from local ChromaDB and using Gemini for precise, strictly-grounded answering.
    """
    context = get_rag_context(ticker, query=query, limit=5)
    
    # If retrieval says no filings found or has error, we immediately return that message
    if "No recent quarterly report filings found in context" in context or "Error retrieving RAG context" in context:
        return context
        
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        system_prompt = (
            "You are a helpful and precise financial analyst assistant.\n"
            "Your job is to answer the user's query using ONLY the provided company report context chunks.\n"
            "Do not make up facts, do not extrapolate, and do not use external or training dataset information.\n"
            "If the answer to the query cannot be found within the provided context chunks, state exactly:\n"
            "'I cannot answer this query based on the provided report context.'\n"
            "Be quantitative and reference the source files and sections from the chunk headers in your answer."
        )
        
        user_content = (
            f"Retrieved Report Context:\n"
            f"=========================\n"
            f"{context}\n"
            f"=========================\n\n"
            f"User Query: {query}\n"
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0
            )
        )
        
        return response.text.strip()
    except Exception as e:
        print(f"Error generating RAG answer: {e}")
        return f"Error generating answer for query: {query}. Details: {e}"

