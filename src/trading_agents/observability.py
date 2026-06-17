"""
Langfuse Observability Layer (v4 API)

In Langfuse v4 the SDK reads credentials from env vars automatically:
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

The CallbackHandler auto-creates a trace per run. We just need to:
  1. Create a fresh CallbackHandler per pipeline run (one trace per run)
  2. Call flush() after the pipeline so spans are exported before process exits

Gracefully disabled if LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set.
"""
from __future__ import annotations

import warnings
from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler


def get_callback_handler(ticker: str) -> BaseCallbackHandler | None:
    """
    Create and return a fresh Langfuse CallbackHandler for one pipeline run.

    Langfuse v4 reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
    from the environment automatically — no explicit key passing needed.

    A fresh handler instance per run = one separate trace in the dashboard.

    Returns None if Langfuse is not configured or not installed.
    """
    from trading_agents.config import LANGFUSE_ENABLED

    if not LANGFUSE_ENABLED:
        return None

    try:
        from langfuse.langchain import CallbackHandler

        # Fresh handler = fresh trace in Langfuse dashboard
        # Credentials are read from LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY /
        # LANGFUSE_HOST env vars automatically by the SDK's get_client()
        handler = CallbackHandler()
        return handler

    except ImportError:
        return None
    except Exception as exc:
        warnings.warn(
            f"Langfuse handler could not be created for {ticker}: {exc}\n"
            "Analysis will continue without observability.",
            stacklevel=2,
        )
        return None


def get_trace_id(handler: BaseCallbackHandler | None) -> str | None:
    """
    Extract the trace ID from a completed Langfuse CallbackHandler.

    The trace ID is needed to attach evaluation scores to the correct trace
    after the pipeline finishes. Returns None if the handler is None or if
    the trace ID cannot be retrieved.
    """
    if handler is None:
        return None
    try:
        # Langfuse v4 CallbackHandler exposes the trace_id on the instance
        # after at least one LangChain call has been processed.
        trace_id = getattr(handler, "trace_id", None)
        if trace_id:
            return str(trace_id)
        # Fallback: some versions nest it under _trace or _root_span
        for attr in ("_trace", "_root_span", "root_span"):
            obj = getattr(handler, attr, None)
            if obj is not None:
                tid = getattr(obj, "trace_id", None) or getattr(obj, "id", None)
                if tid:
                    return str(tid)
        return None
    except Exception:
        return None


def flush() -> None:
    """
    Flush all pending Langfuse spans to the server.

    MUST be called after pipeline.stream() completes so traces are exported
    before the Python process exits. The SDK batches spans and sends them
    asynchronously — without flush() they may be lost on exit.
    """
    try:
        from langfuse import get_client
        client = get_client()
        client.flush()
    except Exception:
        pass
