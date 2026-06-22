"""
Exponential backoff with full jitter retry mechanism for APIs and tools.
"""
from __future__ import annotations

import time
import random
import functools
import logging
from typing import Callable, Any

logger = logging.getLogger("trading_agents.retry")

# Try importing common API/network exceptions
try:
    from google.genai.errors import APIError as GeminiAPIError
except ImportError:
    class GeminiAPIError(Exception):
        pass

try:
    from google.api_core.exceptions import GoogleAPICallError
except ImportError:
    class GoogleAPICallError(Exception):
        pass

try:
    from requests.exceptions import RequestException
except ImportError:
    class RequestException(Exception):
        pass

RETRYABLE_EXCEPTIONS = (RequestException, GeminiAPIError, GoogleAPICallError)

RETRYABLE_KEYWORDS = [
    "429", "rate limit", "quota", "too many requests", "exhausted",
    "500", "502", "503", "504", "timeout", "connection", "econnrefused"
]

def is_retryable(exc: Exception) -> bool:
    """
    Checks if an exception is retryable.
    Returns True if it belongs to RETRYABLE_EXCEPTIONS or is a general exception
    containing retryable keywords in its message/arguments, and is not a logic/programming error.
    """
    # Exclude programming/logic errors explicitly
    if isinstance(exc, (ValueError, TypeError, KeyError, AttributeError, IndexError, NameError, SyntaxError)):
        return False
        
    if isinstance(exc, RETRYABLE_EXCEPTIONS):
        return True
        
    # Check error message keywords for any exception
    msg = str(exc).lower()
    if any(keyword in msg for keyword in RETRYABLE_KEYWORDS):
        return True
        
    return False

def retry_with_backoff(
    max_retries: int = 5,
    initial_backoff: float = 1.0,
    max_backoff: float = 10.0,
    backoff_factor: float = 2.0,
):
    """
    Decorator that applies exponential backoff with full jitter to a function.
    
    Formula:
        backoff = min(max_backoff, initial_backoff * (backoff_factor ** attempt))
        sleep_time = random.uniform(0, backoff)
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not is_retryable(e):
                        # Not a retryable error, raise immediately
                        raise e
                    
                    attempt += 1
                    if attempt > max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries. Raising exception: {e}"
                        )
                        raise e
                    
                    # Calculate exponential backoff limit (attempt - 1 so first retry starts at initial_backoff)
                    backoff = min(max_backoff, initial_backoff * (backoff_factor ** (attempt - 1)))
                    # Apply full jitter (random value between 0 and backoff)
                    sleep_time = random.uniform(0, backoff)
                    
                    print(
                        f"[RETRY] Attempt {attempt}/{max_retries} failed for '{func.__name__}' due to: {e}. "
                        f"Retrying in {sleep_time:.2f} seconds (backoff limit: {backoff:.2f}s)..."
                    )
                    time.sleep(sleep_time)
        return wrapper
    return decorator
