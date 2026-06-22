"""
Unit tests for the retry with backoff and jitter decorator.
"""
from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock, patch
from requests.exceptions import RequestException

from trading_agents.tools.retry import retry_with_backoff, is_retryable

def test_is_retryable() -> None:
    # Logic/programming errors should NOT be retryable
    assert not is_retryable(ValueError("invalid value"))
    assert not is_retryable(TypeError("invalid type"))
    assert not is_retryable(KeyError("missing key"))
    
    # Network/API exceptions should be retryable
    assert is_retryable(RequestException("Connection timeout"))
    
    # Generic exceptions containing rate limit keywords should be retryable
    assert is_retryable(Exception("API returned 429 status code"))
    assert is_retryable(Exception("Rate limit exceeded"))
    assert is_retryable(Exception("Resource exhausted (quota limit)"))
    assert is_retryable(Exception("connection refused"))
    assert is_retryable(Exception("503 Service Unavailable"))
    
    # Other generic exceptions without keywords should NOT be retryable by default
    assert not is_retryable(Exception("Some generic unexpected system failure"))

@patch("time.sleep")
def test_retry_success_first_attempt(mock_sleep: MagicMock) -> None:
    call_count = 0
    
    @retry_with_backoff(max_retries=3, initial_backoff=0.1)
    def dummy_func(x):
        nonlocal call_count
        call_count += 1
        return x * 2
        
    res = dummy_func(5)
    assert res == 10
    assert call_count == 1
    mock_sleep.assert_not_called()

@patch("time.sleep")
def test_retry_transient_failure_then_success(mock_sleep: MagicMock) -> None:
    call_count = 0
    
    @retry_with_backoff(max_retries=3, initial_backoff=1.0, max_backoff=5.0, backoff_factor=2.0)
    def dummy_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RequestException("Network timeout")
        return "success"
        
    res = dummy_func()
    assert res == "success"
    assert call_count == 3
    
    # Verify time.sleep was called exactly twice
    assert mock_sleep.call_count == 2
    
    # Verify backoff limits:
    # Attempt 1 backoff limit = 1.0s * (2^0) = 1.0s. Sleep time should be random.uniform(0, 1.0)
    # Attempt 2 backoff limit = 1.0s * (2^1) = 2.0s. Sleep time should be random.uniform(0, 2.0)
    args_list = [call[0][0] for call in mock_sleep.call_args_list]
    assert 0 <= args_list[0] <= 1.0
    assert 0 <= args_list[1] <= 2.0

@patch("time.sleep")
def test_retry_max_retries_exceeded(mock_sleep: MagicMock) -> None:
    call_count = 0
    
    @retry_with_backoff(max_retries=2, initial_backoff=0.1)
    def dummy_func():
        nonlocal call_count
        call_count += 1
        raise Exception("429 Too Many Requests")
        
    with pytest.raises(Exception, match="429 Too Many Requests"):
        dummy_func()
        
    assert call_count == 3 # 1 original + 2 retries
    assert mock_sleep.call_count == 2

@patch("time.sleep")
def test_no_retry_on_logic_error(mock_sleep: MagicMock) -> None:
    call_count = 0
    
    @retry_with_backoff(max_retries=3, initial_backoff=0.1)
    def dummy_func():
        nonlocal call_count
        call_count += 1
        raise ValueError("Invalid format error")
        
    with pytest.raises(ValueError, match="Invalid format error"):
        dummy_func()
        
    assert call_count == 1 # Raises immediately without retrying
    mock_sleep.assert_not_called()
