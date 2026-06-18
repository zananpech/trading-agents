"""
Unit tests for the RAG ingestion and retrieval logic.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from trading_agents.rag.ingestion import extract_ticker_from_filename


def test_extract_ticker_from_filename() -> None:
    assert extract_ticker_from_filename("data/reports/AAPL_10Q.pdf") == "AAPL"
    assert extract_ticker_from_filename("TSLA-10K.html") == "TSLA"
    assert extract_ticker_from_filename("msft_annual_2026.pdf") == "MSFT"
    assert extract_ticker_from_filename("NVDA_10Q_Q3.pdf") == "NVDA"
    assert extract_ticker_from_filename("AMD_file.pdf") == "AMD"
    assert extract_ticker_from_filename("12345_report.pdf") == "12345"
    # Tickers are typically 1-5 alphanumeric characters
    assert extract_ticker_from_filename("VERYLONGTICKER_report.pdf") == "UNKNOWN"


@patch("trading_agents.rag.ingestion.genai.Client")
def test_describe_chart_image(mock_client_class: MagicMock) -> None:
    from trading_agents.rag.ingestion import describe_chart_image

    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_response = MagicMock()
    mock_response.text = "This is a revenue growth chart showing 15% increase."
    mock_client.models.generate_content.return_value = mock_response

    with patch("trading_agents.rag.ingestion.Image.open") as mock_open:
        mock_open.return_value = MagicMock()
        desc = describe_chart_image("dummy_path.png", "dummy_api_key")
        assert "revenue growth chart" in desc
        mock_client.models.generate_content.assert_called_once()
