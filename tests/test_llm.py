"""Tests for Google Gemini LLM API integration."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from google.genai import errors, types
from pydantic import BaseModel

from qs.config import DEFAULT_GEMINI_MODEL, FALLBACK_GEMINI_MODEL
from qs.errors.exceptions import (
    LLMAuthenticationError,
    LLMInferenceError,
    LLMParsingError,
    LLMRateLimitError,
    MissingAPIKeyError,
)
from qs.llm.google import (
    GeminiService,
    check_api_status,
    close_client,
    get_genai_client,
    send_request,
    validate_request,
    validate_response,
)
from qs.llm.google import (
    logger as llm_logger,
)


class SampleAnswer(BaseModel):
    choice: str
    confidence: float


@pytest.fixture
def mock_client():
    client = MagicMock()
    return client


@pytest.fixture
def log_capture():
    records = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = CaptureHandler()
    handler.setLevel(logging.DEBUG)
    llm_logger.addHandler(handler)
    original_level = llm_logger.level
    llm_logger.setLevel(logging.DEBUG)
    yield records
    llm_logger.removeHandler(handler)
    llm_logger.setLevel(original_level)


# --- Client & Status Tests ---


def test_get_genai_client(monkeypatch):
    valid_key = "AIzaSy" + "K" * 33
    monkeypatch.setenv("GEMINI_API_KEY", valid_key)
    client = get_genai_client()
    assert client is not None


def test_close_client():
    client = MagicMock()
    close_client(client)
    client.close.assert_called_once()
    # None client should not raise
    close_client(None)


def test_check_api_status_success(mock_client, log_capture):
    mock_client.models.get.return_value = MagicMock()
    assert check_api_status(client=mock_client, model="gemini-2.5-flash") is True
    mock_client.models.get.assert_called_once_with(model="gemini-2.5-flash")
    assert any("connection check successful" in r.getMessage() for r in log_capture)


def test_check_api_status_missing_key(monkeypatch, log_capture):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(
        "qs.credentials.Credentials.get_api_key",
        MagicMock(side_effect=MissingAPIKeyError("No key found")),
    )
    assert check_api_status() is False
    assert any("API key missing" in r.getMessage() for r in log_capture)


def test_check_api_status_api_error(mock_client, log_capture):
    mock_client.models.get.side_effect = errors.APIError(403, {"error": {"message": "Forbidden"}})
    assert check_api_status(client=mock_client) is False
    assert any("status check failed" in r.getMessage() for r in log_capture)


def test_check_api_status_system_exit(mock_client):
    mock_client.models.get.side_effect = SystemExit(1)
    with pytest.raises(SystemExit):
        check_api_status(client=mock_client)


# --- Request Validation Tests ---


def test_validate_request_prompt_only():
    contents = validate_request(prompt="What is the capital of France?")
    assert contents == ["What is the capital of France?"]


def test_validate_request_prompt_and_image():
    fake_png = b"\x89PNG\r\n\x1a\nfakeimagebytes"
    contents = validate_request(prompt="Analyze this", images=fake_png)
    assert len(contents) == 2
    assert contents[0] == "Analyze this"
    assert isinstance(contents[1], types.Part)


def test_validate_request_image_list():
    fake_png = b"\x89PNG\r\n\x1a\nfakeimagebytes"
    contents = validate_request(images=[fake_png, fake_png])
    assert len(contents) == 2
    assert all(isinstance(p, types.Part) for p in contents)


def test_validate_request_empty_fails():
    with pytest.raises(LLMInferenceError, match="At least one prompt, image"):
        validate_request()


def test_validate_request_empty_prompt_fails():
    with pytest.raises(LLMInferenceError, match="must not be empty"):
        validate_request(prompt="   ")


def test_validate_request_invalid_prompt_type():
    with pytest.raises(LLMInferenceError, match="must be a string"):
        validate_request(prompt=12345)  # type: ignore


def test_validate_request_empty_image_bytes():
    with pytest.raises(LLMInferenceError, match="non-empty bytes"):
        validate_request(images=b"")


# --- Response Validation Tests ---


def test_validate_response_text():
    resp = MagicMock(spec=types.GenerateContentResponse)
    candidate = MagicMock()
    candidate.finish_reason = types.FinishReason.STOP
    resp.candidates = [candidate]
    resp.text = "Hello world"
    assert validate_response(resp) == "Hello world"


def test_validate_response_no_candidates():
    resp = MagicMock(spec=types.GenerateContentResponse)
    resp.candidates = []
    with pytest.raises(LLMInferenceError, match="no response candidates"):
        validate_response(resp)


def test_validate_response_safety_system_exit(log_capture):
    resp = MagicMock(spec=types.GenerateContentResponse)
    candidate = MagicMock()
    candidate.finish_reason = types.FinishReason.SAFETY
    resp.candidates = [candidate]
    with pytest.raises(LLMInferenceError, match="blocked by safety policy"):
        validate_response(resp)
    assert any("blocked by GenAI safety policy" in r.getMessage() for r in log_capture)


def test_validate_response_recitation_system_exit():
    resp = MagicMock(spec=types.GenerateContentResponse)
    candidate = MagicMock()
    candidate.finish_reason = types.FinishReason.RECITATION
    resp.candidates = [candidate]
    with pytest.raises(LLMInferenceError, match="blocked by safety policy"):
        validate_response(resp)


def test_validate_response_schema_sdk_parsed():
    resp = MagicMock(spec=types.GenerateContentResponse)
    candidate = MagicMock()
    candidate.finish_reason = types.FinishReason.STOP
    resp.candidates = [candidate]
    resp.parsed = SampleAnswer(choice="B", confidence=0.95)
    parsed = validate_response(resp, response_schema=SampleAnswer)
    assert parsed.choice == "B"
    assert parsed.confidence == 0.95


def test_validate_response_schema_json_text_fallback():
    resp = MagicMock(spec=types.GenerateContentResponse)
    candidate = MagicMock()
    candidate.finish_reason = types.FinishReason.STOP
    resp.candidates = [candidate]
    resp.parsed = None
    resp.text = '{"choice": "A", "confidence": 0.88}'
    parsed = validate_response(resp, response_schema=SampleAnswer)
    assert parsed.choice == "A"
    assert parsed.confidence == 0.88


def test_validate_response_schema_invalid_json():
    resp = MagicMock(spec=types.GenerateContentResponse)
    candidate = MagicMock()
    candidate.finish_reason = types.FinishReason.STOP
    resp.candidates = [candidate]
    resp.parsed = None
    resp.text = "This is not valid json"
    with pytest.raises(LLMParsingError, match="Failed to parse response"):
        validate_response(resp, response_schema=SampleAnswer)


# --- Send Request, Retry, and Fallback Tests ---


def test_send_request_success(mock_client):
    resp = MagicMock(spec=types.GenerateContentResponse)
    candidate = MagicMock()
    candidate.finish_reason = types.FinishReason.STOP
    resp.candidates = [candidate]
    resp.text = "Answer text"
    mock_client.models.generate_content.return_value = resp

    result = send_request(prompt="Test prompt", client=mock_client)
    assert result == "Answer text"
    mock_client.models.generate_content.assert_called_once()


def test_send_request_retry_transient_then_succeeds(mock_client, log_capture):
    err = errors.APIError(429, {"error": {"message": "Rate limit exceeded"}})
    resp = MagicMock(spec=types.GenerateContentResponse)
    candidate = MagicMock()
    candidate.finish_reason = types.FinishReason.STOP
    resp.candidates = [candidate]
    resp.text = "Success after retry"

    mock_client.models.generate_content.side_effect = [err, resp]

    result = send_request(
        prompt="Test prompt",
        client=mock_client,
        max_retries=2,
        initial_retry_delay=0.01,
    )
    assert result == "Success after retry"
    assert mock_client.models.generate_content.call_count == 2
    assert any("Transient error querying Gemini model" in r.getMessage() for r in log_capture)


def test_send_request_fallback_model_on_failure(mock_client, log_capture):
    err = errors.APIError(500, {"error": {"message": "Internal Server Error"}})
    fallback_resp = MagicMock(spec=types.GenerateContentResponse)
    candidate = MagicMock()
    candidate.finish_reason = types.FinishReason.STOP
    fallback_resp.candidates = [candidate]
    fallback_resp.text = "Fallback model success"

    def side_effect(model, contents, config):
        if model == DEFAULT_GEMINI_MODEL:
            raise err
        if model == FALLBACK_GEMINI_MODEL:
            return fallback_resp
        raise ValueError(f"Unexpected model {model}")

    mock_client.models.generate_content.side_effect = side_effect

    result = send_request(
        prompt="Test prompt",
        client=mock_client,
        model=DEFAULT_GEMINI_MODEL,
        fallback_model=FALLBACK_GEMINI_MODEL,
        max_retries=1,
        initial_retry_delay=0.01,
    )
    assert result == "Fallback model success"
    assert any("Initiating fallback to model" in r.getMessage() for r in log_capture)


def test_send_request_authentication_error(mock_client):
    mock_client.models.generate_content.side_effect = errors.APIError(
        403, {"error": {"message": "API key invalid"}}
    )
    with pytest.raises(LLMAuthenticationError, match="Gemini authentication failed"):
        send_request(
            prompt="Test prompt",
            client=mock_client,
            fallback_model=None,
            max_retries=0,
        )


def test_send_request_rate_limit_error(mock_client):
    mock_client.models.generate_content.side_effect = errors.APIError(
        429, {"error": {"message": "Resource exhausted"}}
    )
    with pytest.raises(LLMRateLimitError, match="Gemini rate limit exceeded"):
        send_request(
            prompt="Test prompt",
            client=mock_client,
            fallback_model=None,
            max_retries=0,
        )


# --- GeminiService Class Tests ---


def test_gemini_service_lifecycle(mock_client):
    service = GeminiService(api_key="AIzaSy" + "S" * 33)
    service._client = mock_client

    mock_client.models.get.return_value = MagicMock()
    assert service.check_status() is True

    resp = MagicMock(spec=types.GenerateContentResponse)
    candidate = MagicMock()
    candidate.finish_reason = types.FinishReason.STOP
    resp.candidates = [candidate]
    resp.text = "Service response"
    mock_client.models.generate_content.return_value = resp

    out = service.send(prompt="Hello")
    assert out == "Service response"

    service.close()
    mock_client.close.assert_called_once()
    assert service._client is None


def test_gemini_service_context_manager(mock_client):
    with GeminiService(api_key="AIzaSy" + "S" * 33) as svc:
        svc._client = mock_client
    mock_client.close.assert_called_once()
