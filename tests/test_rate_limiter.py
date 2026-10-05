"""Unit tests for RateLimiter and LLM integration."""

from __future__ import annotations

import concurrent.futures
from unittest.mock import MagicMock

import pytest
from google.genai import types

from qs.config import FREE_TIER_RPD, FREE_TIER_RPM, FREE_TIER_TPM
from qs.errors.exceptions import LLMRateLimitError
from qs.llm.google import GeminiService, send_request
from qs.llm.rate_limiter import (
    RateLimiter,
    estimate_tokens,
    get_default_rate_limiter,
    set_default_rate_limiter,
)
from qs.rate_limiter import RateLimiter as TopLevelRateLimiter


class FakeClock:
    """Simulates monotonic clock advancement for deterministic testing."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self.current_time = start_time
        self.sleep_history: list[float] = []

    def now(self) -> float:
        return self.current_time

    def sleep(self, duration: float) -> None:
        self.sleep_history.append(duration)
        self.current_time += duration


def test_top_level_import_identity():
    """Verify top-level import aliases the implementation."""
    assert TopLevelRateLimiter is RateLimiter


def test_estimate_tokens():
    """Verify token estimation for text, images, and multimodal content."""
    # Text
    assert estimate_tokens(prompt="12345678") == 2
    assert estimate_tokens(prompt="short") == 1

    # Single and multiple images (approx 258 tokens per tile)
    fake_img = b"\x89PNG\r\n\x1a\nfake"
    assert estimate_tokens(images=fake_img) == 258
    assert estimate_tokens(images=[fake_img, fake_img]) == 516

    # Mixed prompt and image
    assert estimate_tokens(prompt="A" * 40, images=fake_img) == 10 + 258

    # Empty payload defaults to at least 1
    assert estimate_tokens() == 1


def test_rate_limiter_defaults():
    """Verify default quotas match Google Gemini Free Tier."""
    limiter = RateLimiter()
    assert limiter.rpm == FREE_TIER_RPM  # 5
    assert limiter.tpm == FREE_TIER_TPM  # 250,000
    assert limiter.rpd == FREE_TIER_RPD  # 20


def test_rate_limiter_rpm_throttling():
    """Verify that exceeding 5 RPM triggers sleep until earliest request expires."""
    clock = FakeClock()
    limiter = RateLimiter(rpm=5, tpm=250_000, rpd=20, clock=clock.now, sleep_fn=clock.sleep)

    # First 5 requests spaced 10 seconds apart: t=1000, 1010, 1020, 1030, 1040
    for _ in range(5):
        limiter.acquire(tokens=10)
        clock.current_time += 10.0

    stats = limiter.get_stats()
    # At t=1050, all 5 requests (1000, 1010, 1020, 1030, 1040) are within 60s window (cutoff is 990)
    assert stats["rpm_used"] == 5
    assert stats["rpm_remaining"] == 0
    assert len(clock.sleep_history) == 0

    # 6th request at t=1050: window is full. Oldest request was at 1000.
    # It must wait until 1000 + 60 = 1060 (wait ~10s).
    limiter.acquire(tokens=10)
    assert len(clock.sleep_history) == 1
    assert 9.9 <= clock.sleep_history[0] <= 10.1

    # After sleep to 1060.01, request 1 (1000) expired, but requests 2-5 (1010, 1020, 1030, 1040) + request 6 (1060.01) remain
    stats = limiter.get_stats()
    assert stats["rpm_used"] == 5


def test_rate_limiter_tpm_throttling():
    """Verify exceeding TPM triggers sleep."""
    clock = FakeClock()
    limiter = RateLimiter(rpm=10, tpm=1000, rpd=20, clock=clock.now, sleep_fn=clock.sleep)

    # 900 tokens acquired
    limiter.acquire(tokens=900)
    assert len(clock.sleep_history) == 0

    # Request requiring 200 more tokens exceeds 1000 TPM limit
    limiter.acquire(tokens=200)
    assert len(clock.sleep_history) == 1
    assert clock.sleep_history[0] >= 60.0


def test_rate_limiter_rpd_exhaustion():
    """Verify exceeding daily 20 RPD raises LLMRateLimitError."""
    clock = FakeClock()
    limiter = RateLimiter(rpm=100, tpm=1_000_000, rpd=20, clock=clock.now, sleep_fn=clock.sleep)

    for _ in range(20):
        limiter.acquire(tokens=1)

    assert limiter.get_stats()["rpd_used"] == 20
    assert limiter.get_stats()["rpd_remaining"] == 0

    # 21st request in the 24-hour period must fail
    with pytest.raises(LLMRateLimitError, match="daily quota of 20 requests"):
        limiter.acquire(tokens=1)

    # Fast forward 24 hours + 1 second
    clock.current_time += 86401.0
    # Now slot is free
    limiter.acquire(tokens=1)
    assert limiter.get_stats()["rpd_used"] == 1


def test_rate_limiter_reset():
    """Verify reset clears internal tracking queues."""
    limiter = RateLimiter(rpm=5, rpd=20)
    limiter.acquire(tokens=100)
    stats = limiter.get_stats()
    assert stats["rpm_used"] == 1
    assert stats["rpd_used"] == 1

    limiter.reset()
    stats_after = limiter.get_stats()
    assert stats_after["rpm_used"] == 0
    assert stats_after["rpd_used"] == 0


def test_rate_limiter_thread_safety():
    """Verify concurrent acquires maintain correct counts and do not race."""
    limiter = RateLimiter(rpm=50, tpm=100_000, rpd=50)

    def worker():
        limiter.acquire(tokens=5)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker) for _ in range(20)]
        for f in futures:
            f.result()

    stats = limiter.get_stats()
    assert stats["rpm_used"] == 20
    assert stats["rpd_used"] == 20
    assert stats["tpm_used"] == 100


def test_send_request_integrates_rate_limiter():
    """Verify send_request calls rate_limiter.acquire with estimated tokens."""
    mock_client = MagicMock()
    resp = MagicMock(spec=types.GenerateContentResponse)
    candidate = MagicMock()
    candidate.finish_reason = types.FinishReason.STOP
    resp.candidates = [candidate]
    resp.text = "Answer"
    mock_client.models.generate_content.return_value = resp

    mock_limiter = MagicMock(spec=RateLimiter)
    mock_limiter.estimate_tokens.return_value = 42

    result = send_request(
        prompt="Test question",
        client=mock_client,
        rate_limiter=mock_limiter,
    )
    assert result == "Answer"
    mock_limiter.estimate_tokens.assert_called_once()
    mock_limiter.acquire.assert_called_once_with(tokens=42)


def test_gemini_service_uses_configured_rate_limiter():
    """Verify GeminiService passes its configured rate limiter to send_request."""
    mock_client = MagicMock()
    resp = MagicMock(spec=types.GenerateContentResponse)
    candidate = MagicMock()
    candidate.finish_reason = types.FinishReason.STOP
    resp.candidates = [candidate]
    resp.text = "Service response"
    mock_client.models.generate_content.return_value = resp

    mock_limiter = MagicMock(spec=RateLimiter)
    mock_limiter.estimate_tokens.return_value = 15

    service = GeminiService(api_key="AIzaSy" + "K" * 33, rate_limiter=mock_limiter)
    service._client = mock_client

    out = service.send(prompt="What is X?")
    assert out == "Service response"
    mock_limiter.acquire.assert_called_once_with(tokens=15)


def test_gemini_service_bypass_rate_limiter():
    """Verify passing rate_limiter=False bypasses rate limiting."""
    mock_client = MagicMock()
    resp = MagicMock(spec=types.GenerateContentResponse)
    candidate = MagicMock()
    candidate.finish_reason = types.FinishReason.STOP
    resp.candidates = [candidate]
    resp.text = "Bypassed response"
    mock_client.models.generate_content.return_value = resp

    mock_limiter = MagicMock(spec=RateLimiter)

    service = GeminiService(api_key="AIzaSy" + "K" * 33, rate_limiter=mock_limiter)
    service._client = mock_client

    out = service.send(prompt="No rate limiting", rate_limiter=False)
    assert out == "Bypassed response"
    mock_limiter.acquire.assert_not_called()


def test_global_rate_limiter_accessor():
    """Verify get_default_rate_limiter and set_default_rate_limiter."""
    orig = get_default_rate_limiter()
    assert isinstance(orig, RateLimiter)

    custom = RateLimiter(rpm=2, rpd=5)
    set_default_rate_limiter(custom)
    assert get_default_rate_limiter() is custom

    # Restore original
    set_default_rate_limiter(orig)


def test_rate_limiter_create_paid_tier():
    """Verify create_paid_tier initializes with paid tier limits."""
    limiter = RateLimiter.create_paid_tier()
    assert limiter.enabled is True
    assert limiter.rpm == 1000
    assert limiter.tpm == 4_000_000
    assert limiter.rpd == 100_000


def test_rate_limiter_disabled_does_not_block():
    """Verify disabled rate limiter never sleeps or blocks requests."""
    clock = FakeClock()
    limiter = RateLimiter(enabled=False, rpm=1, clock=clock.now, sleep_fn=clock.sleep)

    # Acquire 100 times without delay or exception
    for _ in range(100):
        limiter.acquire(tokens=10_000)

    assert clock.sleep_history == []
    stats = limiter.get_stats()
    assert stats["enabled"] is False

