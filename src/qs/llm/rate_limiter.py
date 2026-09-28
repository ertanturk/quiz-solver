"""Rate limiter for Google Gemini API conforming to Free Tier limits.

Free Tier Quotas:
- RPM: 5 requests per minute
- TPM: 250,000 tokens per minute
- RPD: 20 requests per day
"""

from __future__ import annotations

import collections
import threading
import time
from collections.abc import Callable
from typing import Any

from qs.config import FREE_TIER_RPD, FREE_TIER_RPM, FREE_TIER_TPM
from qs.errors.exceptions import LLMRateLimitError
from qs.logger import get_logger

logger = get_logger(__name__)

# Standard Gemini vision token count per image tile (~258 tokens)
DEFAULT_IMAGE_TOKENS = 258
CHARS_PER_TOKEN = 4


def estimate_tokens(
    prompt: str | None = None,
    images: list[bytes] | bytes | None = None,
    contents: list[Any] | None = None,
) -> int:
    """Estimate token count for Gemini request payloads."""
    tokens = 0

    if prompt:
        tokens += max(1, len(prompt) // CHARS_PER_TOKEN)

    if images:
        raw_images = [images] if isinstance(images, bytes) else images
        tokens += len(raw_images) * DEFAULT_IMAGE_TOKENS

    if contents:
        for item in contents:
            if isinstance(item, str):
                tokens += max(1, len(item) // CHARS_PER_TOKEN)
            elif hasattr(item, "text") and getattr(item, "text", None):
                tokens += max(1, len(str(item.text)) // CHARS_PER_TOKEN)
            elif (
                hasattr(item, "inline_data") and getattr(item, "inline_data", None)
            ) or isinstance(item, (bytes, bytearray)):
                tokens += DEFAULT_IMAGE_TOKENS

            else:
                tokens += max(1, len(str(item)) // CHARS_PER_TOKEN)

    return max(1, tokens)


class RateLimiter:
    """Thread-safe rate limiter enforcing Google Gemini Free Tier quotas.

    Auto-throttles by sleeping when 5 RPM or 250k TPM thresholds are reached.
    Raises LLMRateLimitError when daily 20 RPD quota is exhausted.
    """

    def __init__(
        self,
        rpm: int = FREE_TIER_RPM,
        tpm: int = FREE_TIER_TPM,
        rpd: int = FREE_TIER_RPD,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self._clock = clock
        self._sleep_fn = sleep_fn
        self._lock = threading.Lock()

        # Sliding 60-second window: entries are (timestamp, tokens)
        self._requests_window: collections.deque[tuple[float, int]] = collections.deque()
        # Sliding 24-hour (86400s) window: entries are timestamps
        self._daily_requests: collections.deque[float] = collections.deque()

    def _purge_expired(self, now: float) -> None:
        """Purge entries outside the sliding 60s and 24h windows."""
        minute_cutoff = now - 60.0
        while self._requests_window and self._requests_window[0][0] <= minute_cutoff:
            self._requests_window.popleft()

        day_cutoff = now - 86400.0
        while self._daily_requests and self._daily_requests[0] <= day_cutoff:
            self._daily_requests.popleft()

    def acquire(self, tokens: int = 1) -> None:
        """Acquire permission to execute an LLM request.

        Blocks and sleeps until RPM and TPM windows have available capacity.

        Raises:
            LLMRateLimitError: If daily request quota (RPD) is exceeded.
        """
        while True:
            sleep_duration = 0.0
            with self._lock:
                now = self._clock()
                self._purge_expired(now)

                # Check daily quota (RPD)
                if len(self._daily_requests) >= self.rpd:
                    logger.error(
                        "Gemini Free Tier daily quota exhausted (%d/%d RPD)",
                        len(self._daily_requests),
                        self.rpd,
                    )
                    raise LLMRateLimitError(
                        f"Gemini Free Tier daily quota of {self.rpd} requests per day (RPD) exceeded",
                        details={
                            "rpd_limit": self.rpd,
                            "daily_requests": len(self._daily_requests),
                        },
                    )

                # Check RPM window (5 requests per 60 seconds)
                if len(self._requests_window) >= self.rpm:
                    oldest_ts = self._requests_window[0][0]
                    sleep_duration = max(sleep_duration, oldest_ts + 60.0 - now)

                # Check TPM window (250,000 tokens per 60 seconds)
                current_tokens = sum(t for _, t in self._requests_window)
                if current_tokens + tokens > self.tpm:
                    freed = 0
                    for ts, tok in self._requests_window:
                        freed += tok
                        if current_tokens - freed + tokens <= self.tpm:
                            sleep_duration = max(sleep_duration, ts + 60.0 - now)
                            break

                # If no sleep needed, register request and proceed
                if sleep_duration <= 0.0:
                    self._requests_window.append((now, tokens))
                    self._daily_requests.append(now)
                    logger.debug(
                        "Rate limiter slot acquired (tokens: %d, RPM: %d/%d, RPD: %d/%d)",
                        tokens,
                        len(self._requests_window),
                        self.rpm,
                        len(self._daily_requests),
                        self.rpd,
                    )
                    return

            # Sleep outside lock to let other threads inspect state
            logger.info(
                "Gemini Free Tier limit reached; throttling for %.2fs...",
                sleep_duration,
            )
            self._sleep_fn(sleep_duration + 0.01)

    def estimate_tokens(
        self,
        prompt: str | None = None,
        images: list[bytes] | bytes | None = None,
        contents: list[Any] | None = None,
    ) -> int:
        """Estimate tokens for a given request payload."""
        return estimate_tokens(prompt=prompt, images=images, contents=contents)

    def get_stats(self) -> dict[str, Any]:
        """Return current rate limiter metrics and remaining allowances."""
        with self._lock:
            now = self._clock()
            self._purge_expired(now)
            current_tokens = sum(t for _, t in self._requests_window)
            return {
                "rpm_limit": self.rpm,
                "rpm_used": len(self._requests_window),
                "rpm_remaining": max(0, self.rpm - len(self._requests_window)),
                "tpm_limit": self.tpm,
                "tpm_used": current_tokens,
                "tpm_remaining": max(0, self.tpm - current_tokens),
                "rpd_limit": self.rpd,
                "rpd_used": len(self._daily_requests),
                "rpd_remaining": max(0, self.rpd - len(self._daily_requests)),
            }

    def reset(self) -> None:
        """Reset all tracked request timestamps and token counters."""
        with self._lock:
            self._requests_window.clear()
            self._daily_requests.clear()


_default_limiter: RateLimiter | None = None
_limiter_lock = threading.Lock()


def get_default_rate_limiter() -> RateLimiter:
    """Return the shared global RateLimiter instance."""
    global _default_limiter
    if _default_limiter is None:
        with _limiter_lock:
            if _default_limiter is None:
                _default_limiter = RateLimiter()
    return _default_limiter


def set_default_rate_limiter(limiter: RateLimiter | None) -> None:
    """Override or reset the shared global RateLimiter instance."""
    global _default_limiter
    with _limiter_lock:
        _default_limiter = limiter
