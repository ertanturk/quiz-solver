"""Domain-specific exceptions for Quiz Solver."""

from typing import Any


class QuizSolverError(Exception):
    """Base exception for all Quiz Solver runtime errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# Configuration & Credentials


class ConfigurationError(QuizSolverError):
    """Raised when configuration parameters are invalid or missing."""


class MissingAPIKeyError(ConfigurationError):
    """Raised when Gemini API key cannot be located in keyring or environment."""


# Browser & Session


class BrowserSessionError(QuizSolverError):
    """Raised when Playwright fails to initialize or connect to browser."""


class PageNavigationError(BrowserSessionError):
    """Raised when page fails to load the specified URL or times out."""


# Scraper


class ScraperError(QuizSolverError):
    """Base error for DOM scraping and screenshot extraction failures."""


class QuestionNotFoundError(ScraperError):
    """Raised when no quiz question containers are found on the active page."""


class ScreenshotCaptureError(ScraperError):
    """Raised when capturing question element screenshot fails."""


class DomExtractionError(ScraperError):
    """Raised when parsing options or labels from question DOM fails."""


# LLM & Inference


class LLMInferenceError(QuizSolverError):
    """Base error for Gemini Vision inference failures."""


class LLMParsingError(LLMInferenceError):
    """Raised when LLM returns invalid JSON or fails Pydantic schema validation."""


class LLMRateLimitError(LLMInferenceError):
    """Raised when Gemini API returns 429 quota or rate-limit error."""


class LLMAuthenticationError(LLMInferenceError):
    """Raised when Gemini API returns 401/403 invalid credentials."""


# Filler & Form Injection


class FillerError(QuizSolverError):
    """Base error for UI injection and answer filling."""


class UnsupportedQuestionTypeError(FillerError):
    """Raised when question type cannot be mapped to a fill strategy."""


class InjectionFailedError(FillerError):
    """Raised when locator fails to click, select, or fill an answer."""


class ElementNotInteractableError(InjectionFailedError):
    """Raised when target input is disabled, obscured, or not actionable."""
