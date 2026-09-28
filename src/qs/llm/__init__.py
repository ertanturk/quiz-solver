"""LLM service module for Google Gemini."""

from qs.llm.google import (
    GeminiService,
    check_api_status,
    close_client,
    get_genai_client,
    send_request,
    validate_request,
    validate_response,
)
from qs.llm.prompts import (
    BATCH_SYSTEM_INSTRUCTION,
    SYSTEM_INSTRUCTION,
    build_batch_question_prompt,
    build_question_prompt,
    get_solution_schema,
)
from qs.llm.rate_limiter import (
    RateLimiter,
    estimate_tokens,
    get_default_rate_limiter,
    set_default_rate_limiter,
)

__all__ = [
    "BATCH_SYSTEM_INSTRUCTION",
    "GeminiService",
    "RateLimiter",
    "SYSTEM_INSTRUCTION",
    "build_batch_question_prompt",
    "build_question_prompt",
    "check_api_status",
    "close_client",
    "estimate_tokens",
    "get_default_rate_limiter",
    "get_genai_client",
    "get_solution_schema",
    "send_request",
    "set_default_rate_limiter",
    "validate_request",
    "validate_response",
]
