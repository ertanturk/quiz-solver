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
    SYSTEM_INSTRUCTION,
    build_question_prompt,
    get_solution_schema,
)

__all__ = [
    "GeminiService",
    "SYSTEM_INSTRUCTION",
    "build_question_prompt",
    "check_api_status",
    "close_client",
    "get_genai_client",
    "get_solution_schema",
    "send_request",
    "validate_request",
    "validate_response",
]
