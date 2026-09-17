"""Google Gemini LLM client and inference API for Quiz Solver."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from qs.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_INITIAL_RETRY_DELAY,
    DEFAULT_MAX_RETRIES,
    FALLBACK_GEMINI_MODEL,
)
from qs.credentials import Credentials
from qs.errors.exceptions import (
    ConfigurationError,
    LLMAuthenticationError,
    LLMInferenceError,
    LLMParsingError,
    LLMRateLimitError,
    MissingAPIKeyError,
)
from qs.logger import get_logger
from qs.models import MatchingLLMSchema, MatchingSolution

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# GenAI system finish reasons that indicate safety intervention or policy exit
SAFETY_FINISH_REASONS = {
    types.FinishReason.SAFETY,
    types.FinishReason.RECITATION,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.SPII,
    types.FinishReason.IMAGE_SAFETY,
}

# 429 Too Many Requests, 500 Internal Server Error, 502 Bad Gateway, 503 Service Unavailable, 504 Gateway Timeout
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def get_genai_client(api_key: str | None = None) -> genai.Client:
    """Create and return a configured google.genai Client.

    Retrieves the API key via Credentials if not explicitly supplied.
    """
    resolved_key = Credentials.get_api_key(api_key=api_key)
    logger.debug("Initializing Google GenAI client")
    return genai.Client(api_key=resolved_key)


def close_client(client: genai.Client | None) -> None:
    """Safely close a google.genai Client instance to release network resources."""
    if client is not None:
        try:
            client.close()
            logger.debug("Closed google.genai Client")
        except Exception as e:
            logger.warning("Error encountered while closing google.genai Client: %s", e)


def check_api_status(
    client: genai.Client | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
) -> bool:
    """Check status of Gemini API connection and model accessibility.

    Returns:
        bool: True if connection and API key are valid, False otherwise.
    """
    logger.debug("Checking Gemini API connection status for model '%s'", model)
    try:
        active_client = client or get_genai_client(api_key=api_key)
        active_client.models.get(model=model)
        logger.info("Gemini API connection check successful for model '%s'", model)
        return True
    except KeyboardInterrupt, SystemExit:
        raise
    except (MissingAPIKeyError, ConfigurationError) as e:
        logger.warning("Gemini API status check failed: API key missing or invalid: %s", e)
        return False
    except errors.APIError as e:
        logger.warning(
            "Gemini API status check failed (status %s): %s",
            getattr(e, "code", "UNKNOWN"),
            getattr(e, "message", str(e)),
        )
        return False
    except Exception as e:
        logger.warning("Gemini API status check failed with unexpected error: %s", e)
        return False


def validate_request(
    prompt: str | None = None,
    images: list[bytes] | bytes | None = None,
    contents: list[Any] | Any | None = None,
) -> list[Any]:
    """Validate request inputs before sending to Gemini API.

    Ensures prompt, images, or contents are present and well-formed.
    Returns normalized contents list ready for generate_content.

    Raises:
        LLMInferenceError: If inputs are invalid or empty.
    """
    normalized: list[Any] = []

    if contents is not None:
        if isinstance(contents, (list, tuple)):
            if not contents:
                raise LLMInferenceError("Contents list must not be empty")
            normalized.extend(contents)
        else:
            normalized.append(contents)

    if prompt is not None:
        if not isinstance(prompt, str):
            raise LLMInferenceError("Prompt must be a string")
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise LLMInferenceError("Prompt must not be empty or whitespace")
        normalized.append(cleaned_prompt)

    if images is not None:
        raw_images = [images] if isinstance(images, bytes) else images
        if not isinstance(raw_images, (list, tuple)):
            raise LLMInferenceError("Images must be bytes or a list of bytes")
        for idx, img in enumerate(raw_images):
            if not isinstance(img, bytes) or len(img) == 0:
                raise LLMInferenceError(f"Image at index {idx} must be non-empty bytes")
            normalized.append(types.Part.from_bytes(data=img, mime_type="image/png"))

    if not normalized:
        raise LLMInferenceError("At least one prompt, image, or content item must be provided")

    return normalized


def validate_response(
    response: types.GenerateContentResponse,
    response_schema: type[BaseModel] | None = None,
) -> Any:
    """Validate and parse Gemini response.

    Checks candidates, handles GenAI system finish reasons (safety, limits),
    and validates structured Pydantic schema if requested.

    Raises:
        LLMInferenceError: When response contains no candidates or safety policy triggered.
        LLMParsingError: When response fails structured schema validation.
    """
    candidates = response.candidates if response else None
    if not candidates:
        logger.error("Gemini returned empty response with no candidates")
        raise LLMInferenceError("Gemini returned no response candidates")

    candidate = candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)

    # Follow GenAI system exits: handle safety and content blocks gracefully
    if finish_reason in SAFETY_FINISH_REASONS:
        logger.warning(
            "Gemini response blocked by GenAI safety policy: finish_reason=%s",
            finish_reason,
        )
        raise LLMInferenceError(
            f"Gemini generation blocked by safety policy: {finish_reason}",
            details={"finish_reason": str(finish_reason)},
        )

    if finish_reason == types.FinishReason.MAX_TOKENS:
        logger.warning("Gemini response reached MAX_TOKENS limit; content may be truncated")

    if response_schema is not None:
        # Check if SDK already parsed into requested schema
        sdk_parsed = getattr(response, "parsed", None)
        if sdk_parsed is not None:
            if isinstance(sdk_parsed, response_schema):
                logger.debug(
                    "Validated response via SDK parsed schema %s", response_schema.__name__
                )
                return sdk_parsed
            try:
                data = sdk_parsed.model_dump() if hasattr(sdk_parsed, "model_dump") else sdk_parsed
                validated = response_schema.model_validate(data)
                logger.debug("Validated SDK parsed dict into schema %s", response_schema.__name__)
                return validated
            except Exception as e:
                logger.warning(
                    "SDK parsed object did not match %s directly: %s. Falling back to text JSON",
                    response_schema.__name__,
                    e,
                )

        # Fallback to validating raw JSON text
        text = getattr(response, "text", None)
        if not text or not text.strip():
            logger.error(
                "Empty text in Gemini response when expecting schema %s", response_schema.__name__
            )
            raise LLMParsingError(
                f"Expected structured schema {response_schema.__name__} but received empty response text",
                details={"finish_reason": str(finish_reason)},
            )

        try:
            parsed_obj = response_schema.model_validate_json(text)
            logger.debug("Successfully validated response text via model_validate_json")
            return parsed_obj
        except Exception as e:
            logger.error(
                "Failed to parse Gemini response text into schema %s: %s",
                response_schema.__name__,
                e,
            )
            raise LLMParsingError(
                f"Failed to parse response into {response_schema.__name__}: {e}",
                details={"text": text, "error": str(e)},
            ) from e

    text = getattr(response, "text", None)
    if text is None:
        logger.warning("Gemini response candidate text is None (finish_reason=%s)", finish_reason)
        return ""

    return text


def _is_transient_error(e: Exception) -> bool:
    """Check if exception represents a retryable transient error."""
    if isinstance(e, errors.APIError):
        code = getattr(e, "code", None)
        return code in TRANSIENT_STATUS_CODES or code is None
    return isinstance(e, (httpx.RequestError, ConnectionError, TimeoutError))


def _map_gemini_error(e: Exception) -> Exception:
    """Map raw Google GenAI and network errors into typed domain exceptions."""
    if isinstance(
        e,
        (
            LLMInferenceError,
            LLMAuthenticationError,
            LLMRateLimitError,
            LLMParsingError,
        ),
    ):
        return e

    if isinstance(e, errors.APIError):
        code = getattr(e, "code", None)
        message = getattr(e, "message", str(e))
        if code in (401, 403):
            return LLMAuthenticationError(
                f"Gemini authentication failed (status {code}): {message}",
                details={"code": code, "message": message},
            )
        if code == 429:
            return LLMRateLimitError(
                f"Gemini rate limit exceeded (status {code}): {message}",
                details={"code": code, "message": message},
            )
        return LLMInferenceError(
            f"Gemini API error (status {code}): {message}",
            details={"code": code, "message": message},
        )

    return LLMInferenceError(
        f"Gemini request failed: {e}",
        details={"original_error": str(e)},
    )


def _execute_generation(
    client: genai.Client,
    model: str,
    contents: list[Any],
    config: types.GenerateContentConfig,
    response_schema: type[BaseModel] | None,
    max_retries: int,
    initial_retry_delay: float,
) -> Any:
    """Execute generate_content with retry logic on transient errors."""
    for attempt in range(max_retries + 1):
        try:
            logger.debug(
                "Sending request to Gemini model '%s' (attempt %d/%d)",
                model,
                attempt + 1,
                max_retries + 1,
            )
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return validate_response(response, response_schema=response_schema)
        except KeyboardInterrupt, SystemExit:
            raise
        except Exception as e:
            if _is_transient_error(e) and attempt < max_retries:
                delay = initial_retry_delay * (2**attempt)
                logger.warning(
                    "Transient error querying Gemini model '%s': %s. Retrying in %.2fs (attempt %d/%d)",
                    model,
                    e,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(delay)
                continue
            raise


def send_request(
    prompt: str | None = None,
    images: list[bytes] | bytes | None = None,
    contents: list[Any] | Any | None = None,
    client: genai.Client | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    fallback_model: str | None = FALLBACK_GEMINI_MODEL,
    system_instruction: str | None = None,
    response_schema: type[BaseModel] | None = None,
    temperature: float = 0.2,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_retry_delay: float = DEFAULT_INITIAL_RETRY_DELAY,
) -> Any:
    """Send multimodal or text request to Gemini with retry, fallback, and validation.

    Args:
        prompt: Optional textual prompt.
        images: Optional image bytes or list of image bytes.
        contents: Optional raw contents list.
        client: Optional genai.Client instance.
        api_key: Optional Gemini API key override.
        model: Primary model name.
        fallback_model: Fallback model name if primary fails.
        system_instruction: Optional system instruction prompt.
        response_schema: Optional Pydantic model class for structured output.
        temperature: Sampling temperature (default: 0.2).
        max_retries: Maximum transient retry attempts.
        initial_retry_delay: Initial retry backoff in seconds.

    Returns:
        Structured Pydantic model instance if response_schema provided, else text string.

    Raises:
        LLMAuthenticationError: If API credentials invalid (401/403).
        LLMRateLimitError: If rate limit exceeded (429).
        LLMParsingError: If response fails schema parsing.
        LLMInferenceError: On generation failure or safety block.
    """
    validated_contents = validate_request(
        prompt=prompt,
        images=images,
        contents=contents,
    )

    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        "thinking_config": types.ThinkingConfig(thinking_budget=0),
    }
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    api_schema = response_schema
    if response_schema is MatchingSolution:
        api_schema = MatchingLLMSchema

    if api_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = api_schema

    config = types.GenerateContentConfig(**config_kwargs)

    try:
        active_client = client or get_genai_client(api_key=api_key)
    except Exception as e:
        logger.error("Failed to acquire Gemini client: %s", e)
        raise _map_gemini_error(e) from e

    # Primary model attempt
    try:
        return _execute_generation(
            client=active_client,
            model=model,
            contents=validated_contents,
            config=config,
            response_schema=response_schema,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay,
        )
    except KeyboardInterrupt, SystemExit:
        raise
    except Exception as primary_err:
        # Check if fallback model can be used
        if fallback_model and fallback_model != model:
            logger.warning(
                "Primary model '%s' failed: %s. Initiating fallback to model '%s'",
                model,
                primary_err,
                fallback_model,
            )
            try:
                return _execute_generation(
                    client=active_client,
                    model=fallback_model,
                    contents=validated_contents,
                    config=config,
                    response_schema=response_schema,
                    max_retries=max_retries,
                    initial_retry_delay=initial_retry_delay,
                )
            except KeyboardInterrupt, SystemExit:
                raise
            except Exception as fallback_err:
                logger.error(
                    "Fallback model '%s' also failed: %s",
                    fallback_model,
                    fallback_err,
                )
                raise _map_gemini_error(fallback_err) from fallback_err

        logger.error("Request to model '%s' failed: %s", model, primary_err)
        raise _map_gemini_error(primary_err) from primary_err


class GeminiService:
    """Encapsulates Gemini API client lifecycle, status checking, and requests."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        fallback_model: str | None = FALLBACK_GEMINI_MODEL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_retry_delay: float = DEFAULT_INITIAL_RETRY_DELAY,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.fallback_model = fallback_model
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        """Lazily initialize and return the google.genai Client."""
        if self._client is None:
            self._client = get_genai_client(api_key=self.api_key)
        return self._client

    def check_status(self) -> bool:
        """Check connection status for the configured model."""
        return check_api_status(client=self.client, model=self.model)

    def send(
        self,
        prompt: str | None = None,
        images: list[bytes] | bytes | None = None,
        contents: list[Any] | Any | None = None,
        system_instruction: str | None = None,
        response_schema: type[BaseModel] | None = None,
        temperature: float = 0.2,
        model: str | None = None,
        fallback_model: str | None = None,
    ) -> Any:
        """Send a request via this service instance."""
        target_model = model or self.model
        target_fallback = fallback_model if fallback_model is not None else self.fallback_model
        return send_request(
            prompt=prompt,
            images=images,
            contents=contents,
            client=self.client,
            model=target_model,
            fallback_model=target_fallback,
            system_instruction=system_instruction,
            response_schema=response_schema,
            temperature=temperature,
            max_retries=self.max_retries,
            initial_retry_delay=self.initial_retry_delay,
        )

    def close(self) -> None:
        """Cleanly close underlying client connection."""
        close_client(self._client)
        self._client = None

    def __enter__(self) -> GeminiService:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
