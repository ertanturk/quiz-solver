"""Quiz Solver Orchestrator service coordinating Scraper, LLM, and Filler."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from playwright.sync_api import Locator, Page

from qs.config import DEFAULT_QUESTION_WAIT_TIMEOUT_MS
from qs.errors.exceptions import QuestionNotFoundError
from qs.filler import Filler
from qs.llm.google import GeminiService
from qs.llm.prompts import SYSTEM_INSTRUCTION, build_question_prompt, get_solution_schema
from qs.logger import get_logger
from qs.models import (
    QuestionExecutionResult,
    QuestionType,
    QuizBatchResult,
)
from qs.scraper import Scraper, find_question_elements
from qs.scraper.scraper import extract_question_id

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class QuizSolverService:
    """Orchestrates end-to-end quiz solving: Scraper -> LLM -> Filler."""

    def __init__(
        self,
        gemini_service: GeminiService | None = None,
        scraper: Scraper | None = None,
        filler: Filler | None = None,
        capture_screenshots: bool = True,
    ) -> None:
        self.gemini_service = gemini_service or GeminiService()
        self.scraper = scraper or Scraper(capture_screenshots=capture_screenshots)
        self.filler = filler or Filler()
        self.capture_screenshots = capture_screenshots
        self._owns_gemini_service = gemini_service is None

    def solve_question(
        self,
        card: Locator,
        order: int,
        on_stage: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> QuestionExecutionResult:
        """Process, infer, and inject solution for a single question element.

        Handles extraction, prompt generation, LLM reasoning, and DOM injection.
        Catches any failure gracefully to ensure remaining questions can proceed.
        """
        start_time = time.perf_counter()

        # Best-effort initial metadata in case scraping fails early
        try:
            raw_id = extract_question_id(card, order)
            q_id = str(raw_id) if raw_id is not None else f"question_{order}"
        except Exception:
            q_id = f"question_{order}"

        q_type = QuestionType.SINGLE_CHOICE
        prompt_text = ""
        solution = None

        if on_stage is not None:
            with contextlib.suppress(Exception):
                on_stage("scrape_start", {"order": order, "card": card})

        try:
            # 1. Scrape question context and bounded screenshot
            context = self.scraper.scrape_question(card, order=order)
            q_id = str(context.question_id)
            q_type = context.question_type
            prompt_text = context.prompt

            if on_stage is not None:
                with contextlib.suppress(Exception):
                    on_stage(
                        "scraped",
                        {
                            "order": order,
                            "question_id": q_id,
                            "question_type": q_type,
                            "prompt": prompt_text,
                            "context": context,
                        },
                    )

            logger.info(
                "Processing question #%d [ID: %s, Type: %s]",
                order,
                q_id,
                q_type.value,
            )

            # Build prompt and resolve strict Pydantic schema
            prompt = build_question_prompt(context)
            schema = get_solution_schema(context.question_type)

            if on_stage is not None:
                with contextlib.suppress(Exception):
                    on_stage(
                        "reason_start",
                        {
                            "order": order,
                            "question_id": q_id,
                            "question_type": q_type,
                            "schema": schema,
                        },
                    )

            # Call LLM for vision-guided structured inference
            solution = self.gemini_service.send(
                prompt=prompt,
                images=context.screenshot_bytes,
                response_schema=schema,
                system_instruction=SYSTEM_INSTRUCTION,
            )

            if on_stage is not None:
                with contextlib.suppress(Exception):
                    on_stage(
                        "reasoned",
                        {
                            "order": order,
                            "question_id": q_id,
                            "question_type": q_type,
                            "solution": solution,
                        },
                    )

            if on_stage is not None:
                with contextlib.suppress(Exception):
                    on_stage(
                        "fill_start",
                        {
                            "order": order,
                            "question_id": q_id,
                            "solution": solution,
                        },
                    )

            # Inject solution into DOM with synthetic event dispatch
            self.filler.fill(card, solution)

            duration = round(time.perf_counter() - start_time, 2)
            logger.info(
                "Question #%d [ID: %s] successfully answered and filled in %.2fs",
                order,
                q_id,
                duration,
            )

            if on_stage is not None:
                with contextlib.suppress(Exception):
                    on_stage(
                        "filled",
                        {
                            "order": order,
                            "question_id": q_id,
                            "solution": solution,
                            "duration": duration,
                        },
                    )

            return QuestionExecutionResult(
                question_id=q_id,
                question_type=q_type,
                prompt=prompt_text,
                solution=solution,
                filled=True,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = round(time.perf_counter() - start_time, 2)
            logger.error(
                "Failed processing question #%d [ID: %s]: %s",
                order,
                q_id,
                e,
            )
            if on_stage is not None:
                with contextlib.suppress(Exception):
                    on_stage(
                        "error",
                        {
                            "order": order,
                            "question_id": q_id,
                            "error": str(e),
                            "duration": duration,
                        },
                    )
            return QuestionExecutionResult(
                question_id=q_id,
                question_type=q_type,
                prompt=prompt_text,
                solution=solution,
                filled=False,
                error=str(e),
                duration_seconds=duration,
            )

    def solve_quiz(
        self,
        page: Page,
        wait_timeout_ms: int = DEFAULT_QUESTION_WAIT_TIMEOUT_MS,
        on_question_complete: Callable[[QuestionExecutionResult, int, int], None] | None = None,
        on_stage: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> QuizBatchResult:
        """Scan page for quiz questions, solve, and inject each sequentially.

        CRITICAL SAFETY RULE: This function NEVER submits or finalizes the quiz.
        The user retains final review and submission authority.
        """
        start_time = time.perf_counter()
        logger.info("Initiating quiz solve orchestration (poll timeout: %dms)...", wait_timeout_ms)

        if on_stage is not None:
            with contextlib.suppress(Exception):
                on_stage("discovery_start", {"timeout_ms": wait_timeout_ms})

        try:
            cards = find_question_elements(page, wait_timeout_ms=wait_timeout_ms)
        except QuestionNotFoundError as e:
            logger.warning(
                "No question elements found on the page within %dms: %s", wait_timeout_ms, e
            )
            total_duration = round(time.perf_counter() - start_time, 2)
            return QuizBatchResult(
                total_questions=0,
                successful_fills=0,
                failed_fills=0,
                results=[],
                execution_time_seconds=total_duration,
                status="failed",
            )
        except Exception as e:
            logger.error("Unexpected error locating question containers: %s", e)
            total_duration = round(time.perf_counter() - start_time, 2)
            return QuizBatchResult(
                total_questions=0,
                successful_fills=0,
                failed_fills=0,
                results=[],
                execution_time_seconds=total_duration,
                status="failed",
            )

        total_questions = len(cards)
        logger.info("Identified %d question container(s) to process", total_questions)

        if on_stage is not None:
            with contextlib.suppress(Exception):
                on_stage("discovered", {"total": total_questions, "cards": cards})

        results: list[QuestionExecutionResult] = []
        for index, card in enumerate(cards, start=1):
            res = self.solve_question(card, order=index, on_stage=on_stage)
            results.append(res)
            if on_question_complete is not None:
                with contextlib.suppress(Exception):
                    on_question_complete(res, index, total_questions)

        successful_fills = sum(1 for r in results if r.filled)
        failed_fills = total_questions - successful_fills
        total_duration = round(time.perf_counter() - start_time, 2)

        if failed_fills == 0 and total_questions > 0:
            status = "completed"
        elif successful_fills > 0:
            status = "partial"
        else:
            status = "failed"

        logger.info(
            "Quiz processing completed in %.2fs: %d/%d succeeded, %d failed (status: %s)",
            total_duration,
            successful_fills,
            total_questions,
            failed_fills,
            status,
        )

        return QuizBatchResult(
            total_questions=total_questions,
            successful_fills=successful_fills,
            failed_fills=failed_fills,
            results=results,
            execution_time_seconds=total_duration,
            status=status,
        )

    def close(self) -> None:
        """Close underlying resources."""
        if self._owns_gemini_service:
            self.gemini_service.close()

    def __enter__(self) -> QuizSolverService:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def solve_quiz(
    page: Page,
    gemini_service: GeminiService | None = None,
    wait_timeout_ms: int = DEFAULT_QUESTION_WAIT_TIMEOUT_MS,
    on_question_complete: Callable[[QuestionExecutionResult, int, int], None] | None = None,
    on_stage: Callable[[str, dict[str, Any]], None] | None = None,
) -> QuizBatchResult:
    """Solve all quiz questions on the page using QuizSolverService."""
    with QuizSolverService(gemini_service=gemini_service) as service:
        return service.solve_quiz(
            page,
            wait_timeout_ms=wait_timeout_ms,
            on_question_complete=on_question_complete,
            on_stage=on_stage,
        )
