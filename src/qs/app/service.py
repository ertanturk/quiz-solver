"""Quiz Solver Orchestrator service coordinating Scraper, LLM, and Filler."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from playwright.sync_api import Locator, Page

from qs.config import DEFAULT_BATCH_SIZE, DEFAULT_QUESTION_WAIT_TIMEOUT_MS
from qs.errors.exceptions import QuestionNotFoundError
from qs.filler import Filler
from qs.llm.google import GeminiService
from qs.llm.prompts import (
    BATCH_SYSTEM_INSTRUCTION,
    SYSTEM_INSTRUCTION,
    build_batch_question_prompt,
    build_question_prompt,
    get_solution_schema,
)
from qs.logger import get_logger
from qs.models import (
    BatchSolution,
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
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.gemini_service = gemini_service or GeminiService()
        self.scraper = scraper or Scraper(capture_screenshots=capture_screenshots)
        self.filler = filler or Filler()
        self.capture_screenshots = capture_screenshots
        self.batch_size = max(1, batch_size)
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

    def _solve_single_task(self, task: dict[str, Any]) -> Any:
        """Execute single-question LLM inference for a scraped task."""
        context = task["context"]
        prompt = build_question_prompt(context)
        schema = get_solution_schema(context.question_type)
        return self.gemini_service.send(
            prompt=prompt,
            images=context.screenshot_bytes,
            response_schema=schema,
            system_instruction=SYSTEM_INSTRUCTION,
        )

    def _solve_task_chunk(self, chunk: list[dict[str, Any]]) -> dict[str, Any]:
        """Execute batch LLM inference for a chunk of tasks with individual fallback."""
        if len(chunk) == 1:
            task = chunk[0]
            try:
                sol = self._solve_single_task(task)
                return {task["q_id"]: sol}
            except Exception as e:
                return {task["q_id"]: e}

        # Multi-question batch inference
        try:
            contexts = [t["context"] for t in chunk]
            prompt, images = build_batch_question_prompt(contexts)
            batch_solution = self.gemini_service.send(
                prompt=prompt,
                images=images,
                response_schema=BatchSolution,
                system_instruction=BATCH_SYSTEM_INSTRUCTION,
            )

            resolved: dict[str, Any] = {}
            if isinstance(batch_solution, BatchSolution) and batch_solution.solutions:
                by_id = {item.question_id: item for item in batch_solution.solutions}
                for idx, task in enumerate(chunk):
                    item = by_id.get(task["q_id"])
                    if item is None and idx < len(batch_solution.solutions):
                        item = batch_solution.solutions[idx]
                    if item is not None:
                        try:
                            resolved[task["q_id"]] = item.to_solution()
                        except Exception as conv_err:
                            logger.warning(
                                "Failed converting batch item for %s: %s",
                                task["q_id"],
                                conv_err,
                            )

            # Individual fallback for any missing task in the batch
            for task in chunk:
                if task["q_id"] not in resolved:
                    try:
                        resolved[task["q_id"]] = self._solve_single_task(task)
                    except Exception as e:
                        resolved[task["q_id"]] = e

            return resolved

        except Exception as batch_err:
            logger.warning(
                "Batch inference failed (%s: %s). Retrying questions individually.",
                type(batch_err).__name__,
                batch_err,
            )
            fallback_resolved: dict[str, Any] = {}
            for task in chunk:
                try:
                    fallback_resolved[task["q_id"]] = self._solve_single_task(task)
                except Exception as e:
                    fallback_resolved[task["q_id"]] = e
            return fallback_resolved

    def solve_quiz(
        self,
        page: Page,
        wait_timeout_ms: int = DEFAULT_QUESTION_WAIT_TIMEOUT_MS,
        on_question_complete: Callable[[QuestionExecutionResult, int, int], None] | None = None,
        on_stage: Callable[[str, dict[str, Any]], None] | None = None,
        batch_size: int | None = None,
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

        from concurrent.futures import ThreadPoolExecutor

        results: list[QuestionExecutionResult] = []
        tasks: list[dict[str, Any]] = []
        effective_batch_size = max(1, batch_size if batch_size is not None else self.batch_size)

        with ThreadPoolExecutor(max_workers=max(1, min(total_questions, 8))) as executor:
            # 1. Scrape cards sequentially on browser thread
            for index, card in enumerate(cards, start=1):
                start_time_q = time.perf_counter()
                try:
                    raw_id = extract_question_id(card, index)
                    q_id = str(raw_id) if raw_id is not None else f"question_{index}"
                except Exception:
                    q_id = f"question_{index}"

                try:
                    context = self.scraper.scrape_question(card, order=index)
                    q_id = str(context.question_id)
                    q_type = context.question_type
                    prompt_text = context.prompt
                    schema = get_solution_schema(context.question_type)
                    tasks.append(
                        {
                            "order": index,
                            "card": card,
                            "q_id": q_id,
                            "q_type": q_type,
                            "prompt_text": prompt_text,
                            "context": context,
                            "schema": schema,
                            "start_time": start_time_q,
                            "scrape_error": None,
                        }
                    )
                except Exception as scrape_err:
                    logger.error(
                        "Failed scraping question #%d [ID: %s]: %s", index, q_id, scrape_err
                    )
                    tasks.append(
                        {
                            "order": index,
                            "card": card,
                            "q_id": q_id,
                            "q_type": QuestionType.SINGLE_CHOICE,
                            "prompt_text": "",
                            "context": None,
                            "schema": None,
                            "start_time": start_time_q,
                            "scrape_error": str(scrape_err),
                        }
                    )

            # 2. Group valid tasks into chunks of effective_batch_size and dispatch
            valid_tasks = [t for t in tasks if t["scrape_error"] is None]
            task_to_future: dict[str, Any] = {}

            if effective_batch_size <= 1:
                chunks = [[t] for t in valid_tasks]
            else:
                chunks = [
                    valid_tasks[i : i + effective_batch_size]
                    for i in range(0, len(valid_tasks), effective_batch_size)
                ]

            for chunk in chunks:
                chunk_future = executor.submit(self._solve_task_chunk, chunk)
                for t in chunk:
                    task_to_future[t["q_id"]] = chunk_future

            # 3. Await LLM responses in order, inject solutions into DOM, and emit callbacks
            for task in tasks:
                order = task["order"]
                card = task["card"]
                q_id = task["q_id"]
                q_type = task["q_type"]
                prompt_text = task["prompt_text"]
                start_time_q = task["start_time"]

                if on_stage is not None:
                    with contextlib.suppress(Exception):
                        on_stage("scrape_start", {"order": order, "card": card})

                if task["scrape_error"] is not None:
                    duration = round(time.perf_counter() - start_time_q, 2)
                    if on_stage is not None:
                        with contextlib.suppress(Exception):
                            on_stage(
                                "error",
                                {
                                    "order": order,
                                    "question_id": q_id,
                                    "error": task["scrape_error"],
                                    "duration": duration,
                                },
                            )
                    res = QuestionExecutionResult(
                        question_id=q_id,
                        question_type=q_type,
                        prompt=prompt_text,
                        solution=None,
                        filled=False,
                        error=task["scrape_error"],
                        duration_seconds=duration,
                    )
                    results.append(res)
                    if on_question_complete is not None:
                        with contextlib.suppress(Exception):
                            on_question_complete(res, order, total_questions)
                    continue

                if on_stage is not None:
                    with contextlib.suppress(Exception):
                        on_stage(
                            "scraped",
                            {
                                "order": order,
                                "question_id": q_id,
                                "question_type": q_type,
                                "prompt": prompt_text,
                                "context": task["context"],
                            },
                        )

                if on_stage is not None:
                    with contextlib.suppress(Exception):
                        on_stage(
                            "reason_start",
                            {
                                "order": order,
                                "question_id": q_id,
                                "question_type": q_type,
                                "schema": task["schema"],
                            },
                        )

                solution = None
                try:
                    chunk_future = task_to_future[q_id]
                    chunk_outcome_map = chunk_future.result()
                    outcome = chunk_outcome_map.get(q_id)

                    if isinstance(outcome, Exception):
                        raise outcome
                    if outcome is None:
                        raise RuntimeError(f"No solution returned for question {q_id}")

                    solution = outcome

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

                    self.filler.fill(card, solution)
                    duration = round(time.perf_counter() - start_time_q, 2)
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

                    res = QuestionExecutionResult(
                        question_id=q_id,
                        question_type=q_type,
                        prompt=prompt_text,
                        solution=solution,
                        filled=True,
                        duration_seconds=duration,
                    )
                    results.append(res)

                except Exception as e:
                    duration = round(time.perf_counter() - start_time_q, 2)
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
                    res = QuestionExecutionResult(
                        question_id=q_id,
                        question_type=q_type,
                        prompt=prompt_text,
                        solution=solution,
                        filled=False,
                        error=str(e),
                        duration_seconds=duration,
                    )
                    results.append(res)

                if on_question_complete is not None:
                    with contextlib.suppress(Exception):
                        on_question_complete(res, order, total_questions)

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
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> QuizBatchResult:
    """Solve all quiz questions on the page using QuizSolverService."""
    with QuizSolverService(gemini_service=gemini_service, batch_size=batch_size) as service:
        return service.solve_quiz(
            page,
            wait_timeout_ms=wait_timeout_ms,
            on_question_complete=on_question_complete,
            on_stage=on_stage,
            batch_size=batch_size,
        )
