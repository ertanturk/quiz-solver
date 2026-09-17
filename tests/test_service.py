"""Tests for Quiz Solver Orchestrator service in qs.app.service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from playwright.sync_api import Browser, Page, sync_playwright

from qs.app.service import QuizSolverService, solve_quiz
from qs.errors.exceptions import InjectionFailedError, LLMInferenceError
from qs.filler import Filler
from qs.llm.google import GeminiService
from qs.models import (
    EssaySolution,
    FillInBlankSolution,
    MatchingSolution,
    MultipleChoiceSolution,
    QuestionExecutionResult,
    QuestionType,
    QuizBatchResult,
    SingleChoiceSolution,
    TrueFalseSolution,
)
from qs.scraper import Scraper

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "blackboard_quiz.html"
FIXTURE_URL = f"file://{FIXTURE_PATH.resolve()}"


@pytest.fixture(scope="module")
def browser():
    """Shared headless browser instance for service tests."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def quiz_page(browser: Browser):
    """Fresh page loaded with blackboard_quiz.html fixture."""
    page = browser.new_page()
    page.goto(FIXTURE_URL)
    yield page
    page.close()


# --- Unit Tests for Service Lifecycle & Component Wiring ---


def test_service_initialization_defaults():
    """Verify default initialization wires all submodules and manages lifecycle."""
    with QuizSolverService() as service:
        assert isinstance(service.gemini_service, GeminiService)
        assert isinstance(service.scraper, Scraper)
        assert isinstance(service.filler, Filler)
        assert service._owns_gemini_service is True


def test_service_custom_components_lifecycle():
    """Verify injected custom dependencies are respected and external Gemini client is not closed."""
    mock_gemini = MagicMock(spec=GeminiService)
    mock_scraper = MagicMock(spec=Scraper)
    mock_filler = MagicMock(spec=Filler)

    service = QuizSolverService(
        gemini_service=mock_gemini,
        scraper=mock_scraper,
        filler=mock_filler,
        capture_screenshots=False,
    )
    assert service.gemini_service is mock_gemini
    assert service.scraper is mock_scraper
    assert service.filler is mock_filler
    assert service.capture_screenshots is False
    assert service._owns_gemini_service is False

    service.close()
    mock_gemini.close.assert_not_called()


# --- Unit Tests for solve_question ---


def test_solve_question_success(quiz_page: Page):
    """Verify solve_question orchestrates Scraper -> LLM -> Filler and produces QuestionExecutionResult."""
    card = quiz_page.locator('[data-question-id="q1"]')

    mock_gemini = MagicMock(spec=GeminiService)
    expected_sol = SingleChoiceSolution(
        selected_option="Merge Sort",
        confidence=0.99,
        explanation="O(n log n) worst case",
    )
    mock_gemini.send.return_value = expected_sol

    service = QuizSolverService(gemini_service=mock_gemini)
    res = service.solve_question(card, order=1)

    assert isinstance(res, QuestionExecutionResult)
    assert res.question_id == "q1"
    assert res.question_type == QuestionType.SINGLE_CHOICE
    assert res.filled is True
    assert res.solution == expected_sol
    assert res.error is None
    assert res.duration_seconds >= 0.0

    mock_gemini.send.assert_called_once()
    assert card.locator('input[type="radio"][value="Merge Sort"]').is_checked() is True


def test_solve_question_scraper_failure():
    """Verify scraper exceptions are caught without raising and recorded in result."""
    mock_card = MagicMock()
    mock_scraper = MagicMock(spec=Scraper)
    mock_scraper.scrape_question.side_effect = RuntimeError("Scraping timeout")

    service = QuizSolverService(scraper=mock_scraper)
    res = service.solve_question(mock_card, order=1)

    assert res.filled is False
    assert "Scraping timeout" in (res.error or "")
    assert res.solution is None


def test_solve_question_llm_failure(quiz_page: Page):
    """Verify LLM inference exceptions are caught and recorded gracefully."""
    card = quiz_page.locator('[data-question-id="q1"]')

    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.send.side_effect = LLMInferenceError("API quota exhausted")

    service = QuizSolverService(gemini_service=mock_gemini)
    res = service.solve_question(card, order=1)

    assert res.filled is False
    assert "API quota exhausted" in (res.error or "")
    assert res.question_id == "q1"
    assert res.question_type == QuestionType.SINGLE_CHOICE


def test_solve_question_filler_failure(quiz_page: Page):
    """Verify filler injection errors are caught and recorded gracefully with the solution retained."""
    card = quiz_page.locator('[data-question-id="q1"]')

    mock_gemini = MagicMock(spec=GeminiService)
    sol = SingleChoiceSolution(selected_option="Nonexistent Option")
    mock_gemini.send.return_value = sol

    mock_filler = MagicMock(spec=Filler)
    mock_filler.fill.side_effect = InjectionFailedError("Element not found")

    service = QuizSolverService(gemini_service=mock_gemini, filler=mock_filler)
    res = service.solve_question(card, order=1)

    assert res.filled is False
    assert "Element not found" in (res.error or "")
    assert res.solution == sol


# --- End-to-End Orchestrator Tests on Fixture ---


def _mock_gemini_solver(prompt=None, images=None, response_schema=None, system_instruction=None):
    """Mock Gemini response dispatcher returning correct solution by schema type."""
    if response_schema == SingleChoiceSolution:
        return SingleChoiceSolution(
            selected_option="Merge Sort",
            confidence=0.95,
            explanation="Merge sort is O(n log n)",
        )
    if response_schema == MultipleChoiceSolution:
        return MultipleChoiceSolution(
            selected_options=["Queue", "Stack"],
            confidence=0.90,
            explanation="Linear data structures",
        )
    if response_schema == TrueFalseSolution:
        return TrueFalseSolution(
            value=True,
            confidence=1.0,
            explanation="TCP is connection-oriented",
        )
    if response_schema == FillInBlankSolution:
        return FillInBlankSolution(
            answers=["log n"],
            confidence=0.88,
            explanation="Binary search is O(log n)",
        )
    if response_schema == MatchingSolution:
        return MatchingSolution(
            pairs={
                "HTTP": "Port 80",
                "HTTPS": "Port 443",
                "SSH": "Port 22",
            },
            confidence=0.95,
        )
    if response_schema == EssaySolution:
        return EssaySolution(
            response_text="Hash tables have O(1) expected time complexity.",
            confidence=0.92,
        )
    raise ValueError(f"Unknown schema: {response_schema}")


def test_solve_quiz_full_workflow_on_fixture(quiz_page: Page):
    """Verify end-to-end quiz resolution across all 6 question types."""
    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.send.side_effect = _mock_gemini_solver

    progress_events: list[tuple[QuestionExecutionResult, int, int]] = []

    def on_progress(res: QuestionExecutionResult, current: int, total: int):
        progress_events.append((res, current, total))

    service = QuizSolverService(gemini_service=mock_gemini)
    batch_result = service.solve_quiz(quiz_page, on_question_complete=on_progress)

    # 1. Verify batch metrics
    assert isinstance(batch_result, QuizBatchResult)
    assert batch_result.total_questions == 6
    assert batch_result.successful_fills == 6
    assert batch_result.failed_fills == 0
    assert batch_result.status == "completed"
    assert len(batch_result.results) == 6
    assert batch_result.execution_time_seconds >= 0.0

    # 2. Verify progress callbacks
    assert len(progress_events) == 6
    assert [p[1] for p in progress_events] == [1, 2, 3, 4, 5, 6]
    assert all(p[2] == 6 for p in progress_events)

    # 3. Verify Blackboard DOM state updated (all 6 questions answered)
    remaining_badge = quiz_page.locator("#questions-remaining-indicator")
    assert "0 OF 6 QUESTIONS REMAINING" in remaining_badge.inner_text()


def test_solve_quiz_never_clicks_submit(quiz_page: Page):
    """Critical safety test: Ensure the submit button is NEVER clicked during quiz solving."""
    # Track submit button clicks via JavaScript
    quiz_page.evaluate(
        """() => {
            window.__submit_clicked = false;
            const submitBtn = document.getElementById('submit-quiz-btn');
            if (submitBtn) {
                submitBtn.addEventListener('click', () => {
                    window.__submit_clicked = true;
                });
            }
        }"""
    )

    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.send.side_effect = _mock_gemini_solver

    service = QuizSolverService(gemini_service=mock_gemini)
    service.solve_quiz(quiz_page)

    # Verify submit button was never clicked
    submit_clicked = quiz_page.evaluate("() => window.__submit_clicked")
    assert submit_clicked is False


def test_solve_quiz_resilience_to_partial_failure(quiz_page: Page):
    """Verify that a failure on one question does NOT halt execution for other questions."""
    mock_gemini = MagicMock(spec=GeminiService)

    def failing_gemini_solver(
        prompt=None, images=None, response_schema=None, system_instruction=None
    ):
        # Fail on question 3 (TrueFalseSolution)
        if response_schema == TrueFalseSolution:
            raise LLMInferenceError("Simulated LLM network drop on Q3")
        return _mock_gemini_solver(
            prompt=prompt,
            images=images,
            response_schema=response_schema,
            system_instruction=system_instruction,
        )

    mock_gemini.send.side_effect = failing_gemini_solver

    service = QuizSolverService(gemini_service=mock_gemini)
    batch = service.solve_quiz(quiz_page)

    assert batch.total_questions == 6
    assert batch.successful_fills == 5
    assert batch.failed_fills == 1
    assert batch.status == "partial"

    # Verify Q3 specifically failed
    q3_res = next(r for r in batch.results if r.question_id == "q3")
    assert q3_res.filled is False
    assert "Simulated LLM network drop on Q3" in (q3_res.error or "")

    # Verify other questions succeeded
    assert all(r.filled is True for r in batch.results if r.question_id != "q3")


def test_solve_quiz_no_questions_found(browser: Browser):
    """Verify handling when page contains no quiz question containers."""
    page = browser.new_page()
    page.goto("data:text/html,<html><body><div>Not a quiz</div></body></html>")

    service = QuizSolverService()
    batch = service.solve_quiz(page, wait_timeout_ms=100)

    assert batch.total_questions == 0
    assert batch.successful_fills == 0
    assert batch.failed_fills == 0
    assert batch.status == "failed"
    assert batch.results == []
    page.close()


def test_module_solve_quiz_convenience_function(quiz_page: Page):
    """Verify module-level solve_quiz convenience function."""
    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.send.side_effect = _mock_gemini_solver

    batch = solve_quiz(quiz_page, gemini_service=mock_gemini)
    assert batch.total_questions == 6
    assert batch.successful_fills == 6
    assert batch.status == "completed"
