"""Tests for question batching and quota optimization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from playwright.sync_api import Browser, Page, sync_playwright

from qs.app.service import QuizSolverService, solve_quiz
from qs.errors.exceptions import LLMInferenceError
from qs.llm.google import GeminiService
from qs.llm.prompts import (
    BATCH_SYSTEM_INSTRUCTION,
    SYSTEM_INSTRUCTION,
    build_batch_question_prompt,
)
from qs.models import (
    BatchQuestionItem,
    BatchSolution,
    EssaySolution,
    FillInBlankSolution,
    MatchingPair,
    MatchingSolution,
    MultipleChoiceSolution,
    QuestionContext,
    QuestionType,
    QuizBatchResult,
    SingleChoiceSolution,
    TrueFalseSolution,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "blackboard_quiz.html"
FIXTURE_URL = f"file://{FIXTURE_PATH.resolve()}"


@pytest.fixture(scope="module")
def browser():
    """Shared headless browser instance for batching tests."""
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


# --- Unit Tests for Batch Models & Prompt Builder ---


def test_batch_question_item_to_solution_all_types():
    """Verify BatchQuestionItem transforms into each typed solution correctly."""
    # Single choice
    sc_item = BatchQuestionItem(
        question_id="q1",
        question_type=QuestionType.SINGLE_CHOICE,
        selected_option="Merge Sort",
        confidence=0.95,
        explanation="O(n log n)",
    )
    sc_sol = sc_item.to_solution()
    assert isinstance(sc_sol, SingleChoiceSolution)
    assert sc_sol.selected_option == "Merge Sort"
    assert sc_sol.confidence == 0.95

    # Multiple choice
    mc_item = BatchQuestionItem(
        question_id="q2",
        question_type=QuestionType.MULTIPLE_CHOICE,
        selected_options=["Queue", "Stack"],
        confidence=0.90,
    )
    mc_sol = mc_item.to_solution()
    assert isinstance(mc_sol, MultipleChoiceSolution)
    assert mc_sol.selected_options == ["Queue", "Stack"]

    # True / False
    tf_item = BatchQuestionItem(
        question_id="q3",
        question_type=QuestionType.TRUE_FALSE,
        bool_value=True,
        confidence=1.0,
    )
    tf_sol = tf_item.to_solution()
    assert isinstance(tf_sol, TrueFalseSolution)
    assert tf_sol.value is True

    # Fill in blank
    fib_item = BatchQuestionItem(
        question_id="q4",
        question_type=QuestionType.FILL_IN_BLANK,
        fill_blanks=["log n"],
        confidence=0.88,
    )
    fib_sol = fib_item.to_solution()
    assert isinstance(fib_sol, FillInBlankSolution)
    assert fib_sol.answers == ["log n"]

    # Matching
    m_item = BatchQuestionItem(
        question_id="q5",
        question_type=QuestionType.MATCHING,
        matching_pairs=[
            MatchingPair(prompt="HTTP", option="Port 80"),
            MatchingPair(prompt="SSH", option="Port 22"),
        ],
        confidence=0.95,
    )
    m_sol = m_item.to_solution()
    assert isinstance(m_sol, MatchingSolution)
    assert m_sol.pairs == {"HTTP": "Port 80", "SSH": "Port 22"}

    # Essay
    e_item = BatchQuestionItem(
        question_id="q6",
        question_type=QuestionType.ESSAY,
        essay_text="Direct essay text answer.",
        confidence=0.92,
    )
    e_sol = e_item.to_solution()
    assert isinstance(e_sol, EssaySolution)
    assert e_sol.response_text == "Direct essay text answer."


def test_build_batch_question_prompt_structures_multimodal():
    """Verify build_batch_question_prompt compiles sequential questions and images."""
    ctx1 = QuestionContext(
        question_id="q1",
        question_type=QuestionType.SINGLE_CHOICE,
        prompt="Select the O(n log n) sort algorithm.",
        options=["Bubble Sort", "Merge Sort"],
        screenshot_bytes=b"fake_image_bytes_1",
    )
    ctx2 = QuestionContext(
        question_id="q2",
        question_type=QuestionType.TRUE_FALSE,
        prompt="TCP is connection-oriented.",
        options=["True", "False"],
        screenshot_bytes=b"fake_image_bytes_2",
    )

    prompt, images = build_batch_question_prompt([ctx1, ctx2])

    assert len(images) == 2
    assert images[0] == b"fake_image_bytes_1"
    assert images[1] == b"fake_image_bytes_2"
    assert "You are given 2 question(s)" in prompt
    assert "Question #1 [ID: q1] -> Screenshot #1" in prompt
    assert "Question #2 [ID: q2] -> Screenshot #2" in prompt
    assert "BatchSolution" in prompt


# --- Mock Batch Dispatcher for Fixture Integration ---


def _batch_aware_gemini_solver(
    prompt=None, images=None, response_schema=None, system_instruction=None
):
    """Mock Gemini dispatcher supporting both BatchSolution and individual schemas."""
    prompt_str = prompt or ""

    if response_schema == BatchSolution:
        solutions: list[BatchQuestionItem] = []
        if "q1" in prompt_str:
            solutions.append(
                BatchQuestionItem(
                    question_id="q1",
                    question_type=QuestionType.SINGLE_CHOICE,
                    selected_option="Merge Sort",
                    confidence=0.99,
                    explanation="Merge sort is O(n log n)",
                )
            )
        if "q2" in prompt_str:
            solutions.append(
                BatchQuestionItem(
                    question_id="q2",
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    selected_options=["Queue", "Stack"],
                    confidence=0.95,
                    explanation="Linear data structures",
                )
            )
        if "q3" in prompt_str:
            solutions.append(
                BatchQuestionItem(
                    question_id="q3",
                    question_type=QuestionType.TRUE_FALSE,
                    bool_value=True,
                    confidence=1.0,
                    explanation="TCP is connection-oriented",
                )
            )
        if "q4" in prompt_str:
            solutions.append(
                BatchQuestionItem(
                    question_id="q4",
                    question_type=QuestionType.FILL_IN_BLANK,
                    fill_blanks=["log n"],
                    confidence=0.88,
                    explanation="Binary search",
                )
            )
        if "q5" in prompt_str:
            solutions.append(
                BatchQuestionItem(
                    question_id="q5",
                    question_type=QuestionType.MATCHING,
                    matching_pairs=[
                        MatchingPair(prompt="HTTP", option="Port 80"),
                        MatchingPair(prompt="HTTPS", option="Port 443"),
                        MatchingPair(prompt="SSH", option="Port 22"),
                    ],
                    confidence=0.98,
                )
            )
        if "q6" in prompt_str:
            solutions.append(
                BatchQuestionItem(
                    question_id="q6",
                    question_type=QuestionType.ESSAY,
                    essay_text="Hash tables have O(1) expected time complexity.",
                    confidence=0.95,
                )
            )
        return BatchSolution(solutions=solutions)

    # Individual schema fallback
    if response_schema == SingleChoiceSolution:
        return SingleChoiceSolution(selected_option="Merge Sort", confidence=0.99)
    if response_schema == MultipleChoiceSolution:
        return MultipleChoiceSolution(selected_options=["Queue", "Stack"], confidence=0.95)
    if response_schema == TrueFalseSolution:
        return TrueFalseSolution(value=True, confidence=1.0)
    if response_schema == FillInBlankSolution:
        return FillInBlankSolution(answers=["log n"], confidence=0.88)
    if response_schema == MatchingSolution:
        return MatchingSolution(
            pairs={"HTTP": "Port 80", "HTTPS": "Port 443", "SSH": "Port 22"}, confidence=0.98
        )
    if response_schema == EssaySolution:
        return EssaySolution(
            response_text="Hash tables have O(1) expected time complexity.", confidence=0.95
        )

    raise ValueError(f"Unknown response_schema: {response_schema}")


# --- Integration Tests for Batch Execution & Halved API Calls ---


def test_solve_quiz_batches_2_questions_halving_requests(quiz_page: Page):
    """Verify solve_quiz with batch_size=2 solves 6 questions in exactly 3 API calls."""
    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.send.side_effect = _batch_aware_gemini_solver

    service = QuizSolverService(gemini_service=mock_gemini, batch_size=2)
    batch_result = service.solve_quiz(quiz_page)

    assert isinstance(batch_result, QuizBatchResult)
    assert batch_result.total_questions == 6
    assert batch_result.successful_fills == 6
    assert batch_result.status == "completed"

    # Crucial quota verification: 6 questions batched into chunks of 2 = 3 requests
    assert mock_gemini.send.call_count == 3

    # Verify every call used BatchSolution
    for call in mock_gemini.send.call_args_list:
        assert call.kwargs["response_schema"] == BatchSolution
        assert call.kwargs["system_instruction"] == BATCH_SYSTEM_INSTRUCTION


def test_solve_quiz_batch_size_odd_partition(quiz_page: Page):
    """Verify solve_quiz with batch_size=4 partitions 6 questions into 2 requests (4 + 2)."""
    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.send.side_effect = _batch_aware_gemini_solver

    service = QuizSolverService(gemini_service=mock_gemini, batch_size=4)
    batch_result = service.solve_quiz(quiz_page)

    assert batch_result.total_questions == 6
    assert batch_result.successful_fills == 6
    # 6 questions chunked by 4 = 2 chunks ([4], [2]) -> 2 API calls
    assert mock_gemini.send.call_count == 2


def test_solve_quiz_batch_failure_falls_back_to_individual(quiz_page: Page):
    """Verify that when a batch call raises an error, all questions in the batch fall back individually."""
    mock_gemini = MagicMock(spec=GeminiService)

    call_count = 0

    def failing_batch_solver(
        prompt=None, images=None, response_schema=None, system_instruction=None
    ):
        nonlocal call_count
        call_count += 1
        # Fail if BatchSolution is requested
        if response_schema == BatchSolution:
            raise LLMInferenceError("Batch structured output validation failed")
        return _batch_aware_gemini_solver(
            prompt=prompt,
            images=images,
            response_schema=response_schema,
            system_instruction=system_instruction,
        )

    mock_gemini.send.side_effect = failing_batch_solver

    service = QuizSolverService(gemini_service=mock_gemini, batch_size=2)
    batch_result = service.solve_quiz(quiz_page)

    assert batch_result.total_questions == 6
    assert batch_result.successful_fills == 6
    assert batch_result.status == "completed"

    # 3 failed batch calls + 6 successful individual fallback calls = 9 total calls
    assert mock_gemini.send.call_count == 9


def test_solve_quiz_partial_batch_omission_fallback(quiz_page: Page):
    """Verify that if Gemini omits a question from BatchSolution.solutions, it falls back individually."""
    mock_gemini = MagicMock(spec=GeminiService)

    def partial_batch_solver(
        prompt=None, images=None, response_schema=None, system_instruction=None
    ):
        if response_schema == BatchSolution:
            full_batch = _batch_aware_gemini_solver(
                prompt=prompt,
                images=images,
                response_schema=response_schema,
                system_instruction=system_instruction,
            )
            # Omit q2 from solutions
            full_batch.solutions = [s for s in full_batch.solutions if s.question_id != "q2"]
            return full_batch
        return _batch_aware_gemini_solver(
            prompt=prompt,
            images=images,
            response_schema=response_schema,
            system_instruction=system_instruction,
        )

    mock_gemini.send.side_effect = partial_batch_solver

    service = QuizSolverService(gemini_service=mock_gemini, batch_size=2)
    batch_result = service.solve_quiz(quiz_page)

    assert batch_result.total_questions == 6
    assert batch_result.successful_fills == 6
    assert batch_result.status == "completed"

    # 3 batch calls + 1 fallback call for missing q2 = 4 total calls
    assert mock_gemini.send.call_count == 4


def test_solve_quiz_batch_size_1_uses_single_question_mode(quiz_page: Page):
    """Verify that batch_size=1 dispatches each question with its individual schema."""
    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.send.side_effect = _batch_aware_gemini_solver

    service = QuizSolverService(gemini_service=mock_gemini, batch_size=1)
    batch_result = service.solve_quiz(quiz_page)

    assert batch_result.total_questions == 6
    assert batch_result.successful_fills == 6
    assert mock_gemini.send.call_count == 6

    # Verify no BatchSolution was used
    for call in mock_gemini.send.call_args_list:
        assert call.kwargs["response_schema"] != BatchSolution
        assert call.kwargs["system_instruction"] == SYSTEM_INSTRUCTION


def test_module_solve_quiz_with_batch_size(quiz_page: Page):
    """Verify module-level solve_quiz respects batch_size parameter."""
    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.send.side_effect = _batch_aware_gemini_solver

    batch_result = solve_quiz(quiz_page, gemini_service=mock_gemini, batch_size=2)

    assert batch_result.total_questions == 6
    assert batch_result.successful_fills == 6
    assert mock_gemini.send.call_count == 3
