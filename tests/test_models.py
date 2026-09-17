"""Tests for Pydantic data schemas in qs.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from qs.models import (
    EssaySolution,
    FillInBlankSolution,
    GenericQuestionSolution,
    MatchingSolution,
    MultipleChoiceSolution,
    QuestionContext,
    QuestionExecutionResult,
    QuestionType,
    QuizBatchResult,
    SingleChoiceSolution,
    TrueFalseSolution,
    parse_question_solution,
)

# --- QuestionType Tests ---


def test_question_type_values():
    assert QuestionType.SINGLE_CHOICE == "single_choice"
    assert QuestionType.MULTIPLE_CHOICE == "multiple_choice"
    assert QuestionType.TRUE_FALSE == "true_false"
    assert QuestionType.FILL_IN_BLANK == "fill_in_blank"
    assert QuestionType.MATCHING == "matching"
    assert QuestionType.ESSAY == "essay"


# --- QuestionContext Tests ---


def test_question_context_valid():
    ctx = QuestionContext(
        question_id="q1",
        question_type=QuestionType.SINGLE_CHOICE,
        prompt="Which sorting algorithm is O(n log n)?",
        points=10.0,
        options=["Bubble Sort", "Merge Sort", "Quick Sort", "Insertion Sort"],
        order=1,
    )
    assert ctx.question_id == "q1"
    assert ctx.question_type == QuestionType.SINGLE_CHOICE
    assert len(ctx.options) == 4
    assert ctx.screenshot_bytes is None


def test_question_context_with_screenshot():
    fake_png = b"\x89PNG\r\n\x1a\nfakeimagebytes"
    ctx = QuestionContext(
        question_id="q2",
        question_type=QuestionType.MULTIPLE_CHOICE,
        prompt="Select all FIFO data structures:",
        options=["Queue", "Stack", "Deque"],
        screenshot_bytes=fake_png,
    )
    assert ctx.screenshot_bytes == fake_png


# --- Individual Solution Models Tests ---


def test_single_choice_solution():
    sol = SingleChoiceSolution(
        selected_option="Merge Sort",
        confidence=0.95,
        explanation="Merge sort is guaranteed O(n log n) in all cases.",
    )
    assert sol.question_type == QuestionType.SINGLE_CHOICE
    assert sol.selected_option == "Merge Sort"
    assert sol.confidence == 0.95


def test_multiple_choice_solution():
    sol = MultipleChoiceSolution(
        selected_options=["Queue", "Deque (Double-Ended Queue)"],
        confidence=0.9,
    )
    assert sol.question_type == QuestionType.MULTIPLE_CHOICE
    assert len(sol.selected_options) == 2


def test_true_false_solution():
    sol = TrueFalseSolution(value=True, confidence=1.0)
    assert sol.question_type == QuestionType.TRUE_FALSE
    assert sol.value is True


def test_fill_in_blank_solution():
    sol = FillInBlankSolution(answers=["log n"], confidence=0.85)
    assert sol.question_type == QuestionType.FILL_IN_BLANK
    assert sol.answers == ["log n"]


def test_matching_solution():
    pairs = {"HTTP": "Port 80", "HTTPS": "Port 443", "SSH": "Port 22"}
    sol = MatchingSolution(pairs=pairs)
    assert sol.question_type == QuestionType.MATCHING
    assert sol.pairs["HTTPS"] == "Port 443"


def test_essay_solution():
    sol = EssaySolution(
        response_text="Hash table lookup averages O(1) through bucket hashing.",
        confidence=0.99,
    )
    assert sol.question_type == QuestionType.ESSAY
    assert "O(1)" in sol.response_text


# --- Discriminated Union & Parsing Tests ---


def test_parse_single_choice_from_dict():
    data = {
        "question_type": "single_choice",
        "selected_option": "Quick Sort",
        "confidence": 0.8,
    }
    sol = parse_question_solution(data)
    assert isinstance(sol, SingleChoiceSolution)
    assert sol.selected_option == "Quick Sort"


def test_parse_multiple_choice_from_json():
    json_str = (
        '{"question_type": "multiple_choice", "selected_options": ["A", "B"], "confidence": 1.0}'
    )
    sol = parse_question_solution(json_str)
    assert isinstance(sol, MultipleChoiceSolution)
    assert sol.selected_options == ["A", "B"]


def test_parse_matching_from_dict():
    data = {
        "question_type": "matching",
        "pairs": {"HTTP": "Port 80"},
    }
    sol = parse_question_solution(data)
    assert isinstance(sol, MatchingSolution)
    assert sol.pairs == {"HTTP": "Port 80"}


def test_parse_invalid_discriminator():
    data = {"question_type": "unknown_type", "selected_option": "Option A"}
    with pytest.raises(ValidationError):
        parse_question_solution(data)


def test_confidence_validation():
    with pytest.raises(ValidationError):
        SingleChoiceSolution(selected_option="A", confidence=1.5)
    with pytest.raises(ValidationError):
        SingleChoiceSolution(selected_option="A", confidence=-0.1)


# --- Generic Solution & Results Tests ---


def test_generic_question_solution():
    sol = GenericQuestionSolution(
        question_id="q1",
        question_type=QuestionType.SINGLE_CHOICE,
        selected_option="Merge Sort",
    )
    assert sol.selected_option == "Merge Sort"


def test_quiz_batch_result():
    q1_res = QuestionExecutionResult(
        question_id="q1",
        question_type=QuestionType.SINGLE_CHOICE,
        prompt="Which sort?",
        filled=True,
        duration_seconds=1.2,
    )
    q2_res = QuestionExecutionResult(
        question_id="q2",
        question_type=QuestionType.TRUE_FALSE,
        prompt="Is HTTP stateless?",
        filled=False,
        error="Locator timed out",
        duration_seconds=2.0,
    )
    batch = QuizBatchResult(
        total_questions=2,
        successful_fills=1,
        failed_fills=1,
        results=[q1_res, q2_res],
        execution_time_seconds=3.2,
        status="partial",
    )
    assert batch.total_questions == 2
    assert batch.successful_fills == 1
    assert batch.failed_fills == 1
    assert batch.status == "partial"
