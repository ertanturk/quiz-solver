"""Tests for DOM form injection in qs.filler."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, sync_playwright

from qs.errors.exceptions import InjectionFailedError, UnsupportedQuestionTypeError
from qs.filler import (
    Filler,
    fill_question,
)
from qs.filler.filler import (
    logger as filler_logger,
)
from qs.models import (
    EssaySolution,
    FillInBlankSolution,
    GenericQuestionSolution,
    MatchingSolution,
    MultipleChoiceSolution,
    QuestionType,
    SingleChoiceSolution,
    TrueFalseSolution,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "blackboard_quiz.html"
FIXTURE_URL = f"file://{FIXTURE_PATH.resolve()}"


@pytest.fixture(scope="module")
def browser():
    """Shared headless browser instance for filler fixture tests."""
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


@pytest.fixture
def log_capture():
    """Capture logs emitted by filler module."""
    records = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = CaptureHandler()
    handler.setLevel(logging.DEBUG)
    filler_logger.addHandler(handler)
    original_level = filler_logger.level
    filler_logger.setLevel(logging.DEBUG)
    yield records
    filler_logger.removeHandler(handler)
    filler_logger.setLevel(original_level)


# --- Question Type Injection Tests ---


def test_fill_single_choice(quiz_page: Page, log_capture):
    card = quiz_page.locator('[data-question-id="q1"]')
    sol = SingleChoiceSolution(
        selected_option="Merge Sort",
        confidence=0.95,
        explanation="Merge Sort is guaranteed O(n log n)",
    )

    success = fill_question(card, sol)
    assert success is True

    # Assert correct radio is checked
    target_radio = card.locator('input[type="radio"][value="Merge Sort"]')
    assert target_radio.is_checked() is True

    # Assert other radios are unchecked
    other_radio = card.locator('input[type="radio"][value="Bubble Sort"]')
    assert other_radio.is_checked() is False

    # Assert parent choice-item received is-selected class
    parent_item = target_radio.locator("xpath=ancestor::li[contains(@class, 'choice-item')]")
    assert "is-selected" in (parent_item.get_attribute("class") or "")

    assert any(
        "Successfully injected solution for single_choice" in r.getMessage() for r in log_capture
    )


def test_fill_multiple_choice(quiz_page: Page):
    card = quiz_page.locator('[data-question-id="q2"]')
    sol = MultipleChoiceSolution(
        selected_options=["Queue", "Stack"],
        confidence=0.90,
    )

    success = fill_question(card, sol)
    assert success is True

    cb_queue = card.locator('input[type="checkbox"][value="Queue"]')
    cb_stack = card.locator('input[type="checkbox"][value="Stack"]')
    cb_deque = card.locator('input[type="checkbox"][value="Deque (Double-Ended Queue)"]')
    cb_bst = card.locator('input[type="checkbox"][value="Binary Search Tree"]')

    assert cb_queue.is_checked() is True
    assert cb_stack.is_checked() is True
    assert cb_deque.is_checked() is False
    assert cb_bst.is_checked() is False


def test_fill_multiple_choice_fallback_when_choice_items_empty_inputs(browser: Browser):
    page = browser.new_page()
    html_content = """
    <article data-question-id="q_mc_fallback">
      <div class="choice-item">Text 1</div>
      <div class="choice-item">Text 2</div>
      <input type="checkbox" value="Option A" />
      <input type="checkbox" value="Option B" />
    </article>
    """
    page.goto(f"data:text/html,{html_content}")
    card = page.locator("article").first
    sol = MultipleChoiceSolution(selected_options=["Option A"])

    assert fill_question(card, sol) is True
    assert card.locator('input[value="Option A"]').is_checked() is True
    assert card.locator('input[value="Option B"]').is_checked() is False
    page.close()


def test_fill_true_false(quiz_page: Page):
    card = quiz_page.locator('[data-question-id="q3"]')
    sol = TrueFalseSolution(value=True, confidence=1.0)

    success = fill_question(card, sol)
    assert success is True

    tf_true = card.locator('input[type="radio"][value="True"]')
    tf_false = card.locator('input[type="radio"][value="False"]')

    assert tf_true.is_checked() is True
    assert tf_false.is_checked() is False


def test_fill_fill_in_blank(quiz_page: Page):
    card = quiz_page.locator('[data-question-id="q4"]')
    sol = FillInBlankSolution(answers=["log n"], confidence=0.88)

    success = fill_question(card, sol)
    assert success is True

    fib_input = card.locator(".fib-input")
    assert fib_input.input_value() == "log n"


def test_fill_matching(quiz_page: Page):
    card = quiz_page.locator('[data-question-id="q5"]')
    sol = MatchingSolution(
        pairs={
            "HTTP": "Port 80",
            "HTTPS": "Port 443",
            "SSH": "Port 22",
        },
        confidence=0.95,
    )

    success = fill_question(card, sol)
    assert success is True

    sel_http = card.locator("#q5-match-0")
    sel_https = card.locator("#q5-match-1")
    sel_ssh = card.locator("#q5-match-2")

    assert sel_http.input_value() == "Port 80"
    assert sel_https.input_value() == "Port 443"
    assert sel_ssh.input_value() == "Port 22"


def test_fill_essay(quiz_page: Page):
    card = quiz_page.locator('[data-question-id="q6"]')
    essay_text = "In-memory hash tables have O(1) expected time due to uniform distribution, but hash collisions can degrade performance to O(n)."
    sol = EssaySolution(response_text=essay_text, confidence=0.95)

    success = fill_question(card, sol)
    assert success is True

    essay_textarea = card.locator("#q6-essay-input")
    assert essay_textarea.input_value() == essay_text


# --- Full Quiz Fill & Blackboard Auto-Save Tracking ---


def test_fill_entire_quiz_updates_progress_bar(quiz_page: Page):
    """Fill all 6 questions sequentially and verify Blackboard auto-save counter reaches 0 remaining."""
    filler = Filler()

    # Q1
    filler.fill(
        quiz_page.locator('[data-question-id="q1"]'),
        SingleChoiceSolution(selected_option="Merge Sort"),
    )
    # Q2
    filler.fill(
        quiz_page.locator('[data-question-id="q2"]'),
        MultipleChoiceSolution(selected_options=["Queue", "Stack"]),
    )
    # Q3
    filler.fill(
        quiz_page.locator('[data-question-id="q3"]'),
        TrueFalseSolution(value=True),
    )
    # Q4
    filler.fill(
        quiz_page.locator('[data-question-id="q4"]'),
        FillInBlankSolution(answers=["log n"]),
    )
    # Q5
    filler.fill(
        quiz_page.locator('[data-question-id="q5"]'),
        MatchingSolution(pairs={"HTTP": "Port 80", "HTTPS": "Port 443", "SSH": "Port 22"}),
    )
    # Q6
    filler.fill(
        quiz_page.locator('[data-question-id="q6"]'),
        EssaySolution(response_text="Complete essay response"),
    )

    remaining_badge = quiz_page.locator("#questions-remaining-indicator")
    assert "0 OF 6 QUESTIONS REMAINING" in remaining_badge.inner_text()


# --- GenericQuestionSolution Support ---


def test_fill_with_generic_solution(quiz_page: Page):
    card = quiz_page.locator('[data-question-id="q1"]')
    generic_sol = GenericQuestionSolution(
        question_type=QuestionType.SINGLE_CHOICE,
        selected_option="Quick Sort",
    )
    assert fill_question(card, generic_sol) is True
    assert card.locator('input[type="radio"][value="Quick Sort"]').is_checked() is True


# --- Error Handling Tests ---


def test_fill_missing_option_raises(quiz_page: Page):
    card = quiz_page.locator('[data-question-id="q1"]')
    sol = SingleChoiceSolution(selected_option="Nonexistent Option")
    with pytest.raises(InjectionFailedError, match="Could not locate radio option"):
        fill_question(card, sol)


def test_fill_unsupported_question_type(quiz_page: Page):
    from unittest.mock import MagicMock

    card = quiz_page.locator('[data-question-id="q1"]')
    mock_solution = MagicMock()
    mock_solution.question_type = "unsupported_type"
    with pytest.raises(UnsupportedQuestionTypeError, match="Unsupported question type"):
        fill_question(card, mock_solution)


# --- Real Assessment Fixture Tests ---

VIEW_ASSESSMENT_PATH = Path(__file__).parent / "fixtures" / "View_Assessment.html"
if not VIEW_ASSESSMENT_PATH.exists():
    VIEW_ASSESSMENT_PATH = Path(__file__).parent / "fixtures" / "View Assessment.html"
VIEW_ASSESSMENT_URL = f"file://{VIEW_ASSESSMENT_PATH.resolve()}"


def test_fill_view_assessment_questions(browser: Browser):
    page = browser.new_page()
    page.goto(VIEW_ASSESSMENT_URL)

    cards = page.locator(".assessment-question").all()
    assert len(cards) == 20

    # Fill Q1 with correct target option text
    card1 = cards[0]
    sol1 = SingleChoiceSolution(
        question_id="q1",
        selected_option="To compare how algorithms grow in resource usage as input size increases",
        explanation="Definition of asymptotic analysis",
    )
    assert fill_question(card1, sol1) is True

    # Check that option 5 radio in card 1 is checked
    q1_labels = card1.locator("label").all()
    assert q1_labels[4].locator('input[type="radio"]').first.is_checked() is True

    # Fill Q2 with mathematical formula option
    card2 = cards[1]
    sol2 = SingleChoiceSolution(
        question_id="q2",
        selected_option="T(n) = 2T(n/2) + Θ(n)",
        explanation="Standard recurrence for merge sort",
    )
    assert fill_question(card2, sol2) is True

    q2_labels = card2.locator("label").all()
    assert q2_labels[2].locator('input[type="radio"]').first.is_checked() is True

    # Also test option index/letter fallback on Q3
    card3 = cards[2]
    sol3 = SingleChoiceSolution(
        question_id="q3",
        selected_option="Option B",
        explanation="Testing option letter fallback",
    )
    assert fill_question(card3, sol3) is True
    q3_labels = card3.locator("label").all()
    assert q3_labels[1].locator('input[type="radio"]').first.is_checked() is True

    page.close()
