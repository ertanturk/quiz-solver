"""Tests for DOM scraping and screenshot extraction in qs.scraper."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from playwright.sync_api import Browser, Page, sync_playwright
from playwright.sync_api import Error as PlaywrightError

from qs.errors.exceptions import QuestionNotFoundError, ScreenshotCaptureError
from qs.models import QuestionType
from qs.scraper import (
    Scraper,
    capture_question_screenshot,
    detect_question_type,
    extract_matching_data,
    find_question_elements,
    scrape_question,
    scrape_quiz,
)
from qs.scraper.scraper import (
    logger as scraper_logger,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "blackboard_quiz.html"
FIXTURE_URL = f"file://{FIXTURE_PATH.resolve()}"


@pytest.fixture(scope="module")
def browser():
    """Shared headless browser instance for scraper fixture tests."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def quiz_page(browser: Browser):
    """New page loaded with blackboard_quiz.html fixture."""
    page = browser.new_page()
    page.goto(FIXTURE_URL)
    yield page
    page.close()


@pytest.fixture
def log_capture():
    """Capture logs emitted by scraper module."""
    records = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = CaptureHandler()
    handler.setLevel(logging.DEBUG)
    scraper_logger.addHandler(handler)
    original_level = scraper_logger.level
    scraper_logger.setLevel(logging.DEBUG)
    yield records
    scraper_logger.removeHandler(handler)
    scraper_logger.setLevel(original_level)


# --- Boundary Detection Tests ---


def test_find_question_elements(quiz_page: Page):
    elements = find_question_elements(quiz_page)
    assert len(elements) == 6


def test_find_question_elements_missing_raises(browser: Browser):
    page = browser.new_page()
    page.goto("data:text/html,<html><body><div>No questions here</div></body></html>")
    with pytest.raises(QuestionNotFoundError, match="No question containers found"):
        find_question_elements(page)
    page.close()


# --- Full Quiz Scrape Tests ---


def test_scrape_quiz_all_questions(quiz_page: Page, log_capture):
    contexts = scrape_quiz(quiz_page, capture_screenshots=True)
    assert len(contexts) == 6

    # Verify orders are 1..6
    assert [c.order for c in contexts] == [1, 2, 3, 4, 5, 6]

    # Verify all captured screenshots
    for c in contexts:
        assert c.screenshot_bytes is not None
        assert c.screenshot_bytes.startswith(b"\x89PNG")

    # Logging verification
    assert any("Found 6 question container(s)" in r.getMessage() for r in log_capture)
    assert any("Successfully scraped 6/6 questions" in r.getMessage() for r in log_capture)


def test_scrape_quiz_without_screenshots(quiz_page: Page):
    contexts = scrape_quiz(quiz_page, capture_screenshots=False)
    assert len(contexts) == 6
    for c in contexts:
        assert c.screenshot_bytes is None


# --- Question 1: Single Choice (Radio) ---


def test_question_1_single_choice(quiz_page: Page):
    elements = find_question_elements(quiz_page)
    ctx = scrape_question(elements[0], order=1, capture_screenshot=False)

    assert ctx.question_id == "q1"
    assert ctx.question_type == QuestionType.SINGLE_CHOICE
    assert ctx.points == 10.0
    assert "sorting algorithms guarantees a worst-case" in ctx.prompt
    assert ctx.options == ["Bubble Sort", "Merge Sort", "Quick Sort", "Insertion Sort"]
    assert ctx.matching_prompts == []
    assert ctx.matching_options == []


# --- Question 2: Multiple Choice (Checkboxes) ---


def test_question_2_multiple_choice(quiz_page: Page):
    elements = find_question_elements(quiz_page)
    ctx = scrape_question(elements[1], order=2, capture_screenshot=False)

    assert ctx.question_id == "q2"
    assert ctx.question_type == QuestionType.MULTIPLE_CHOICE
    assert ctx.points == 10.0
    assert "FIFO (First-In, First-Out)" in ctx.prompt
    assert ctx.options == [
        "Queue",
        "Stack",
        "Deque (Double-Ended Queue)",
        "Binary Search Tree",
    ]


# --- Question 3: True / False ---


def test_question_3_true_false(quiz_page: Page):
    elements = find_question_elements(quiz_page)
    ctx = scrape_question(elements[2], order=3, capture_screenshot=False)

    assert ctx.question_id == "q3"
    assert ctx.question_type == QuestionType.TRUE_FALSE
    assert ctx.points == 5.0
    assert "HTTP is a stateless" in ctx.prompt
    assert ctx.options == ["True", "False"]


# --- Question 4: Fill in the Blank ---


def test_question_4_fill_in_blank(quiz_page: Page):
    elements = find_question_elements(quiz_page)
    ctx = scrape_question(elements[3], order=4, capture_screenshot=False)

    assert ctx.question_id == "q4"
    assert ctx.question_type == QuestionType.FILL_IN_BLANK
    assert ctx.points == 10.0
    assert "Complete the statement by typing the exact complexity" in ctx.prompt
    assert ctx.options == []


# --- Question 5: Matching ---


def test_question_5_matching(quiz_page: Page):
    elements = find_question_elements(quiz_page)
    ctx = scrape_question(elements[4], order=5, capture_screenshot=False)

    assert ctx.question_id == "q5"
    assert ctx.question_type == QuestionType.MATCHING
    assert ctx.points == 10.0
    assert "Match each network protocol" in ctx.prompt
    assert ctx.matching_prompts == ["HTTP", "HTTPS", "SSH"]
    assert ctx.matching_options == [
        "Port 80",
        "Port 443",
        "Port 22",
        "Port 53 (Distractor)",
    ]


def test_matching_react_portal_combobox(browser: Browser):
    page = browser.new_page()
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>Portal Test</title></head>
    <body>
      <article class="element-card question-card" data-question-id="q_portal" data-question-type="matching">
        <div class="matching-container">
          <div class="matching-row">
            <span class="matching-prompt-text">Protocol A</span>
            <div role="combobox" aria-haspopup="listbox" id="combo-btn">Select match</div>
          </div>
        </div>
      </article>
      <script>
        document.getElementById('combo-btn').addEventListener('click', () => {
          if (!document.getElementById('portal-listbox')) {
            const listbox = document.createElement('div');
            listbox.id = 'portal-listbox';
            listbox.setAttribute('role', 'listbox');
            listbox.innerHTML = `
              <div role="option">Option Alpha</div>
              <div role="option">Option Beta</div>
              <div role="option">Option Gamma</div>
            `;
            document.body.appendChild(listbox);
          }
        });
        document.addEventListener('keydown', (e) => {
          if (e.key === 'Escape') {
            const lb = document.getElementById('portal-listbox');
            if (lb) lb.remove();
          }
        });
      </script>
    </body>
    </html>
    """
    page.goto(f"data:text/html,{html_content}")
    el = page.locator("article").first
    prompts, options = extract_matching_data(el)

    assert prompts == ["Protocol A"]
    assert options == ["Option Alpha", "Option Beta", "Option Gamma"]
    # Verify listbox was closed by Escape key
    assert page.locator('[role="listbox"]').count() == 0
    page.close()


# --- Question 6: Essay / Written Response ---


def test_question_6_essay(quiz_page: Page):
    elements = find_question_elements(quiz_page)
    ctx = scrape_question(elements[5], order=6, capture_screenshot=False)

    assert ctx.question_id == "q6"
    assert ctx.question_type == QuestionType.ESSAY
    assert ctx.points == 5.0
    assert "hash table lookup is average-case O(1)" in ctx.prompt
    assert ctx.options == []


# --- Resilient Heuristics & Error Handling Tests ---


def test_detect_question_type_heuristics(browser: Browser):
    page = browser.new_page()

    # Radio without data-question-type
    page.goto(
        "data:text/html,<article><input type='radio' value='A'><label>Choice A</label></article>"
    )
    el = page.locator("article")
    assert detect_question_type(el) == QuestionType.SINGLE_CHOICE

    # Checkbox without data-question-type
    page.goto(
        "data:text/html,<article><input type='checkbox' value='A'><label>Choice A</label></article>"
    )
    el = page.locator("article")
    assert detect_question_type(el) == QuestionType.MULTIPLE_CHOICE

    # True / False without data-question-type
    page.goto(
        "data:text/html,<article><label class='choice-text'>True</label><input type='radio'><label class='choice-text'>False</label><input type='radio'></article>"
    )
    el = page.locator("article")
    assert detect_question_type(el) == QuestionType.TRUE_FALSE

    # Textarea / Essay without data-question-type
    page.goto("data:text/html,<article><textarea class='essay-textarea'></textarea></article>")
    el = page.locator("article")
    assert detect_question_type(el) == QuestionType.ESSAY

    # Fill in blank without data-question-type
    page.goto(
        "data:text/html,<article><div class='fib-container'><input type='text'></div></article>"
    )
    el = page.locator("article")
    assert detect_question_type(el) == QuestionType.FILL_IN_BLANK

    # Matching without data-question-type
    page.goto(
        "data:text/html,<article><div class='matching-container'><select><option value='1'>Opt 1</option></select></div></article>"
    )
    el = page.locator("article")
    assert detect_question_type(el) == QuestionType.MATCHING

    page.close()


def test_capture_screenshot_error_raises():
    mock_el = MagicMock()
    mock_el.scroll_into_view_if_needed.side_effect = PlaywrightError("Scroll timeout")
    with pytest.raises(ScreenshotCaptureError, match="Failed to capture question screenshot"):
        capture_question_screenshot(mock_el)


def test_scrape_question_screenshot_failure_continues(quiz_page: Page, log_capture):
    elements = find_question_elements(quiz_page)
    scraper = Scraper(capture_screenshots=True)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "qs.scraper.scraper.capture_question_screenshot",
            MagicMock(side_effect=ScreenshotCaptureError("Screenshot failed")),
        )
        ctx = scraper.scrape_question(elements[0], order=1)
        assert ctx.screenshot_bytes is None
        assert ctx.question_id == "q1"

    assert any(
        "Screenshot capture failed for question 'q1', proceeding without screenshot"
        in r.getMessage()
        for r in log_capture
    )
