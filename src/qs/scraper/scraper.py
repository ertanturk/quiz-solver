"""DOM container boundary detection and screenshot extraction for Blackboard Ultra."""

from __future__ import annotations

import contextlib
import re

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Locator, Page

from qs.errors.exceptions import QuestionNotFoundError, ScreenshotCaptureError
from qs.logger import get_logger
from qs.models import QuestionContext, QuestionType

logger = get_logger(__name__)

# Candidate selectors for Blackboard question containers in order of precision
QUESTION_SELECTORS: list[str] = [
    '[data-analytics-id="question-container"]',
    "article.question-card",
    "div.question-card",
    ".element-card",
    '[role="region"][aria-label*="Question"]',
    'div[class*="question"]',
]

POINTS_REGEX = re.compile(r"(\d+(?:\.\d+)?)\s*(?:points)", re.IGNORECASE)


def find_question_elements(page: Page, wait_timeout_ms: int = 0) -> list[Locator]:
    """Locate all question container elements on the active Blackboard page.

    Args:
        page: Active Playwright page.
        wait_timeout_ms: Timeout in milliseconds to poll for questions if not yet mounted.

    Returns:
        List of question element Locators.

    Raises:
        QuestionNotFoundError: If no question containers are found within the timeout.
    """
    for selector in QUESTION_SELECTORS:
        locator = page.locator(selector)
        count = locator.count()
        if count > 0:
            logger.debug("Located %d question container(s) using '%s'", count, selector)
            return locator.all()

    if wait_timeout_ms > 0:
        combined = ", ".join(QUESTION_SELECTORS)
        logger.debug(
            "Waiting up to %dms for question containers matching '%s'",
            wait_timeout_ms,
            combined,
        )
        try:
            page.wait_for_selector(combined, timeout=wait_timeout_ms, state="attached")
            for selector in QUESTION_SELECTORS:
                locator = page.locator(selector)
                count = locator.count()
                if count > 0:
                    logger.debug(
                        "Located %d question container(s) after wait using '%s'",
                        count,
                        selector,
                    )
                    return locator.all()
        except PlaywrightError:
            pass

    logger.warning("No question containers found on the active page")
    raise QuestionNotFoundError("No question containers found on the active page")


def extract_question_id(element: Locator, order: int) -> str:
    """Extract or generate unique question ID."""
    for attr in ("data-question-id", "data-testid", "id"):
        try:
            val = element.get_attribute(attr)
            if isinstance(val, str) and val.strip():
                return val.strip()
        except PlaywrightError:
            continue
    return f"question_{order}"


def extract_points(element: Locator) -> float | None:
    """Extract point value assigned to the question."""
    try:
        badge = element.locator(".points-badge, [class*='points']").first
        if badge.count() > 0:
            text = badge.inner_text().strip()
            match = POINTS_REGEX.search(text)
            if match:
                return float(match.group(1))

        # Fallback to whole card text for points indicator
        header = element.locator(".question-card-header").first
        if header.count() > 0:
            text = header.inner_text().strip()
            match = POINTS_REGEX.search(text)
            if match:
                return float(match.group(1))
    except PlaywrightError as e:
        logger.debug("Failed to extract points: %s", e)

    return None


def detect_question_type(element: Locator) -> QuestionType:
    """Classify the question type using data attributes and DOM inspection.

    Args:
        element: Question container Locator.

    Returns:
        Detected QuestionType enum value.
    """
    # Direct attribute check
    try:
        raw_type = element.get_attribute("data-question-type")
        if raw_type:
            normalized = raw_type.strip().lower()
            for q_type in QuestionType:
                if normalized == q_type.value:
                    return q_type
    except PlaywrightError:
        pass

    # Heuristic DOM inspection
    try:
        # Matching: contains select dropdowns or matching container
        if element.locator(".matching-container, .matching-row, select").count() > 0:
            return QuestionType.MATCHING

        # Essay: contains textarea or rich text editor
        if (
            element.locator(
                "textarea, [role='textbox'][aria-multiline='true'], .essay-textarea, .essay-editor-container"
            ).count()
            > 0
        ):
            return QuestionType.ESSAY

        # Fill in blank: contains text input for blanks
        if element.locator("input[type='text'], .fib-input, .fib-container").count() > 0:
            return QuestionType.FILL_IN_BLANK

        # Checkboxes: multi-select
        if element.locator("input[type='checkbox'], [role='checkbox']").count() > 0:
            return QuestionType.MULTIPLE_CHOICE

        # Radio buttons: single choice or true/false
        if element.locator("input[type='radio'], [role='radio']").count() > 0:
            labels = [
                txt.strip().lower()
                for txt in element.locator(".choice-text, label").all_inner_texts()
                if txt.strip()
            ]
            tf_values = {"true", "false", "doğru", "yanlış"}
            if labels and all(lbl in tf_values for lbl in labels):
                return QuestionType.TRUE_FALSE
            return QuestionType.SINGLE_CHOICE
    except PlaywrightError as e:
        logger.debug("Heuristic question type detection error: %s", e)

    return QuestionType.SINGLE_CHOICE


def extract_prompt(element: Locator) -> str:
    """Extract question text prompt."""
    try:
        prompt_el = element.locator(".question-prompt, [id$='-prompt'], [class*='prompt']").first
        if prompt_el.count() > 0:
            return prompt_el.inner_text().strip()

        # Fallback to header or aria-label
        header = element.locator(".question-card-header").first
        if header.count() > 0:
            return header.inner_text().strip()

        aria = element.get_attribute("aria-label")
        if aria:
            return aria.strip()
    except PlaywrightError as e:
        logger.debug("Failed to extract prompt: %s", e)

    return ""


def extract_options(element: Locator, question_type: QuestionType) -> list[str]:
    """Extract answer choices for single-choice, multiple-choice, or true-false questions."""
    if question_type not in (
        QuestionType.SINGLE_CHOICE,
        QuestionType.MULTIPLE_CHOICE,
        QuestionType.TRUE_FALSE,
    ):
        return []

    try:
        # Prefer .choice-text to exclude letter badges like 'A', 'B'
        choice_elements = element.locator(".choice-text").all()
        if choice_elements:
            texts = [el.inner_text().strip() for el in choice_elements]
            filtered = [t for t in texts if t]
            if filtered:
                return filtered

        # Fallback to label inner text, stripping leading letters (e.g. 'A.', 'B)')
        label_elements = element.locator(".choice-label, label").all()
        labels = [el.inner_text().strip() for el in label_elements]
        cleaned = []
        for text in labels:
            stripped = re.sub(r"^[A-Za-z0-9][\.\)\s]+\s*", "", text).strip()
            if stripped:
                cleaned.append(stripped)
            elif text:
                cleaned.append(text)
        if cleaned:
            return cleaned

        # Fallback to input values
        input_elements = element.locator("input[type='radio'], input[type='checkbox']").all()
        values = [inp.get_attribute("value") for inp in input_elements]
        return [v.strip() for v in values if v and v.strip()]
    except PlaywrightError as e:
        logger.debug("Failed to extract options: %s", e)
        return []


def extract_matching_data(element: Locator) -> tuple[list[str], list[str]]:
    """Extract matching prompts and available target dropdown options."""
    prompts: list[str] = []
    options: list[str] = []

    try:
        # Extract matching prompt keys (prefer inner text span to avoid numeric badge)
        prompt_elements = element.locator(".matching-prompt-text").all()
        if not prompt_elements:
            prompt_elements = element.locator(".matching-prompt-box").all()

        for p_el in prompt_elements:
            text = p_el.inner_text().strip()
            text = re.sub(r"^\d+[\.\)\s]*\n?", "", text).strip()
            if text:
                prompts.append(text)

        # Extract options from standard <select> element if present
        first_select = element.locator("select").first
        if first_select.count() > 0:
            for opt in first_select.locator("option").all():
                val = opt.get_attribute("value") or ""
                text = opt.inner_text().strip()
                if val and text and "select" not in text.lower():
                    options.append(text)

        # Fallback for React Portals / ARIA combobox (MUI / Blackboard Ultra)
        if not options:
            combobox = element.locator('[role="combobox"], button[aria-haspopup="listbox"]').first
            if combobox.count() > 0:
                page = element.page
                try:
                    combobox.click()
                    page.wait_for_timeout(100)
                    listbox = page.locator('[role="listbox"]').first
                    opt_locator = (
                        listbox.locator('[role="option"]')
                        if listbox.count() > 0
                        else page.locator('[role="option"]')
                    )
                    for opt in opt_locator.all():
                        text = opt.inner_text().strip()
                        if text and "select" not in text.lower() and text not in options:
                            options.append(text)
                except PlaywrightError as e:
                    logger.debug("Failed opening combobox to read options: %s", e)
                finally:
                    with contextlib.suppress(Exception):
                        page.keyboard.press("Escape")
                        page.wait_for_timeout(50)

    except PlaywrightError as e:
        logger.debug("Failed to extract matching data: %s", e)

    return prompts, options


def capture_question_screenshot(element: Locator, timeout: float = 10000.0) -> bytes:
    """Scroll question element into view and capture bounded PNG screenshot."""
    try:
        element.scroll_into_view_if_needed(timeout=timeout)
        return element.screenshot(type="png", timeout=timeout)
    except PlaywrightError as e:
        logger.error("Failed to capture question screenshot: %s", e)
        raise ScreenshotCaptureError(f"Failed to capture question screenshot: {e}") from e


class Scraper:
    """Scraper engine for extracting questions from Blackboard Ultra pages."""

    def __init__(
        self,
        capture_screenshots: bool = True,
        timeout: float = 10000.0,
    ) -> None:
        """Initialize scraper settings.

        Args:
            capture_screenshots: Whether to capture element screenshots.
            timeout: Timeout in milliseconds for Playwright element actions.
        """
        self.capture_screenshots = capture_screenshots
        self.timeout = timeout

    def scrape_question(
        self,
        element: Locator,
        order: int,
        capture_screenshot: bool | None = None,
    ) -> QuestionContext:
        """Extract structured QuestionContext from a single question container.

        Args:
            element: Locator for question container.
            order: 1-indexed order of question on page.
            capture_screenshot: Override instance screenshot capture flag.

        Returns:
            Populated QuestionContext instance.
        """
        q_id = extract_question_id(element, order)
        q_type = detect_question_type(element)
        prompt = extract_prompt(element)
        points = extract_points(element)

        options: list[str] = []
        matching_prompts: list[str] = []
        matching_options: list[str] = []

        if q_type == QuestionType.MATCHING:
            matching_prompts, matching_options = extract_matching_data(element)
        else:
            options = extract_options(element, q_type)

        should_capture = (
            capture_screenshot if capture_screenshot is not None else self.capture_screenshots
        )
        screenshot_bytes: bytes | None = None
        if should_capture:
            try:
                screenshot_bytes = capture_question_screenshot(element, timeout=self.timeout)
            except ScreenshotCaptureError:
                logger.warning(
                    "Screenshot capture failed for question '%s', proceeding without screenshot",
                    q_id,
                )

        logger.debug(
            "Scraped question #%d [id=%s, type=%s, options=%d, points=%s]",
            order,
            q_id,
            q_type,
            len(options),
            points,
        )

        return QuestionContext(
            question_id=q_id,
            question_type=q_type,
            prompt=prompt,
            points=points,
            options=options,
            matching_prompts=matching_prompts,
            matching_options=matching_options,
            order=order,
            screenshot_bytes=screenshot_bytes,
        )

    def scrape_quiz(
        self,
        page: Page,
        capture_screenshots: bool | None = None,
    ) -> list[QuestionContext]:
        """Scrape all questions from the active quiz page.

        Args:
            page: Active Playwright Page.
            capture_screenshots: Override screenshot capture flag.

        Returns:
            List of QuestionContext objects.

        Raises:
            QuestionNotFoundError: If no question elements are found.
        """
        logger.info("Scanning page for quiz question containers...")
        elements = find_question_elements(page)
        logger.info("Found %d question container(s)", len(elements))

        contexts: list[QuestionContext] = []
        for index, el in enumerate(elements, start=1):
            try:
                ctx = self.scrape_question(
                    element=el,
                    order=index,
                    capture_screenshot=capture_screenshots,
                )
                contexts.append(ctx)
            except Exception as e:
                logger.error("Failed to scrape question container #%d: %s", index, e)

        logger.info("Successfully scraped %d/%d questions", len(contexts), len(elements))
        return contexts


# Module-level convenience functions


def scrape_quiz(page: Page, capture_screenshots: bool = True) -> list[QuestionContext]:
    """Scrape all questions from page using default Scraper."""
    return Scraper(capture_screenshots=capture_screenshots).scrape_quiz(page)


def scrape_question(
    element: Locator,
    order: int = 1,
    capture_screenshot: bool = True,
) -> QuestionContext:
    """Scrape single question element using default Scraper."""
    return Scraper(capture_screenshots=capture_screenshot).scrape_question(
        element=element,
        order=order,
    )
