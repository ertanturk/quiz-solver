"""Semantic DOM injection and form filling for Blackboard Ultra."""

from __future__ import annotations

import contextlib
import re

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Locator

from qs.errors.exceptions import (
    ElementNotInteractableError,
    InjectionFailedError,
    UnsupportedQuestionTypeError,
)
from qs.logger import get_logger
from qs.models import (
    EssaySolution,
    FillInBlankSolution,
    GenericQuestionSolution,
    MatchingSolution,
    MultipleChoiceSolution,
    QuestionSolution,
    QuestionType,
    SingleChoiceSolution,
    TrueFalseSolution,
)

logger = get_logger(__name__)


def _normalize_str(s: str) -> str:
    """Normalize string for case-insensitive, whitespace-insensitive comparison."""
    clean = s.strip()
    stripped_prefix = re.sub(
        r"^(?:option\s+[a-z0-9]+|[a-z0-9][\.\)\s]+)\s*",
        "",
        clean,
        flags=re.IGNORECASE,
    ).strip()
    if stripped_prefix:
        clean = stripped_prefix

    stripped_selected = re.sub(r"\s+selected$", "", clean, flags=re.IGNORECASE).strip()
    if stripped_selected:
        clean = stripped_selected

    return re.sub(r"\s+", " ", clean).strip().lower()


def _click_choice_element(choice_item: Locator, input_element: Locator | None = None) -> None:
    """Click a choice item or its associated input using multi-tier resilient interaction."""
    try:
        if input_element is not None and input_element.is_visible():
            input_element.click()
            return
    except PlaywrightError:
        pass

    try:
        choice_item.scroll_into_view_if_needed()
        choice_item.click()
    except PlaywrightError:
        try:
            choice_item.click(force=True)
        except PlaywrightError as e:
            if input_element is not None:
                try:
                    input_element.check(force=True)
                    return
                except PlaywrightError:
                    pass
            logger.error("Failed to click choice element: %s", e)
            raise ElementNotInteractableError(f"Failed to interact with choice element: {e}") from e


def _fill_single_choice(card: Locator, target_option: str) -> None:
    """Select a single-choice radio option by target string."""
    norm_target = _normalize_str(target_option)

    # Tier 1: Look for .choice-item or label containing target text
    choice_items = card.locator(
        ".choice-item, li[data-choice-value], label.choice-label, label[class*='answerOption'], label.MuiFormControlLabel-root, label"
    ).all()
    # 1a: Exact match
    for item in choice_items:
        data_val = _normalize_str(item.get_attribute("data-choice-value") or "")
        choice_text = ""
        ct = item.locator(".choice-text, [class*='answerOptionText'], .ql-editor").first
        if ct.count() > 0:
            choice_text = _normalize_str(ct.inner_text())
        else:
            choice_text = _normalize_str(item.inner_text())

        if norm_target in (data_val, choice_text):
            radio = item.locator("input[type='radio']").first
            radio_loc = radio if radio.count() > 0 else None
            _click_choice_element(item, radio_loc)
            if radio_loc is not None:
                with contextlib.suppress(Exception):
                    radio_loc.dispatch_event("change")
            return

    # 1b: Resilient containment match (handles verbose LLM outputs)
    best_item: Locator | None = None
    best_len = 0
    for item in choice_items:
        data_val = _normalize_str(item.get_attribute("data-choice-value") or "")
        choice_text = ""
        ct = item.locator(".choice-text, [class*='answerOptionText'], .ql-editor").first
        if ct.count() > 0:
            choice_text = _normalize_str(ct.inner_text())
        else:
            choice_text = _normalize_str(item.inner_text())

        for candidate in (choice_text, data_val):
            if (
                candidate
                and (norm_target in candidate or candidate in norm_target)
                and len(candidate) > best_len
            ):
                best_len = len(candidate)
                best_item = item

    if best_item is not None:
        radio = best_item.locator("input[type='radio']").first
        radio_loc = radio if radio.count() > 0 else None
        _click_choice_element(best_item, radio_loc)
        if radio_loc is not None:
            with contextlib.suppress(Exception):
                radio_loc.dispatch_event("change")
        return

    # 1c: Fallback by option index/letter (e.g. 'A', 'Option A', '1')
    match_letter = re.match(r"^(?:option\s+)?([a-e]|[1-9])$", target_option.strip(), re.IGNORECASE)
    if match_letter and choice_items:
        token = match_letter.group(1).lower()
        idx = ord(token) - ord("a") if token.isalpha() else int(token) - 1
        if 0 <= idx < len(choice_items):
            item = choice_items[idx]
            radio = item.locator("input[type='radio']").first
            radio_loc = radio if radio.count() > 0 else None
            _click_choice_element(item, radio_loc)
            if radio_loc is not None:
                with contextlib.suppress(Exception):
                    radio_loc.dispatch_event("change")
            return

    # Tier 2: Accessible role match
    try:
        radio_by_role = card.get_by_role(
            "radio", name=re.compile(rf"^\s*{re.escape(target_option)}\s*$", re.IGNORECASE)
        ).first
        if radio_by_role.count() > 0:
            radio_by_role.check(force=True)
            radio_by_role.dispatch_event("change")
            return
    except PlaywrightError:
        pass

    # Tier 3: Direct radio input value match
    radios = card.locator("input[type='radio']").all()
    for r in radios:
        val = _normalize_str(r.get_attribute("value") or "")
        if val and (norm_target == val or norm_target.startswith(val) or val.startswith(norm_target)):
            r.check(force=True)
            r.dispatch_event("change")
            return

    raise InjectionFailedError(f"Could not locate radio option matching '{target_option}'")


def _fill_true_false(card: Locator, value: bool) -> None:
    """Select True/False radio option based on boolean value."""
    target_names = ["true", "doğru", "t"] if value else ["false", "yanlış", "f"]

    # Search through available radios and labels
    choice_items = card.locator(
        ".choice-item, label.choice-label, label[class*='answerOption'], label.MuiFormControlLabel-root, label"
    ).all()
    for item in choice_items:
        ct = item.locator(".choice-text, [class*='answerOptionText'], .ql-editor").first
        text = _normalize_str(ct.inner_text()) if ct.count() > 0 else _normalize_str(item.inner_text())
        val = _normalize_str(item.get_attribute("data-choice-value") or "")
        if any(t in (text, val) for t in target_names) or any(t == text or t == val for t in target_names):
            radio = item.locator("input[type='radio']").first
            radio_loc = radio if radio.count() > 0 else None
            _click_choice_element(item, radio_loc)
            if radio_loc is not None:
                with contextlib.suppress(Exception):
                    radio_loc.dispatch_event("change")
            return

    # Fallback to direct radio value check
    radios = card.locator("input[type='radio']").all()
    for r in radios:
        val = _normalize_str(r.get_attribute("value") or "")
        if any(t == val for t in target_names):
            r.check(force=True)
            r.dispatch_event("change")
            return

    expected = "True" if value else "False"
    raise InjectionFailedError(f"Could not locate True/False option for '{expected}'")


def _fill_multiple_choice(card: Locator, selected_options: list[str]) -> None:
    """Check matching checkboxes and uncheck non-matching ones."""
    norm_targets = {_normalize_str(opt) for opt in selected_options}

    choice_items = card.locator(
        ".choice-item, label.choice-label, label[class*='answerOption'], label.MuiFormControlLabel-root, label"
    ).all()
    checkboxes_processed = 0
    if choice_items:
        for item in choice_items:
            data_val = _normalize_str(item.get_attribute("data-choice-value") or "")
            choice_text = ""
            ct = item.locator(".choice-text, [class*='answerOptionText'], .ql-editor").first
            if ct.count() > 0:
                choice_text = _normalize_str(ct.inner_text())
            else:
                choice_text = _normalize_str(item.inner_text())

            should_be_checked = (
                (data_val and data_val in norm_targets)
                or (choice_text and choice_text in norm_targets)
                or any(
                    (data_val and (data_val.startswith(t) or t.startswith(data_val)))
                    or (choice_text and (choice_text.startswith(t) or t.startswith(choice_text)))
                    for t in norm_targets
                )
            )
            cb = item.locator("input[type='checkbox']").first
            if cb.count() > 0:
                checkboxes_processed += 1
                is_currently_checked = cb.is_checked()
                if should_be_checked != is_currently_checked:
                    _click_choice_element(item, cb)
                    with contextlib.suppress(Exception):
                        cb.dispatch_event("change")

        if checkboxes_processed > 0:
            return

    # Fallback: direct checkbox inputs
    checkboxes = card.locator("input[type='checkbox']").all()
    for cb in checkboxes:
        val = _normalize_str(cb.get_attribute("value") or "")
        should_be_checked = val in norm_targets
        is_currently_checked = cb.is_checked()
        if should_be_checked != is_currently_checked:
            if should_be_checked:
                cb.check(force=True)
            else:
                cb.uncheck(force=True)
            with contextlib.suppress(Exception):
                cb.dispatch_event("change")


def _fill_fill_in_blank(card: Locator, answers: list[str]) -> None:
    """Inject answers into text inputs with React synthetic event dispatch sequence."""
    blanks = card.locator("input[type='text'], .fib-input, textarea").all()
    if not blanks:
        raise InjectionFailedError("No text inputs or blanks found for fill_in_blank question")

    for idx, ans in enumerate(answers):
        if idx >= len(blanks):
            break
        blank = blanks[idx]
        blank.scroll_into_view_if_needed()
        blank.focus()
        blank.fill(ans)
        blank.dispatch_event("input")
        blank.dispatch_event("change")
        blank.blur()


def _fill_essay(card: Locator, response_text: str) -> None:
    """Inject essay response into textarea or rich text editor."""
    editor = card.locator(
        "textarea, .essay-textarea, [role='textbox'][aria-multiline='true']"
    ).first
    if editor.count() == 0:
        raise InjectionFailedError("No textarea or editor found for essay question")

    editor.scroll_into_view_if_needed()
    editor.focus()
    editor.fill(response_text)
    editor.dispatch_event("input")
    editor.dispatch_event("change")
    editor.blur()


def _fill_matching(card: Locator, pairs: dict[str, str]) -> None:
    """Match prompts to target dropdown options across rows."""
    rows = card.locator(".matching-row").all()
    page = card.page

    for prompt_key, target_value in pairs.items():
        norm_prompt = _normalize_str(prompt_key)
        norm_target = _normalize_str(target_value)

        # Locate corresponding matching row
        target_row: Locator | None = None
        for r in rows:
            if norm_prompt in _normalize_str(r.inner_text()):
                target_row = r
                break

        if target_row is None:
            logger.warning("Could not find matching row for prompt '%s'", prompt_key)
            continue

        # Option A: Standard <select> dropdown
        select = target_row.locator("select").first
        if select.count() > 0:
            options = select.locator("option").all()
            matched_value: str | None = None
            for opt in options:
                val = opt.get_attribute("value") or ""
                txt = opt.inner_text()
                if norm_target in _normalize_str(txt) or norm_target in _normalize_str(val):
                    matched_value = val
                    break

            if matched_value is not None:
                select.select_option(value=matched_value)
                select.dispatch_event("change")
            else:
                try:
                    select.select_option(label=target_value)
                    select.dispatch_event("change")
                except PlaywrightError as e:
                    logger.warning(
                        "Could not select option '%s' for '%s': %s", target_value, prompt_key, e
                    )
            continue

        # Option B: React Portal / ARIA Combobox
        combobox = target_row.locator('[role="combobox"], button[aria-haspopup="listbox"]').first
        if combobox.count() > 0:
            try:
                combobox.click()
                page.wait_for_timeout(100)
                listbox = page.locator('[role="listbox"]').first
                opt_loc = (
                    listbox.locator('[role="option"]')
                    if listbox.count() > 0
                    else page.locator('[role="option"]')
                )

                found = False
                for opt in opt_loc.all():
                    if norm_target in _normalize_str(opt.inner_text()):
                        opt.click()
                        found = True
                        break

                if not found:
                    page.keyboard.press("Escape")
            except PlaywrightError as e:
                logger.warning("Combobox interaction failed for prompt '%s': %s", prompt_key, e)
                with contextlib.suppress(Exception):
                    page.keyboard.press("Escape")


class Filler:
    """Executes physical UI actions to inject LLM solutions into Blackboard Ultra DOM."""

    def __init__(self, timeout: float = 10000.0) -> None:
        self.timeout = timeout

    def fill(
        self,
        card: Locator,
        solution: QuestionSolution | GenericQuestionSolution,
    ) -> bool:
        """Inject solution into the question card element.

        Args:
            card: Question container Locator.
            solution: Solution object (typed or generic).

        Returns:
            True if injection completed successfully.

        Raises:
            InjectionFailedError: If interaction fails.
            UnsupportedQuestionTypeError: If question type cannot be handled.
        """
        q_type = solution.question_type
        logger.info("Injecting solution for question type: %s", q_type)

        try:
            match q_type:
                case QuestionType.SINGLE_CHOICE:
                    target: str | None = None
                    if isinstance(solution, (SingleChoiceSolution, GenericQuestionSolution)):
                        target = solution.selected_option

                    if not target:
                        raise InjectionFailedError(
                            "Missing selected_option in single_choice solution"
                        )
                    _fill_single_choice(card, target)

                case QuestionType.TRUE_FALSE:
                    val: bool | None = None
                    if isinstance(solution, TrueFalseSolution):
                        val = solution.value
                    elif isinstance(solution, GenericQuestionSolution):
                        val = solution.bool_value

                    if val is None:
                        raise InjectionFailedError("Missing bool value in true_false solution")
                    _fill_true_false(card, val)

                case QuestionType.MULTIPLE_CHOICE:
                    options: list[str] = []
                    if isinstance(solution, (MultipleChoiceSolution, GenericQuestionSolution)):
                        options = solution.selected_options
                    else:
                        raise InjectionFailedError(
                            "Missing selected_options in multiple_choice solution"
                        )

                    _fill_multiple_choice(card, options)

                case QuestionType.FILL_IN_BLANK:
                    answers: list[str] = []
                    if isinstance(solution, FillInBlankSolution):
                        answers = solution.answers
                    elif isinstance(solution, GenericQuestionSolution):
                        answers = solution.fill_blanks
                    else:
                        raise InjectionFailedError("Missing answers in fill_in_blank solution")
                    _fill_fill_in_blank(card, answers)

                case QuestionType.MATCHING:
                    pairs: dict[str, str] = {}
                    if isinstance(solution, MatchingSolution):
                        pairs = solution.pairs
                    elif isinstance(solution, GenericQuestionSolution):
                        pairs = solution.matching_pairs
                    else:
                        raise InjectionFailedError("Missing pairs in matching solution")
                    _fill_matching(card, pairs)

                case QuestionType.ESSAY:
                    text: str | None = None
                    if isinstance(solution, EssaySolution):
                        text = solution.response_text
                    elif isinstance(solution, GenericQuestionSolution):
                        text = solution.essay_text

                    if text is None:
                        raise InjectionFailedError("Missing response_text in essay solution")
                    _fill_essay(card, text)

                case _:
                    raise UnsupportedQuestionTypeError(f"Unsupported question type: {q_type}")

            logger.info("Successfully injected solution for %s", q_type)
            return True

        except InjectionFailedError, UnsupportedQuestionTypeError:
            raise
        except Exception as e:
            logger.error("Failed injecting solution for %s: %s", q_type, e)
            raise InjectionFailedError(f"Failed to inject answer: {e}") from e


# Convenience module-level functions


def fill_question(
    card: Locator,
    solution: QuestionSolution | GenericQuestionSolution,
) -> bool:
    """Inject solution into question element using default Filler."""
    return Filler().fill(card, solution)
