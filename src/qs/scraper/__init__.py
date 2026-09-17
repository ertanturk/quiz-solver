"""DOM scraper module for Blackboard Ultra quiz questions."""

from qs.scraper.scraper import (
    QUESTION_SELECTORS,
    Scraper,
    capture_question_screenshot,
    detect_question_type,
    extract_matching_data,
    extract_options,
    extract_points,
    extract_prompt,
    extract_question_id,
    find_question_elements,
    scrape_question,
    scrape_quiz,
)

__all__ = [
    "QUESTION_SELECTORS",
    "Scraper",
    "capture_question_screenshot",
    "detect_question_type",
    "extract_matching_data",
    "extract_options",
    "extract_points",
    "extract_prompt",
    "extract_question_id",
    "find_question_elements",
    "scrape_question",
    "scrape_quiz",
]
