"""Tests for LLM prompts and schema mapping in qs.llm.prompts."""

from __future__ import annotations

import pytest

from qs.llm.prompts import (
    SYSTEM_INSTRUCTION,
    build_question_prompt,
    get_solution_schema,
)
from qs.models import (
    EssaySolution,
    FillInBlankSolution,
    MatchingSolution,
    MultipleChoiceSolution,
    QuestionContext,
    QuestionType,
    SingleChoiceSolution,
    TrueFalseSolution,
)


def test_system_instruction_valid_json_examples():
    """Ensure SYSTEM_INSTRUCTION does not contain illegal JSON comments."""
    assert "//" not in SYSTEM_INSTRUCTION
    assert "true_false" in SYSTEM_INSTRUCTION
    assert '"value": true' in SYSTEM_INSTRUCTION
    assert "rely on the visual screenshot" in SYSTEM_INSTRUCTION


def test_get_solution_schema_mappings():
    assert get_solution_schema(QuestionType.SINGLE_CHOICE) is SingleChoiceSolution
    assert get_solution_schema(QuestionType.MULTIPLE_CHOICE) is MultipleChoiceSolution
    assert get_solution_schema(QuestionType.TRUE_FALSE) is TrueFalseSolution
    assert get_solution_schema(QuestionType.FILL_IN_BLANK) is FillInBlankSolution
    assert get_solution_schema(QuestionType.MATCHING) is MatchingSolution
    assert get_solution_schema(QuestionType.ESSAY) is EssaySolution


def test_get_solution_schema_string_and_case_insensitive():
    assert get_solution_schema("single_choice") is SingleChoiceSolution
    assert get_solution_schema("MULTIPLE_CHOICE") is MultipleChoiceSolution
    assert get_solution_schema("true_false") is TrueFalseSolution


def test_get_solution_schema_invalid_raises():
    with pytest.raises(ValueError, match="Unknown or unsupported question type"):
        get_solution_schema("non_existent_type")


def test_build_question_prompt_single_choice():
    ctx = QuestionContext(
        question_id="q1",
        question_type=QuestionType.SINGLE_CHOICE,
        prompt="Which sort is O(n log n)?",
        points=10.0,
        options=["Bubble Sort", "Merge Sort"],
    )
    prompt = build_question_prompt(ctx)
    assert "Question ID: q1" in prompt
    assert "Points: 10.0" in prompt
    assert "Available Options (Pick exact string(s) from this list):" in prompt
    assert "1. Bubble Sort" in prompt
    assert "2. Merge Sort" in prompt
    assert "rely on screenshot as ground truth" in prompt


def test_build_question_prompt_true_false():
    ctx = QuestionContext(
        question_id="q3",
        question_type=QuestionType.TRUE_FALSE,
        prompt="HTTP is stateless.",
        points=5.0,
        options=["True", "False"],
    )
    prompt = build_question_prompt(ctx)
    assert "return boolean true or false in the 'value' field" in prompt
    assert "Available Labels in UI: True, False" in prompt
    assert "Pick exact string(s) from this list" not in prompt


def test_build_question_prompt_matching():
    ctx = QuestionContext(
        question_id="q5",
        question_type=QuestionType.MATCHING,
        prompt="Match protocols with ports",
        points=10.0,
        matching_prompts=["HTTP", "HTTPS"],
        matching_options=["Port 80", "Port 443"],
    )
    prompt = build_question_prompt(ctx)
    assert "Matching Prompts (Keys to match):" in prompt
    assert "- HTTP" in prompt
    assert "Available Options (Values to pick from):" in prompt
    assert "- Port 80" in prompt


def test_build_question_prompt_fill_in_blank():
    ctx = QuestionContext(
        question_id="q4",
        question_type=QuestionType.FILL_IN_BLANK,
        prompt="Binary search is O(...)",
    )
    prompt = build_question_prompt(ctx)
    assert "ordered list of strings to fill in the blanks" in prompt


def test_build_question_prompt_essay():
    ctx = QuestionContext(
        question_id="q6",
        question_type=QuestionType.ESSAY,
        prompt="Explain hash table lookup complexity",
    )
    prompt = build_question_prompt(ctx)
    assert "Write a comprehensive, accurate response" in prompt
