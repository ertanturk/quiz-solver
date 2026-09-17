"""Data contract models and Pydantic schemas for Quiz Solver."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


class QuestionType(StrEnum):
    """Supported Blackboard Ultra question types."""

    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    FILL_IN_BLANK = "fill_in_blank"
    MATCHING = "matching"
    ESSAY = "essay"


# Scraper Models


class QuestionContext(BaseModel):
    """Context extracted from the Blackboard DOM and screenshots for a question."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    question_id: str = Field(description="Unique identifier for the question container")
    question_type: QuestionType = Field(description="Detected question classification")
    prompt: str = Field(description="Textual prompt or instructions for the question")
    points: float | None = Field(default=None, description="Point value assigned to question")
    options: list[str] = Field(default_factory=list, description="Available answer choices")
    matching_prompts: list[str] = Field(
        default_factory=list, description="Prompts to be matched (for matching type)"
    )
    matching_options: list[str] = Field(
        default_factory=list, description="Target dropdown options (for matching type)"
    )
    order: int = Field(default=0, description="Sequential question index on the page")
    screenshot_bytes: bytes | None = Field(
        default=None, description="Bounded PNG screenshot of question container"
    )


# Discriminated LLM Solution Models


class BaseSolution(BaseModel):
    """Base metadata for all question solution schemas."""

    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0"
    )
    explanation: str = Field(default="", description="Reasoning or justification for chosen answer")


class SingleChoiceSolution(BaseSolution):
    """Solution for single-choice multiple choice questions (radio)."""

    question_type: Literal[QuestionType.SINGLE_CHOICE] = QuestionType.SINGLE_CHOICE
    selected_option: str = Field(description="Exact string of the single correct choice to select")


class MultipleChoiceSolution(BaseSolution):
    """Solution for multi-select multiple choice questions (checkboxes)."""

    question_type: Literal[QuestionType.MULTIPLE_CHOICE] = QuestionType.MULTIPLE_CHOICE
    selected_options: list[str] = Field(
        description="List of exact strings for all correct choices to select"
    )


class TrueFalseSolution(BaseSolution):
    """Solution for true/false questions."""

    question_type: Literal[QuestionType.TRUE_FALSE] = QuestionType.TRUE_FALSE
    value: bool = Field(description="True or False answer")


class FillInBlankSolution(BaseSolution):
    """Solution for fill-in-the-blank text input questions."""

    question_type: Literal[QuestionType.FILL_IN_BLANK] = QuestionType.FILL_IN_BLANK
    answers: list[str] = Field(
        description="Ordered list of string answers corresponding to each blank"
    )


class MatchingPair(BaseModel):
    """Single prompt-to-option matching item."""

    prompt: str = Field(description="Prompt key or left-side item to match")
    option: str = Field(description="Target dropdown option or right-side value")


class MatchingSolution(BaseSolution):
    """Solution for matching questions mapping prompts to target options."""

    question_type: Literal[QuestionType.MATCHING] = QuestionType.MATCHING
    pairs: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of each matching prompt to its corresponding answer option",
    )

    @field_validator("pairs", mode="before")
    @classmethod
    def _coerce_pairs(cls, v: Any) -> dict[str, str]:
        if isinstance(v, list):
            result: dict[str, str] = {}
            for item in v:
                if isinstance(item, dict):
                    p = item.get("prompt", "")
                    o = item.get("option", "")
                    if p:
                        result[str(p)] = str(o)
                elif hasattr(item, "prompt") and hasattr(item, "option"):
                    result[str(item.prompt)] = str(item.option)
            return result
        if isinstance(v, dict):
            return {str(k): str(val) for k, val in v.items()}
        return v


class MatchingLLMSchema(BaseSolution):
    """Schema used specifically for Gemini API structured output (avoids additionalProperties restriction)."""

    question_type: Literal[QuestionType.MATCHING] = QuestionType.MATCHING
    pairs: list[MatchingPair] = Field(
        default_factory=list,
        description="List of matched pairs mapping each prompt to its target option",
    )


class EssaySolution(BaseSolution):
    """Solution for essay or written response questions."""

    question_type: Literal[QuestionType.ESSAY] = QuestionType.ESSAY
    response_text: str = Field(description="Complete written response or essay answer")


# Discriminated union of typed solutions
QuestionSolution = Annotated[
    SingleChoiceSolution
    | MultipleChoiceSolution
    | TrueFalseSolution
    | FillInBlankSolution
    | MatchingSolution
    | EssaySolution,
    Field(discriminator="question_type"),
]

solution_adapter: TypeAdapter[QuestionSolution] = TypeAdapter(QuestionSolution)


def parse_question_solution(data: str | dict[str, Any] | Any) -> QuestionSolution:
    """Parse raw dict, JSON string, or object into a typed QuestionSolution."""
    if isinstance(data, str):
        return solution_adapter.validate_json(data)
    if isinstance(data, dict):
        return solution_adapter.validate_python(data)
    return solution_adapter.validate_python(data)


class GenericQuestionSolution(BaseModel):
    """Unified solution model containing fields for any question type."""

    question_id: str = Field(default="", description="Question ID reference")
    question_type: QuestionType = Field(
        default=QuestionType.SINGLE_CHOICE, description="Question type"
    )
    selected_option: str | None = Field(default=None, description="Single-choice selection")
    selected_options: list[str] = Field(default_factory=list, description="Multi-select selections")
    bool_value: bool | None = Field(default=None, description="True/False value")
    fill_blanks: list[str] = Field(default_factory=list, description="Fill-in-the-blank answers")
    matching_pairs: dict[str, str] = Field(
        default_factory=dict, description="Matching question prompt->option pairs"
    )
    essay_text: str | None = Field(default=None, description="Essay written text")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    explanation: str = Field(default="")


# Execution & Batch Results


class QuestionExecutionResult(BaseModel):
    """Result of processing and filling a single question."""

    question_id: str = Field(description="ID of processed question")
    question_type: QuestionType = Field(description="Question type")
    prompt: str = Field(description="Question prompt")
    solution: QuestionSolution | GenericQuestionSolution | None = Field(
        default=None, description="Generated solution"
    )
    filled: bool = Field(default=False, description="Whether answer was injected into the browser")
    error: str | None = Field(default=None, description="Error message if processing failed")
    duration_seconds: float = Field(default=0.0, description="Processing time for this question")


class QuizBatchResult(BaseModel):
    """Summary review metrics and results for an entire quiz session."""

    total_questions: int = Field(default=0, description="Total questions identified")
    successful_fills: int = Field(default=0, description="Questions successfully filled")
    failed_fills: int = Field(default=0, description="Questions that failed to fill or resolve")
    results: list[QuestionExecutionResult] = Field(
        default_factory=list, description="Individual results for each question"
    )
    execution_time_seconds: float = Field(
        default=0.0, description="Total execution duration in seconds"
    )
    status: str = Field(
        default="completed", description="Session status ('completed', 'partial', 'failed')"
    )
