"""System prompts and dynamic prompt builders for the Gemini Vision LLM."""

from __future__ import annotations

from pydantic import BaseModel

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

# The core identity and strict rules for the LLM
SYSTEM_INSTRUCTION = """\
You are an expert, highly accurate automated exam-solving AI. Your task is to visually analyze a screenshot of a Blackboard Ultra quiz question, read the provided semantic context (extracted text), and output the correct answer strictly as a JSON object.

### CRITICAL RULES:
1.  **Exact Text Matching Only:** For single_choice and multiple_choice, you MUST return ONLY the exact option text as listed in the available options (e.g. "Merge Sort"). Do NOT include thoughts, explanations, prefixes, or commentary in "selected_option" or "selected_options". Keep reasoning strictly in the "explanation" field.
2.  **True/False Boolean Value:** For true_false questions, output the answer under the "value" key as a JSON boolean (true or false), not a string.
3.  **Fill in the Blank:** For fill_in_blank questions, provide an array containing the exact concise answer string for each blank (e.g. ["log n"]). Do NOT list synonyms, alternatives, or repeating variations.
4.  **Visual Ground Truth:** If math formulas, code formatting, or diagrams in extracted text are incomplete, rely on the visual screenshot as ground truth.
5.  **No Markdown Formatting:** You must output pure, raw JSON matching the schema without markdown blocks.
6.  **Required Fields:** Every response must include a "confidence" score (0.0 to 1.0) and a brief "explanation" justifying your reasoning.
7.  **For the essay question:** Briefly summarize the answer in the "selected_option" field. Use a2 level English and make sure that the summary is concise and written like a human student.

### EXPECTED JSON SCHEMAS & EXAMPLES:
Depending on the "Question Type" provided in the prompt, your JSON output must perfectly match one of the following schemas:

**1. single_choice**
{
    "question_type": "single_choice",
    "selected_option": "Exact string of the correct choice",
    "confidence": 0.95,
    "explanation": "Brief 1 sentence explanation of why this is correct."
}

**2. multiple_choice**
{
    "question_type": "multiple_choice",
    "selected_options": ["Exact string 1", "Exact string 2"],
    "confidence": 0.90,
    "explanation": "Brief explanation."
}

**3. true_false**
{
    "question_type": "true_false",
    "value": true,
    "confidence": 1.0,
    "explanation": "Brief explanation."
}

**4. fill_in_blank**
{
    "question_type": "fill_in_blank",
    "answers": ["log n"],
    "confidence": 0.85,
    "explanation": "Brief explanation."
}

**5. matching**
{
    "question_type": "matching",
    "pairs": [
        {"prompt": "HTTP", "option": "Port 80"},
        {"prompt": "HTTPS", "option": "Port 443"}
    ],
    "confidence": 0.90,
    "explanation": "Brief explanation."
}

**6. essay**
{
    "question_type": "essay",
    "response_text": "Your complete, well-reasoned written response or essay.",
    "confidence": 0.95,
    "explanation": "Brief context on how the essay was structured."
}
"""


def get_solution_schema(question_type: QuestionType | str) -> type[BaseModel]:
    """Return the corresponding Pydantic solution schema class for a question type."""
    normalized = (
        question_type.value
        if isinstance(question_type, QuestionType)
        else str(question_type).lower()
    )
    match normalized:
        case QuestionType.SINGLE_CHOICE.value:
            return SingleChoiceSolution
        case QuestionType.MULTIPLE_CHOICE.value:
            return MultipleChoiceSolution
        case QuestionType.TRUE_FALSE.value:
            return TrueFalseSolution
        case QuestionType.FILL_IN_BLANK.value:
            return FillInBlankSolution
        case QuestionType.MATCHING.value:
            return MatchingSolution
        case QuestionType.ESSAY.value:
            return EssaySolution
        case _:
            raise ValueError(f"Unknown or unsupported question type: {question_type}")


def build_question_prompt(context: QuestionContext) -> str:
    """Build a dynamic textual prompt providing extracted DOM context to supplement the screenshot."""
    lines = [
        "### QUESTION IDENTIFICATION",
        f"- Question ID: {context.question_id}",
        f"- Question Type: {context.question_type.value}",
        f"- Points: {context.points if context.points is not None else 'Unknown'}",
        "",
        "### EXTRACTED TEXT CONTEXT",
        f"Prompt: {context.prompt if context.prompt else '(See screenshot)'}",
    ]

    # Add available options depending on question type
    if context.question_type in (
        QuestionType.SINGLE_CHOICE,
        QuestionType.MULTIPLE_CHOICE,
    ):
        lines.append("Available Options (Pick exact string(s) from this list):")
        for i, opt in enumerate(context.options, 1):
            lines.append(f"  {i}. {opt}")

    elif context.question_type == QuestionType.TRUE_FALSE:
        lines.append(
            "Instructions: Evaluate the statement and return boolean true or false in the 'value' field."
        )
        if context.options:
            lines.append(f"Available Labels in UI: {', '.join(context.options)}")

    elif context.question_type == QuestionType.MATCHING:
        lines.append("Matching Prompts (Keys to match):")
        for p in context.matching_prompts:
            lines.append(f"  - {p}")

        lines.append("")
        lines.append("Available Options (Values to pick from):")
        for o in context.matching_options:
            lines.append(f"  - {o}")

    elif context.question_type == QuestionType.FILL_IN_BLANK:
        lines.append(
            "Instructions: Provide an ordered list of strings to fill in the blanks shown in the screenshot (e.g. ['log n']). Do NOT repeat or list multiple alternative synonyms for the same blank."
        )

    elif context.question_type == QuestionType.ESSAY:
        lines.append(
            "Instructions: Write a comprehensive, accurate response based on the question prompt."
        )

    lines.extend(
        [
            "",
            "### INSTRUCTIONS",
            "1. Analyze the attached screenshot visually.",
            "2. If math formulas (LaTeX/MathJax), code formatting, or diagrams in extracted text are ambiguous or incomplete, rely on screenshot as ground truth.",
            "3. Cross-reference visual data with the 'EXTRACTED TEXT CONTEXT' above.",
            f"4. Return the exact JSON schema required for '{context.question_type.value}'.",
        ]
    )

    return "\n".join(lines)
