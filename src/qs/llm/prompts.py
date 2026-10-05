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
1.  **Exact Text Matching Only:** For single_choice and multiple_choice, you MUST return ONLY the exact option text as listed in the available options (e.g. "Option Text"). Do NOT include thoughts, explanations, prefixes, numbers, or commentary in "selected_option" or "selected_options". Keep reasoning strictly in the "explanation" field.
2.  **True/False Boolean Value:** For true_false questions, output the answer under the "value" key as a JSON boolean (true or false), not a string.
3.  **Fill in the Blank:** For fill_in_blank questions, provide an array containing the exact concise answer string for each blank (e.g. ["concise answer"]). Do NOT list synonyms, alternatives, or repeating variations.
4.  **Visual Ground Truth:** If math formulas, code formatting, or diagrams in extracted text are incomplete, rely on the visual screenshot as ground truth.
5.  **No Markdown Formatting:** You must output pure, raw JSON matching the schema without markdown blocks.
6.  **Required Fields:** Every response must include a "confidence" score (0.0 to 1.0) and a brief "explanation" (keep ultra-short, 1-5 words max, e.g. "Standard definition", to maximize generation speed).
7.  **For the essay question:** Write a clear, accurate, and well-structured response in the "response_text" field (around 2-3 sentences, 50-100 words) in simple student English. Output ONLY the direct answer text. Do NOT include conversational filler, greetings, pleasantries, or closing sign-offs (e.g. NEVER write 'I hope this helps' or 'Let me know if you have questions').

8.  **For matching questions:** You MUST provide a matched pair for EVERY item listed under matching prompts.

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
    "answers": ["exact answer"],
    "confidence": 0.85,
    "explanation": "Brief explanation."
}

**5. matching**
{
    "question_type": "matching",
    "pairs": [
        {"prompt": "Item A", "option": "Option 1"},
        {"prompt": "Item B", "option": "Option 2"}
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

BATCH_SYSTEM_INSTRUCTION = """\
You are an expert, highly accurate automated exam-solving AI. Your task is to visually analyze screenshots of Blackboard Ultra quiz questions, read the provided semantic context (extracted text), and output solutions for ALL questions strictly as a JSON object matching the BatchSolution schema.

### CRITICAL RULES:
1.  **Exact Text Matching Only:** For single_choice and multiple_choice, you MUST return ONLY the exact option text verbatim as listed in the available options under "selected_option" or "selected_options". Do NOT include numbers (e.g. '1.', '2.'), prefixes, or commentary. Keep reasoning strictly in the "explanation" field (ultra-short, 1-5 words max, to maximize generation speed).
2.  **True/False Boolean Value:** For true_false questions, output the answer under "bool_value" as a JSON boolean (true or false).
3.  **Fill in the Blank:** For fill_in_blank questions, provide an array under "fill_blanks" containing the exact concise answer string for each blank.
4.  **Matching Questions:** Output "matching_pairs" as a list of {"prompt": "...", "option": "..."} pairs for EVERY item listed under matching prompts.
5.  **Essay Questions:** In "essay_text", write a clear, concise, and well-structured response (around 2-3 sentences, 50-100 words). Output ONLY direct answer text. Do NOT include greetings, pleasantries, or closing sign-offs.
6.  **Visual Ground Truth:** Rely on the corresponding screenshot as ground truth if text is incomplete or ambiguous.
7.  **All Questions Required:** The "solutions" array MUST contain an entry for EVERY question_id provided in the prompt.
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
        lines.append(
            "CRITICAL: Return ONLY the exact option text verbatim. Do NOT include numbers (e.g. '1.', '2.'), prefixes, or reasoning in selected_option or selected_options."
        )

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
        lines.append(
            "CRITICAL: Provide a matched pair for EVERY prompt listed under Matching Prompts."
        )

    elif context.question_type == QuestionType.FILL_IN_BLANK:
        lines.append(
            "Instructions: Provide an ordered list of strings to fill in the blanks shown in the screenshot (e.g. ['log n']). Do NOT repeat or list multiple alternative synonyms for the same blank."
        )

    elif context.question_type == QuestionType.ESSAY:
        lines.append(
            "Instructions: Write a comprehensive, accurate response based on the question prompt (around 2-3 sentences, 50-100 words). Output ONLY the direct answer text. Do NOT include greetings, pleasantries, or closing sign-offs."
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


def build_batch_question_prompt(
    contexts: list[QuestionContext],
) -> tuple[str, list[bytes]]:
    """Build multimodal prompt combining multiple questions into a single batch request."""
    images: list[bytes] = []
    lines = [
        f"You are given {len(contexts)} question(s) from a quiz to solve together.",
        "Attached screenshots correspond sequentially to each question:",
    ]

    for idx, ctx in enumerate(contexts, start=1):
        if ctx.screenshot_bytes:
            images.append(ctx.screenshot_bytes)
            lines.append(f"- Question #{idx} [ID: {ctx.question_id}] -> Screenshot #{len(images)}")
        else:
            lines.append(f"- Question #{idx} [ID: {ctx.question_id}] -> (No screenshot attached)")

    lines.append("")
    lines.append("=" * 40)

    for idx, ctx in enumerate(contexts, start=1):
        lines.append(
            f"### QUESTION #{idx} [ID: {ctx.question_id}, TYPE: {ctx.question_type.value}]"
        )
        lines.append(build_question_prompt(ctx))
        lines.append("-" * 30)

    lines.append("")
    lines.append("### REQUIRED BATCH OUTPUT:")
    lines.append(
        "Return a single JSON object matching the BatchSolution schema with 'solutions' containing an entry for every question_id."
    )

    return "\n".join(lines), images
