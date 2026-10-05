"""Interactive demo of Quiz Solver using Blackboard Ultra fixture."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import keyring
from rich.prompt import Prompt

from qs.app.service import QuizSolverService
from qs.browser.session import BrowserSession
from qs.cli.cli import print_banner, print_quiz_summary
from qs.cli.theme import console
from qs.config import (
    API_KEY_ACCOUNT,
    API_SERVICE_NAME,
    DEFAULT_GEMINI_MODEL,
    GOOGLE_API_KEY_MIN_LENGTH,
)
from qs.credentials import Credentials
from qs.llm.google import GeminiService
from qs.logger import setup_logger
from qs.models import (
    EssaySolution,
    FillInBlankSolution,
    MatchingSolution,
    MultipleChoiceSolution,
    SingleChoiceSolution,
    TrueFalseSolution,
)
from qs.scraper.scraper import detect_question_type, extract_question_id

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "tests"
    / "fixtures"
    / "blackboard_quiz.html"
)
FIXTURE_URL = f"file://{FIXTURE_PATH.resolve()}"


def _mock_gemini_solver(
    prompt: str | None = None,
    images: object = None,
    response_schema: object = None,
    system_instruction: object = None,
) -> object:
    """Provide realistic vision reasoning responses for fixture questions when offline."""
    time.sleep(0.3)

    if response_schema == SingleChoiceSolution:
        return SingleChoiceSolution(
            selected_option="Merge Sort",
            confidence=0.98,
            explanation="Merge Sort guarantees O(n log n) worst-case time complexity.",
        )
    if response_schema == MultipleChoiceSolution:
        return MultipleChoiceSolution(
            selected_options=["Queue", "Deque (Double-Ended Queue)"],
            confidence=0.96,
            explanation="Queue operates on FIFO and Deque supports O(1) queue insertion/removal.",
        )
    if response_schema == TrueFalseSolution:
        return TrueFalseSolution(
            value=True,
            confidence=1.0,
            explanation="HTTP is a stateless application-layer protocol.",
        )
    if response_schema == FillInBlankSolution:
        return FillInBlankSolution(
            answers=["log n"],
            confidence=0.92,
            explanation="Binary search repeatedly halves search interval giving O(log n).",
        )
    if response_schema == MatchingSolution:
        return MatchingSolution(
            pairs={
                "HTTP": "Port 80",
                "HTTPS": "Port 443",
                "SSH": "Port 22",
            },
            confidence=0.96,
            explanation="Standard default TCP network port mappings.",
        )
    if response_schema == EssaySolution:
        return EssaySolution(
            response_text=(
                "In-memory hash tables offer O(1) expected lookup time with a good hash function. "
                "However, frequent hash collisions degrade operations to O(n) in the worst case."
            ),
            confidence=0.93,
            explanation="Covers expected O(1) vs worst-case O(n) collision dynamics.",
        )
    raise ValueError(f"Unknown schema: {response_schema}")


def ensure_auth_for_demo() -> str | None:
    """Verify Gemini API key is configured; prompt user interactively if missing."""
    # Check environment variable
    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env_key and len(env_key.strip()) >= GOOGLE_API_KEY_MIN_LENGTH:
        return env_key.strip()

    # Check keyring
    with contextlib.suppress(Exception):
        stored = keyring.get_password(API_SERVICE_NAME, API_KEY_ACCOUNT)
        if stored and len(stored.strip()) >= GOOGLE_API_KEY_MIN_LENGTH:
            return stored.strip()

    # Interactive key prompt if terminal is interactive
    console.print("[primary]Authentication Required[/primary]")
    console.print("Gemini API key is required to solve questions with AI.")
    console.print("[muted]Get a key at https://aistudio.google.com/[/muted]\n")

    if not sys.stdin.isatty():
        console.print("[error]Interactive terminal not available.[/error]")
        console.print("[muted]Set GEMINI_API_KEY or run 'qs auth setup'.[/muted]")
        return None

    try:
        raw_key = Prompt.ask("Enter Gemini API key", password=True)
    except KeyboardInterrupt, EOFError:
        console.print("\n[muted]Cancelled.[/muted]")
        return None

    sanitized = raw_key.strip() if raw_key else ""
    if not sanitized:
        console.print("[error]API key cannot be empty.[/error]")
        return None

    if len(sanitized) < GOOGLE_API_KEY_MIN_LENGTH:
        console.print(
            f"[error]API key must be at least {GOOGLE_API_KEY_MIN_LENGTH} characters.[/error]"
        )
        return None

    try:
        Credentials.set_api_key(sanitized)
        console.print("[success]Gemini API key saved to keyring.[/success]\n")
    except Exception as e:
        console.print(
            f"[warning]Could not save to keyring ({e}). Using for current session.[/warning]\n"
        )

    return sanitized


def _format_solution_answer(sol: object) -> str:
    """Format solution value cleanly for CLI display."""
    if isinstance(sol, SingleChoiceSolution):
        return sol.selected_option
    if isinstance(sol, MultipleChoiceSolution):
        return ", ".join(sol.selected_options)
    if isinstance(sol, TrueFalseSolution):
        return "True" if sol.value else "False"
    if isinstance(sol, FillInBlankSolution):
        return ", ".join(sol.answers)
    if isinstance(sol, MatchingSolution):
        return "; ".join(f"{k} -> {v}" for k, v in sol.pairs.items())
    if isinstance(sol, EssaySolution):
        text = sol.response_text.replace("\n", " ").strip()
        return text[:75] + "..." if len(text) > 75 else text
    return str(sol)


def run_demo(
    mock: bool = False,
    live: bool = True,
    headless: bool = False,
    slowmo: float = 0.1,
    verbose: bool = False,
    path: str | Path | None = None,
) -> int:
    """Run full quiz-solver demo showing all pipeline stages on Blackboard fixture."""
    setup_logger(verbose=verbose)
    print_banner()

    target_fixture = Path(path).resolve() if path else FIXTURE_PATH
    if not target_fixture.exists():
        console.print(f"[error]Fixture file not found: {target_fixture}[/error]")
        return 1

    fixture_url = f"file://{target_fixture.resolve()}"
    use_ai = live and not mock

    console.print("[bold white]Blackboard Assessment Demo[/bold white]")
    console.print(f"[muted]Fixture:[/] {target_fixture.name}")

    gemini_service: GeminiService | MagicMock
    if use_ai:
        api_key = ensure_auth_for_demo()
        if not api_key:
            console.print("[error]Cannot run demo without Gemini API authentication.[/error]")
            console.print("[muted]Pass --mock to run offline demo without API key.[/muted]")
            return 1

        try:
            gemini_service = GeminiService(api_key=api_key)
            console.print(
                f"[success]Authenticated with Gemini API[/success] [muted]({DEFAULT_GEMINI_MODEL})[/muted]\n"
            )
        except Exception as e:
            console.print(f"[error]Failed to initialize Gemini API: {e}[/error]")
            return 1
    else:
        console.print("[muted]Mode:[/] Mock Vision Reasoning (offline)\n")
        gemini_service = MagicMock(spec=GeminiService)
        gemini_service.send.side_effect = _mock_gemini_solver

    # Launch browser
    console.print("[primary]1/4 Browser Session[/primary]")
    console.print(f"Launching Chromium browser ({'headless' if headless else 'visible'})...")
    session = BrowserSession(headless=headless)

    try:
        session.open()
        console.print("Navigating to Blackboard quiz fixture...")
        session.goto(fixture_url)
        console.print("[success]Blackboard quiz page loaded.[/success]\n")

        if not headless:
            time.sleep(0.5)

        # Discovery and Sequential Solving
        console.print("[primary]2/4 Scanning Page DOM[/primary]")
        console.print("Locating Blackboard question containers...")

        total_questions = 0

        def on_stage(stage: str, data: dict[str, Any]) -> None:
            nonlocal total_questions
            if stage == "discovered":
                total_questions = data.get("total", 0)
                cards = data.get("cards", [])
                console.print(
                    f"[success]Found {total_questions} question containers in DOM.[/success]"
                )
                for idx, card in enumerate(cards, start=1):
                    q_id = extract_question_id(card, idx)
                    q_type = detect_question_type(card)
                    console.print(f"  {idx}. [bold]{q_id}[/bold] - {q_type.value}")
                console.print()
                console.print("[primary]3/4 Question Solving Pipeline[/primary]")

            elif stage == "scrape_start":
                order = data.get("order", 1)
                console.print(f"\n[bold white]Question {order}/{total_questions}[/bold white]")
                console.print(
                    "[muted]  [1/3 Scrape][/muted] Extracting question context and element screenshot..."
                )

            elif stage == "scraped":
                prompt = data.get("prompt", "")
                clean_prompt = prompt.replace("\n", " ").strip()
                if len(clean_prompt) > 85:
                    clean_prompt = clean_prompt[:82] + "..."
                console.print(f'[muted]  Prompt:[/] "{clean_prompt}"')

            elif stage == "reason_start":
                console.print(
                    f"[muted]  [2/3 Reason][/muted] Querying Gemini AI ({DEFAULT_GEMINI_MODEL})..."
                )

            elif stage == "reasoned":
                sol = data.get("solution")
                answer_str = _format_solution_answer(sol)
                conf = getattr(sol, "confidence", None)
                conf_str = (
                    f" [muted](Confidence: {int(conf * 100)}%)[/muted]" if conf is not None else ""
                )
                console.print(f"  [success]AI Answer:[/] {answer_str}{conf_str}")
                explanation = getattr(sol, "explanation", None)
                if explanation:
                    clean_exp = explanation.replace("\n", " ").strip()
                    if len(clean_exp) > 95:
                        clean_exp = clean_exp[:92] + "..."
                    console.print(f"[muted]  Reasoning:[/] {clean_exp}")

            elif stage == "fill_start":
                console.print(
                    "[muted]  [3/3 Fill][/muted] Injecting solution into Blackboard DOM..."
                )

            elif stage == "filled":
                dur = data.get("duration", 0.0)
                console.print(f"  [success]Verified:[/] Filled and auto-saved in {dur:.2f}s")
                if slowmo > 0:
                    time.sleep(slowmo)

            elif stage == "error":
                err = data.get("error", "Unknown error")
                console.print(f"  [error]Failed:[/] {err}")

        with QuizSolverService(
            gemini_service=gemini_service,
            capture_screenshots=True,
        ) as solver:
            batch_result = solver.solve_quiz(
                session.page,
                on_stage=on_stage,
            )

        # Results & Review Handoff
        console.print("\n[primary]Summary & Review[/primary]")
        print_quiz_summary(batch_result)

        console.print("[bold green]Quiz processing completed.[/bold green]")
        console.print("All answers are filled in the browser. Quiz was NOT submitted.")
        console.print("Review answers directly in Blackboard before submitting.\n")

        if not headless:
            with contextlib.suppress(KeyboardInterrupt, EOFError):
                input("Press Enter to close browser: ")

        return 0

    except KeyboardInterrupt:
        console.print("\n[muted]Interrupted by user.[/muted]")
        return 0
    except Exception as e:
        console.print(f"\n[error]Demo error: {e}[/error]")
        return 1
    finally:
        session.close()
        console.print("[muted]Browser closed.[/muted]")


def main() -> int:
    """CLI entry point for demo script."""
    parser = argparse.ArgumentParser(description="Quiz Solver Demo on Blackboard Fixture")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock vision solver instead of real Gemini API",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=True,
        help="Use real Gemini API (default)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser without GUI",
    )
    parser.add_argument(
        "--slowmo",
        type=float,
        default=0.1,
        help="Delay between question fills (default: 0.1s)",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=None,
        help="Path to HTML fixture file to test (default: tests/fixtures/blackboard_quiz.html)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()
    return run_demo(
        mock=args.mock,
        live=args.live,
        headless=args.headless,
        slowmo=args.slowmo,
        verbose=args.verbose,
        path=args.path,
    )


if __name__ == "__main__":
    sys.exit(main())
