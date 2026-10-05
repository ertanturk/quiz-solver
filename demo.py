"""Demo runner with synchronized Playwright browser recording."""

from __future__ import annotations

import contextlib
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from playwright.sync_api import sync_playwright

from qs.app.service import QuizSolverService
from qs.cli.cli import print_banner, print_quiz_summary
from qs.cli.theme import console
from qs.config import DEFAULT_GEMINI_MODEL
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

FIXTURE_PATH = Path(__file__).resolve().parent / "tests" / "fixtures" / "blackboard_quiz.html"
FIXTURE_URL = f"file://{FIXTURE_PATH.resolve()}"
RECORDING_DIR = Path(__file__).resolve().parent / "recordings" / "browser"


def _mock_gemini_solver(
    prompt: str | None = None,
    images: object = None,
    response_schema: object = None,
    system_instruction: object = None,
) -> object:
    """Provide realistic vision reasoning responses."""
    time.sleep(0.35)

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


def run_recorded_demo(path: str | Path | None = None) -> int:
    """Run quiz-solver demo while recording browser to video."""
    setup_logger(verbose=False)
    print_banner()

    target_fixture = Path(path).resolve() if path else FIXTURE_PATH
    if not target_fixture.exists():
        console.print(f"[error]Fixture file not found: {target_fixture}[/error]")
        return 1

    fixture_url = f"file://{target_fixture.resolve()}"

    console.print("[bold white]Blackboard Assessment Demo[/bold white]")
    console.print(f"[muted]Fixture:[/] {target_fixture.name}")
    try:
        api_key = Credentials.get_api_key()
        gemini_service = GeminiService(api_key=api_key)
        console.print(
            f"[success]Authenticated with Gemini API[/success] [muted]({DEFAULT_GEMINI_MODEL})[/muted]\n"
        )
    except Exception:
        console.print("[muted]Running with mock reasoning (offline)[/muted]\n")
        gemini_service = MagicMock(spec=GeminiService)
        gemini_service.send.side_effect = _mock_gemini_solver

    # Clear previous recordings in recording dir
    RECORDING_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in RECORDING_DIR.glob("*.webm"):
        old_file.unlink(missing_ok=True)

    console.print("[primary]1/4 Browser Session[/primary]")
    console.print("Launching Chromium browser (headless)...")

    pw = sync_playwright().start()
    user_data_dir = Path("/tmp/quiz_solver_demo_profile")
    user_data_dir.mkdir(parents=True, exist_ok=True)

    context = pw.chromium.launch_persistent_context(
        user_data_dir=str(user_data_dir),
        headless=True,
        viewport={"width": 640, "height": 720},
        record_video_dir=str(RECORDING_DIR),
        record_video_size={"width": 640, "height": 720},
        args=["--disable-blink-features=AutomationControlled"],
        ignore_default_args=["--enable-automation"],
    )
    context.set_default_timeout(30000)

    page = context.pages[0] if context.pages else context.new_page()
    video = page.video

    try:
        console.print("Navigating to Blackboard quiz fixture...")
        session_url = fixture_url
        page.goto(session_url)
        console.print("[success]Blackboard quiz page loaded.[/success]\n")
        time.sleep(0.6)

        # Inject smooth scrolling behavior
        page.evaluate("document.documentElement.style.scrollBehavior = 'smooth';")

        console.print("[primary]2/4 Scanning Page DOM[/primary]")
        console.print("Locating Blackboard question containers...")

        total_questions = 0
        discovered_cards: list[Any] = []

        def on_stage(stage: str, data: dict[str, Any]) -> None:
            nonlocal total_questions, discovered_cards
            if stage == "discovered":
                total_questions = data.get("total", 0)
                discovered_cards = data.get("cards", [])
                console.print(
                    f"[success]Found {total_questions} question containers in DOM.[/success]"
                )
                for idx, card in enumerate(discovered_cards, start=1):
                    q_id = extract_question_id(card, idx)
                    q_type = detect_question_type(card)
                    console.print(f"  {idx}. [bold]{q_id}[/bold] - {q_type.value}")
                console.print()
                console.print("[primary]3/4 Question Solving Pipeline[/primary]")
                time.sleep(0.4)

            elif stage == "scrape_start":
                order = data.get("order", 1)
                console.print(f"\n[bold white]Question {order}/{total_questions}[/bold white]")
                console.print(
                    "[muted]  [1/3 Scrape][/muted] Extracting question context and element screenshot..."
                )
                if 1 <= order <= len(discovered_cards):
                    card = discovered_cards[order - 1]
                    with contextlib.suppress(Exception):
                        card.evaluate(
                            "node => { node.scrollIntoView({ behavior: 'smooth', block: 'center' }); node.classList.add('is-active'); }"
                        )
                time.sleep(0.3)

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
                order = data.get("order", 1)
                time.sleep(0.8 if order == total_questions else 0.35)

            elif stage == "error":
                err = data.get("error", "Unknown error")
                console.print(f"  [error]Failed:[/] {err}")

        with QuizSolverService(
            gemini_service=gemini_service,
            capture_screenshots=True,
        ) as solver:
            batch_result = solver.solve_quiz(
                page,
                on_stage=on_stage,
            )

        console.print("\n[primary]Summary & Review[/primary]")
        print_quiz_summary(batch_result)

        # Scroll to top of assessment to show progress bar completed
        with contextlib.suppress(Exception):
            page.evaluate("window.scrollTo({ top: 0, behavior: 'smooth' });")

        console.print("[bold green]Quiz processing completed.[/bold green]")
        console.print("All answers are filled in the browser. Quiz was NOT submitted.")
        console.print("Review answers directly in Blackboard before submitting.\n")

        # Hold on final state so viewer sees results
        time.sleep(2.5)
        return 0

    except KeyboardInterrupt:
        console.print("\n[muted]Interrupted by user.[/muted]")
        return 0
    except Exception as e:
        console.print(f"\n[error]Demo error: {e}[/error]")
        return 1
    finally:
        context.close()
        video_path = video.path() if video else None
        pw.stop()
        if video_path and Path(video_path).exists():
            dest = Path(__file__).resolve().parent / "recordings" / "browser.webm"
            shutil.copy(video_path, dest)
            dest_size = dest.stat().st_size
            console.print(
                f"[muted]Browser recording saved: {dest.name} ({dest_size} bytes)[/muted]"
            )


def main() -> int:
    """CLI entry point for recorded demo script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Demo runner with synchronized Playwright browser recording"
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=None,
        help="Path to HTML fixture file to test (default: tests/fixtures/blackboard_quiz.html)",
    )
    args = parser.parse_args()
    return run_recorded_demo(path=args.path)


if __name__ == "__main__":
    sys.exit(main())
