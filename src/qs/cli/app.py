"""Main CLI application entry point for Quiz Solver."""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence

from qs.app.service import QuizSolverService
from qs.browser.session import BrowserSession
from qs.cli.cli import (
    print_auth_status,
    print_banner,
    print_quiz_summary,
    run_auth_clear,
    run_auth_setup,
    run_auth_update,
)
from qs.cli.theme import console
from qs.config import (
    BB_LINK,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_QUESTION_WAIT_TIMEOUT_MS,
    DEFAULT_VISION_MODE,
    FAST_GEMINI_MODEL,
)
from qs.credentials import Credentials
from qs.errors.exceptions import MissingAPIKeyError
from qs.llm.google import GeminiService
from qs.llm.rate_limiter import RateLimiter
from qs.logger import setup_logger
from qs.models import QuestionExecutionResult


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="qs",
        description="Quiz Solver for Blackboard Ultra",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Solve quiz in browser")
    run_parser.add_argument(
        "-u",
        "--url",
        default=BB_LINK,
        help=f"Quiz or Blackboard URL (default: {BB_LINK})",
    )
    run_parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model (default: {DEFAULT_GEMINI_MODEL})",
    )
    run_parser.add_argument(
        "--fast",
        action="store_true",
        help=f"Use low-latency fast model ({FAST_GEMINI_MODEL})",
    )
    run_parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of questions to batch per API call (default: {DEFAULT_BATCH_SIZE})",
    )
    run_parser.add_argument(
        "--paid-tier",
        action="store_true",
        help="Use Paid Tier quota limits (1000 RPM instead of Free Tier 5 RPM)",
    )
    run_parser.add_argument(
        "--no-rate-limit",
        action="store_true",
        help="Disable rate limiting throttling entirely",
    )
    run_parser.add_argument(
        "--vision",
        choices=["adaptive", "always", "never"],
        default=DEFAULT_VISION_MODE,
        help="Vision screenshot mode: adaptive (text-first, screenshot on media/math), always, or never (default: adaptive)",
    )
    run_parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=240,
        help="Timeout in seconds (default: 240s)",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    run_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
    )

    # Command: auth
    auth_parser = subparsers.add_parser("auth", help="Manage credentials")
    auth_subparsers = auth_parser.add_subparsers(dest="auth_action", help="Auth action")

    auth_subparsers.add_parser("status", help="Show credential status")
    auth_subparsers.add_parser("setup", help="Save credentials")
    auth_subparsers.add_parser("clear", help="Delete credentials")
    auth_subparsers.add_parser("update", help="Update credentials")

    # Command: demo
    demo_parser = subparsers.add_parser("demo", help="Run demo on mock quiz fixture")
    demo_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock vision solver instead of real Gemini AI",
    )
    demo_parser.add_argument(
        "--live",
        action="store_true",
        default=True,
        help="Use real Gemini API (default)",
    )
    demo_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser without GUI",
    )
    demo_parser.add_argument(
        "--slowmo",
        type=float,
        default=0.1,
        help="Delay between question fills (default: 0.1s)",
    )
    demo_parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=None,
        help="Path to HTML fixture file to test (default: tests/fixtures/blackboard_quiz.html)",
    )
    demo_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    return parser


def handle_auth(args: argparse.Namespace) -> int:
    """Dispatch authentication subcommands."""
    action = getattr(args, "auth_action", None) or "status"

    match action:
        case "status":
            print_auth_status()
            return 0
        case "setup":
            run_auth_setup()
            return 0
        case "clear":
            run_auth_clear()
            return 0
        case "update":
            run_auth_update()
            return 0
        case _:
            console.print(f"[error]Unknown auth command: {action}[/]")
            return 1


def handle_run(args: argparse.Namespace) -> int:
    """Execute the quiz solving workflow."""
    setup_logger(verbose=args.verbose)
    print_banner()

    # Verify Gemini API key is available
    try:
        api_key = Credentials.get_api_key()
    except MissingAPIKeyError:
        console.print("[bold red]Gemini API key not found.[/]")
        console.print("[muted]Run 'qs auth setup' or set GEMINI_API_KEY.[/]")
        return 1

    # Check Blackboard credentials
    bb_user = Credentials.get_default_bb_username()

    # Launch browser session
    console.print("[primary]Opening browser...[/]")
    session = BrowserSession(headless=args.headless)

    try:
        session.open()
        console.print(f"[muted]Opening {args.url}...[/]")
        session.goto(args.url)

        # Attempt auto-login if login form visible and username known
        if bb_user:
            with contextlib.suppress(Exception):
                session.login(username=bb_user, url=args.url)

        # User navigation trigger
        console.print("\n1. Log in to Blackboard if needed.")
        console.print("2. Open your quiz.")
        console.print("3. Press Enter once questions are visible on screen.\n")

        try:
            input("Press Enter to solve: ")
        except KeyboardInterrupt, EOFError:
            console.print("\n[muted]Cancelled.[/]")
            return 0

        # Execute vision solver
        console.print("\n[primary]Solving quiz...[/]")

        chosen_model = FAST_GEMINI_MODEL if getattr(args, "fast", False) else args.model
        paid_tier = getattr(args, "paid_tier", False)
        no_rate_limit = getattr(args, "no_rate_limit", False)
        batch_size = max(1, getattr(args, "batch_size", DEFAULT_BATCH_SIZE))
        vision_mode = getattr(args, "vision", DEFAULT_VISION_MODE)

        rate_limiter: RateLimiter | None | bool
        if no_rate_limit:
            rate_limiter = False
        elif paid_tier:
            rate_limiter = RateLimiter.create_paid_tier()
        else:
            rate_limiter = None

        gemini_service = GeminiService(
            api_key=api_key,
            model=chosen_model,
            rate_limiter=rate_limiter,
            paid_tier=paid_tier,
        )

        with console.status("Solving questions...", spinner="dots") as status:

            def on_progress(res: QuestionExecutionResult, current: int, total: int) -> None:
                status_label = "filled" if res.filled else "failed"
                status.update(f"Question {current}/{total} ({res.question_id}): {status_label}...")

            with QuizSolverService(
                gemini_service=gemini_service,
                batch_size=batch_size,
                vision_mode=vision_mode,
            ) as solver:
                batch_result = solver.solve_quiz(
                    session.page,
                    wait_timeout_ms=DEFAULT_QUESTION_WAIT_TIMEOUT_MS,
                    on_question_complete=on_progress,
                    batch_size=batch_size,
                )

        # Display review table
        print_quiz_summary(batch_result)

        # Hand-off back to user for final review & manual submit
        console.print("[bold green]Answers filled in browser.[/bold green]")
        console.print("Review answers in Blackboard and submit when ready.")

        with contextlib.suppress(KeyboardInterrupt, EOFError):
            input("\nPress Enter to close browser: ")

        return 0

    except KeyboardInterrupt:
        console.print("\n[muted]Interrupted.[/]")
        return 0
    except Exception as e:
        console.print(f"\n[error]Error: {e}[/]")
        return 1
    finally:
        session.close()
        console.print("[muted]Browser closed.[/]")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry hook."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        print_banner()
        parser.print_help()
        return 0

    if args.command == "auth":
        return handle_auth(args)
    if args.command == "run":
        return handle_run(args)
    if args.command == "demo":
        from qs.cli.demo import run_demo

        return run_demo(
            mock=args.mock,
            live=args.live,
            headless=args.headless,
            slowmo=args.slowmo,
            verbose=args.verbose,
            path=args.path,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
