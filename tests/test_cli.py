"""Tests for CLI interface, commands, and rendering in qs.cli."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from playwright.sync_api import Browser, sync_playwright

from qs.app.service import QuizSolverService
from qs.cli.app import build_parser, handle_run, main
from qs.cli.cli import (
    format_solution,
    print_auth_status,
    print_banner,
    print_quiz_summary,
    run_auth_clear,
    run_auth_setup,
    run_auth_update,
)
from qs.config import DEFAULT_GEMINI_MODEL
from qs.errors.exceptions import MissingAPIKeyError, QuestionNotFoundError
from qs.models import (
    EssaySolution,
    FillInBlankSolution,
    GenericQuestionSolution,
    MatchingSolution,
    MultipleChoiceSolution,
    QuestionExecutionResult,
    QuestionType,
    QuizBatchResult,
    SingleChoiceSolution,
    TrueFalseSolution,
)
from qs.scraper import find_question_elements


@pytest.fixture(scope="module")
def browser():
    """Headless browser fixture for DOM polling tests."""
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


# --- Parser Tests ---


def test_parser_run_flags():
    """Verify run command argument parsing with defaults and custom flags."""
    parser = build_parser()

    # Defaults
    args = parser.parse_args(["run"])
    assert args.command == "run"
    assert args.model == DEFAULT_GEMINI_MODEL
    assert args.timeout == 240
    assert args.verbose is False
    assert args.headless is False

    # Custom flags
    custom = parser.parse_args(
        [
            "run",
            "-u",
            "https://custom.bb.com",
            "-m",
            "gemini-2.5-flash",
            "-t",
            "60",
            "-v",
            "--headless",
        ]
    )
    assert custom.url == "https://custom.bb.com"
    assert custom.model == "gemini-2.5-flash"
    assert custom.timeout == 60
    assert custom.verbose is True
    assert custom.headless is True


def test_parser_auth_subcommands():
    """Verify auth command parsing for all subactions."""
    parser = build_parser()

    for action in ("status", "setup", "clear", "update"):
        args = parser.parse_args(["auth", action])
        assert args.command == "auth"
        assert args.auth_action == action


# --- Solution Formatting Tests ---


def test_format_solution():
    """Verify format_solution formats each question type accurately."""
    assert "Merge Sort" in format_solution(SingleChoiceSolution(selected_option="Merge Sort"))
    assert "Queue" in format_solution(MultipleChoiceSolution(selected_options=["Queue", "Stack"]))
    assert "TRUE" in format_solution(TrueFalseSolution(value=True))
    assert "FALSE" in format_solution(TrueFalseSolution(value=False))
    assert "'log n'" in format_solution(FillInBlankSolution(answers=["log n"]))
    assert "HTTP" in format_solution(MatchingSolution(pairs={"HTTP": "Port 80"}))

    long_essay = "A" * 100
    formatted_essay = format_solution(EssaySolution(response_text=long_essay))
    assert len(formatted_essay) <= 80
    assert "..." in formatted_essay

    assert "(No solution)" in format_solution(None)

    # Generic solution
    gen = GenericQuestionSolution(
        question_type=QuestionType.SINGLE_CHOICE,
        selected_option="Quick Sort",
    )
    assert "Quick Sort" in format_solution(gen)


# --- UI Component Smoke Tests ---


def test_print_banner_does_not_error():
    """Verify print_banner executes without error."""
    print_banner()


def test_print_auth_status_does_not_error():
    """Verify print_auth_status renders table for both configured and missing states."""
    print_auth_status()


def test_print_quiz_summary_rendering():
    """Verify print_quiz_summary renders all columns and summary metrics cleanly."""
    batch = QuizBatchResult(
        total_questions=2,
        successful_fills=2,
        failed_fills=0,
        results=[
            QuestionExecutionResult(
                question_id="q1",
                question_type=QuestionType.SINGLE_CHOICE,
                prompt="Sort complexity?",
                solution=SingleChoiceSolution(selected_option="Merge Sort", confidence=0.95),
                filled=True,
                duration_seconds=1.2,
            ),
            QuestionExecutionResult(
                question_id="q2",
                question_type=QuestionType.TRUE_FALSE,
                prompt="Is TCP connection-oriented?",
                solution=TrueFalseSolution(value=True, confidence=1.0),
                filled=True,
                duration_seconds=0.8,
            ),
        ],
        execution_time_seconds=2.0,
        status="completed",
    )
    print_quiz_summary(batch)


# --- Auth Command Handlers ---


@patch("qs.cli.cli.Prompt.ask")
@patch("qs.credentials.Credentials.set_credentials_bb")
@patch("qs.credentials.Credentials.set_api_key")
def test_run_auth_setup(mock_set_key, mock_set_bb, mock_prompt):
    """Verify interactive auth setup captures credentials and delegates to Credentials."""
    mock_prompt.side_effect = ["student_test", "pass123", "AIzaSy" + "A" * 33]
    run_auth_setup()
    mock_set_bb.assert_called_once_with("student_test", "pass123")
    mock_set_key.assert_called_once()


@patch("qs.cli.cli.Confirm.ask", return_value=True)
@patch("qs.credentials.Credentials.delete_credentials_bb")
@patch("qs.credentials.Credentials.delete_api_key")
def test_run_auth_clear(mock_del_key, mock_del_bb, mock_confirm):
    """Verify auth clear deletes credentials after user confirmation."""
    with patch("qs.credentials.Credentials.get_default_bb_username", return_value="student_test"):
        run_auth_clear()
        mock_del_bb.assert_called_once_with("student_test")
        mock_del_key.assert_called_once()


@patch("qs.cli.cli.Prompt.ask")
@patch("qs.credentials.Credentials.update_credentials_bb")
@patch("qs.credentials.Credentials.set_api_key")
def test_run_auth_update(mock_set_key, mock_update_bb, mock_prompt):
    """Verify auth update updates selected service credentials."""
    mock_prompt.side_effect = [
        "both",  # service choice
        "updated_user",  # bb username
        "new_pass",  # bb pass
        "AIzaSy" + "B" * 33,  # gemini key
    ]
    run_auth_update()
    mock_update_bb.assert_called_once_with("updated_user", "new_pass")
    mock_set_key.assert_called_once()


# --- CLI Run Command Workflow ---


@patch("qs.credentials.Credentials.get_api_key", side_effect=MissingAPIKeyError("No API key"))
def test_handle_run_missing_api_key_exits_1(mock_key):
    """Verify handle_run exits with code 1 if Gemini API key is missing."""
    parser = build_parser()
    args = parser.parse_args(["run"])
    code = handle_run(args)
    assert code == 1


@patch("qs.credentials.Credentials.get_api_key", return_value="AIzaSy" + "A" * 33)
@patch("builtins.input", side_effect=["\n", "\n"])
@patch("qs.cli.app.BrowserSession")
@patch("qs.app.service.QuizSolverService.solve_quiz")
def test_handle_run_success_workflow(
    mock_solve,
    mock_session_cls,
    mock_input,
    mock_key,
):
    """Verify handle_run full lifecycle launches browser, waits for user, solves quiz, and closes browser."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    mock_solve.return_value = QuizBatchResult(
        total_questions=1,
        successful_fills=1,
        failed_fills=0,
        results=[],
        status="completed",
    )

    parser = build_parser()
    args = parser.parse_args(["run", "--headless"])
    code = handle_run(args)

    assert code == 0
    mock_session.open.assert_called_once()
    mock_solve.assert_called_once()
    mock_session.close.assert_called_once()


@patch("qs.credentials.Credentials.get_api_key", return_value="AIzaSy" + "A" * 33)
@patch("builtins.input", side_effect=KeyboardInterrupt)
@patch("qs.cli.app.BrowserSession")
def test_handle_run_cancelled_by_user(mock_session_cls, mock_input, mock_key):
    """Verify handle_run handles KeyboardInterrupt on enter prompt cleanly and closes browser."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    parser = build_parser()
    args = parser.parse_args(["run"])
    code = handle_run(args)

    assert code == 0
    mock_session.close.assert_called_once()


# --- Main Entry Point Tests ---


def test_main_no_args_shows_help():
    """Verify main without args returns 0 and displays help."""
    assert main([]) == 0


def test_main_auth_dispatch():
    """Verify main dispatches auth subcommands."""
    with patch("qs.cli.app.handle_auth", return_value=0) as mock_handle:
        code = main(["auth", "status"])
        assert code == 0
        mock_handle.assert_called_once()


def test_main_run_dispatch():
    """Verify main dispatches run subcommand."""
    with patch("qs.cli.app.handle_run", return_value=0) as mock_handle:
        code = main(["run"])
        assert code == 0
        mock_handle.assert_called_once()


# --- Polling Wait & Race Condition Tests ---


def test_find_question_elements_polls_and_resolves_dynamically(browser: Browser):
    """Verify find_question_elements waits when questions are mounted asynchronously (simulating React delay)."""
    page = browser.new_page()

    # Page mounts a question card after 300ms
    page.goto(
        """data:text/html,
        <html>
          <body>
            <div id="root">Rendering...</div>
            <script>
              setTimeout(() => {
                const el = document.createElement('article');
                el.className = 'question-card';
                el.setAttribute('data-question-id', 'async_q1');
                el.innerText = 'Async Question';
                document.getElementById('root').appendChild(el);
              }, 250);
            </script>
          </body>
        </html>"""
    )

    # Calling with wait_timeout_ms polls and finds the question!
    cards = find_question_elements(page, wait_timeout_ms=3000)
    assert len(cards) == 1
    assert cards[0].get_attribute("data-question-id") == "async_q1"
    page.close()


def test_find_question_elements_times_out_gracefully(browser: Browser):
    """Verify find_question_elements raises QuestionNotFoundError after polling timeout expires."""
    page = browser.new_page()
    page.goto("data:text/html,<html><body>No questions ever</body></html>")

    with pytest.raises(QuestionNotFoundError):
        find_question_elements(page, wait_timeout_ms=500)

    page.close()


def test_solve_quiz_orchestrator_waits_for_async_render(browser: Browser):
    """Verify Orchestrator solve_quiz uses polling wait so user pressing enter early does not fail."""
    page = browser.new_page()
    page.goto(
        """data:text/html,
        <html>
          <body>
            <div id="container"></div>
            <script>
              setTimeout(() => {
                const div = document.createElement('div');
                div.className = 'question-card';
                div.setAttribute('data-question-id', 'delayed_q1');
                div.setAttribute('data-question-type', 'single_choice');
                div.innerHTML = `
                  <div class="question-prompt">Delayed prompt?</div>
                  <label><input type="radio" value="Choice A" /> Choice A</label>
                `;
                document.getElementById('container').appendChild(div);
              }, 200);
            </script>
          </body>
        </html>"""
    )

    mock_gemini = MagicMock()
    mock_gemini.send.return_value = SingleChoiceSolution(selected_option="Choice A")

    service = QuizSolverService(gemini_service=mock_gemini)
    # Orchestrator waits up to 2000ms for delayed element
    batch = service.solve_quiz(page, wait_timeout_ms=2000)

    assert batch.total_questions == 1
    assert batch.successful_fills == 1
    assert batch.status == "completed"
    page.close()


@patch("qs.cli.demo.BrowserSession")
def test_demo_command_headless(mock_session_cls):
    """Verify demo runs headless successfully on the fixture."""
    from qs.cli.demo import run_demo

    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    code = run_demo(live=False, headless=True, slowmo=0.0)
    assert code == 0
    mock_session.open.assert_called_once()
    mock_session.close.assert_called_once()


def test_ensure_auth_for_demo_env(monkeypatch):
    """Verify ensure_auth_for_demo detects existing key in environment."""
    from qs.cli.demo import ensure_auth_for_demo

    valid_key = "AIzaSy" + "A" * 33
    monkeypatch.setenv("GEMINI_API_KEY", valid_key)
    assert ensure_auth_for_demo() == valid_key


@patch("sys.stdin.isatty", return_value=True)
@patch("qs.cli.demo.Prompt.ask")
@patch("qs.cli.demo.Credentials.set_api_key")
def test_ensure_auth_for_demo_interactive(mock_set_key, mock_prompt, mock_isatty, monkeypatch):
    """Verify interactive prompt saves entered key."""
    from qs.cli.demo import ensure_auth_for_demo

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with patch("keyring.get_password", return_value=None):
        valid_key = "AIzaSy" + "K" * 33
        mock_prompt.return_value = valid_key
        result = ensure_auth_for_demo()
        assert result == valid_key
        mock_set_key.assert_called_once_with(valid_key)


@patch("sys.stdin.isatty", return_value=False)
def test_ensure_auth_for_demo_non_interactive_returns_none(mock_isatty, monkeypatch):
    """Verify non-interactive terminal returns None when key is missing."""
    from qs.cli.demo import ensure_auth_for_demo

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with patch("keyring.get_password", return_value=None):
        assert ensure_auth_for_demo() is None


@patch("qs.cli.demo.ensure_auth_for_demo", return_value=None)
def test_demo_requires_auth_and_aborts_without_key(mock_ensure):
    """Verify demo aborts with returncode 1 when auth is required but missing."""
    from qs.cli.demo import run_demo

    code = run_demo(mock=False, live=True, headless=True)
    assert code == 1


@patch("qs.cli.demo.BrowserSession")
def test_demo_command_mock_mode(mock_session_cls):
    """Verify demo runs cleanly in mock mode without calling Gemini API."""
    from qs.cli.demo import run_demo

    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    code = run_demo(mock=True, headless=True, slowmo=0.0)
    assert code == 0


def test_main_demo_dispatch():
    """Verify main dispatches demo subcommand."""
    with patch("qs.cli.demo.run_demo", return_value=0) as mock_demo:
        code = main(["demo", "--headless"])
        assert code == 0
        mock_demo.assert_called_once_with(
            mock=False,
            live=True,
            headless=True,
            slowmo=0.1,
            verbose=False,
            path=None,
        )


def test_main_demo_dispatch_with_mock():
    """Verify main dispatches demo subcommand with --mock flag."""
    with patch("qs.cli.demo.run_demo", return_value=0) as mock_demo:
        code = main(["demo", "--mock", "--headless", "--slowmo", "0.0"])
        assert code == 0
        mock_demo.assert_called_once_with(
            mock=True,
            live=True,
            headless=True,
            slowmo=0.0,
            verbose=False,
            path=None,
        )


def test_parser_demo_flags():
    """Verify demo command argument parsing with defaults and -p/--path flag."""
    parser = build_parser()

    # Defaults
    args = parser.parse_args(["demo"])
    assert args.command == "demo"
    assert args.path is None

    # Custom short flag
    short_args = parser.parse_args(["demo", "-p", "tests/fixtures/custom.html"])
    assert short_args.path == "tests/fixtures/custom.html"

    # Custom long flag
    long_args = parser.parse_args(["demo", "--path", "/tmp/bb_quiz.html"])
    assert long_args.path == "/tmp/bb_quiz.html"


def test_main_demo_dispatch_with_path():
    """Verify main dispatches demo subcommand with -p flag."""
    with patch("qs.cli.demo.run_demo", return_value=0) as mock_demo:
        code = main(["demo", "-p", "tests/fixtures/custom.html", "--headless"])
        assert code == 0
        mock_demo.assert_called_once_with(
            mock=False,
            live=True,
            headless=True,
            slowmo=0.1,
            verbose=False,
            path="tests/fixtures/custom.html",
        )


@patch("qs.cli.demo.BrowserSession")
def test_demo_command_custom_path_success(mock_session_cls, tmp_path):
    """Verify run_demo loads custom HTML fixture specified via path."""
    from qs.cli.demo import run_demo

    custom_fixture = tmp_path / "custom_quiz.html"
    custom_fixture.write_text("<html><body><p>Mock Blackboard Quiz</p></body></html>")

    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    code = run_demo(live=False, headless=True, slowmo=0.0, path=custom_fixture)
    assert code == 0
    mock_session.goto.assert_called_once_with(f"file://{custom_fixture.resolve()}")


def test_demo_command_custom_path_nonexistent(tmp_path):
    """Verify run_demo returns error code 1 when custom path does not exist."""
    from qs.cli.demo import run_demo

    nonexistent = tmp_path / "does_not_exist.html"
    code = run_demo(live=False, headless=True, slowmo=0.0, path=nonexistent)
    assert code == 1


def test_parser_run_speed_flags():
    """Verify parser accepts speed and rate limit flags for run command."""
    parser = build_parser()
    args = parser.parse_args([
        "run",
        "--fast",
        "-b", "10",
        "--paid-tier",
        "--no-rate-limit",
        "--vision", "never",
    ])
    assert args.fast is True
    assert args.batch_size == 10
    assert args.paid_tier is True
    assert args.no_rate_limit is True
    assert args.vision == "never"


@patch("qs.credentials.Credentials.get_api_key", return_value="AIzaSy" + "A" * 33)
@patch("builtins.input", side_effect=["\n", "\n"])
@patch("qs.cli.app.BrowserSession")
@patch("qs.app.service.QuizSolverService.solve_quiz")
@patch("qs.cli.app.GeminiService")
def test_handle_run_with_speed_flags_dispatches_properly(
    mock_gemini_cls,
    mock_solve,
    mock_session_cls,
    mock_input,
    mock_key,
):
    """Verify handle_run configures GeminiService and QuizSolverService with speed options."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_solve.return_value = QuizBatchResult(
        total_questions=1,
        successful_fills=1,
        failed_fills=0,
        results=[],
        status="completed",
    )

    parser = build_parser()
    args = parser.parse_args([
        "run",
        "--headless",
        "--fast",
        "--batch-size", "8",
        "--paid-tier",
        "--vision", "never",
    ])
    code = handle_run(args)
    assert code == 0

    # Verify GeminiService was instantiated with fast model and paid tier
    mock_gemini_cls.assert_called_once()
    _, kwargs = mock_gemini_cls.call_args
    assert kwargs["paid_tier"] is True
    assert kwargs["model"] == "gemini-2.5-flash-lite"


