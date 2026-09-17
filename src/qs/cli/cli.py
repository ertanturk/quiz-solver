"""CLI UI components, tables, and prompts."""

from __future__ import annotations

import contextlib
import os

import keyring
from rich.box import ROUNDED, SIMPLE_HEAVY
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from qs import __version__
from qs.cli.theme import console
from qs.config import API_KEY_ACCOUNT, API_SERVICE_NAME, DEFAULT_GEMINI_MODEL
from qs.credentials import Credentials
from qs.models import (
    EssaySolution,
    FillInBlankSolution,
    GenericQuestionSolution,
    MatchingSolution,
    MultipleChoiceSolution,
    QuizBatchResult,
    SingleChoiceSolution,
    TrueFalseSolution,
)


def print_banner() -> None:
    """Print clean application header on the left."""
    console.print(
        Panel(
            Text(f"Quiz Solver v{__version__}", style="primary"),
            border_style="panel.border",
            box=ROUNDED,
            padding=(0, 2),
            expand=False,
        )
    )


def format_solution(solution: object) -> str:
    """Format a question solution into readable text."""
    if solution is None:
        return "[muted](No solution)[/muted]"

    if isinstance(solution, SingleChoiceSolution):
        return f"[primary.subtle]-[/] {solution.selected_option}"

    if isinstance(solution, MultipleChoiceSolution):
        items = "\n".join(f"[accent]-[/] {opt}" for opt in solution.selected_options)
        return items or "[muted](None selected)[/muted]"

    if isinstance(solution, TrueFalseSolution):
        badge = "[success]TRUE[/]" if solution.value else "[error]FALSE[/]"
        return f"- {badge}"

    if isinstance(solution, FillInBlankSolution):
        blanks = " | ".join(f"[warning]'{ans}'[/]" for ans in solution.answers)
        return blanks or "[muted](Empty)[/muted]"

    if isinstance(solution, MatchingSolution):
        pairs = "\n".join(
            f"[primary.subtle]{prompt}[/] [muted]->[/] [bold_white]{target}[/]"
            for prompt, target in solution.pairs.items()
        )
        return pairs or "[muted](No pairs)[/muted]"

    if isinstance(solution, EssaySolution):
        text = solution.response_text.strip()
        if len(text) > 80:
            return f"{text[:77]}..."
        return text

    if isinstance(solution, GenericQuestionSolution):
        if solution.selected_option:
            return f"- {solution.selected_option}"
        if solution.selected_options:
            return "\n".join(f"- {opt}" for opt in solution.selected_options)
        if solution.bool_value is not None:
            return "[success]TRUE[/]" if solution.bool_value else "[error]FALSE[/]"
        if solution.fill_blanks:
            return " | ".join(f"'{ans}'" for ans in solution.fill_blanks)
        if solution.matching_pairs:
            return "\n".join(f"{k} -> {v}" for k, v in solution.matching_pairs.items())
        if solution.essay_text:
            text = solution.essay_text.strip()
            return f"{text[:77]}..." if len(text) > 80 else text

    return str(solution)


def print_quiz_summary(batch_result: QuizBatchResult) -> None:
    """Render table of answers and summary metrics."""
    table = Table(
        title="[bold white]Quiz Results[/bold white]",
        box=ROUNDED,
        border_style="table.border",
        header_style="table.header",
        show_lines=True,
    )

    table.add_column("#", style="muted", justify="right")
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="accent")
    table.add_column("Prompt", style="white", max_width=38)
    table.add_column("Answer", max_width=42)
    table.add_column("Conf.", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Time", justify="right", style="muted")

    for index, res in enumerate(batch_result.results, start=1):
        conf_str = "-"
        if res.solution is not None:
            conf_val = getattr(res.solution, "confidence", None)
            if conf_val is not None:
                pct = int(conf_val * 100)
                if pct >= 90:
                    conf_str = f"[success]{pct}%[/]"
                elif pct >= 70:
                    conf_str = f"[warning]{pct}%[/]"
                else:
                    conf_str = f"[error]{pct}%[/]"

        status_badge = "[success]FILLED[/]" if res.filled else "[error]FAILED[/]"

        prompt_snippet = res.prompt.strip()
        if len(prompt_snippet) > 75:
            prompt_snippet = f"{prompt_snippet[:72]}..."
        if not prompt_snippet:
            prompt_snippet = "[muted](Screenshot)[/muted]"

        answer_text = format_solution(res.solution)
        if res.error:
            answer_text += f"\n[error]Error: {res.error}[/]"

        table.add_row(
            str(index),
            res.question_id,
            res.question_type.value,
            prompt_snippet,
            answer_text,
            conf_str,
            status_badge,
            f"{res.duration_seconds:.1f}s",
        )

    console.print()
    console.print(table)
    console.print()

    status_style = "success" if batch_result.status == "completed" else "warning"
    if batch_result.status == "failed":
        status_style = "error"

    metrics_text = Text()
    metrics_text.append("Questions: ", style="muted")
    metrics_text.append(f"{batch_result.total_questions}    ", style="bold_white")
    metrics_text.append("Filled: ", style="muted")
    metrics_text.append(f"{batch_result.successful_fills}    ", style="success")
    metrics_text.append("Failed: ", style="muted")
    metrics_text.append(
        f"{batch_result.failed_fills}    ",
        style="error" if batch_result.failed_fills > 0 else "muted",
    )
    metrics_text.append("Time: ", style="muted")
    metrics_text.append(f"{batch_result.execution_time_seconds:.2f}s    ", style="bold_white")
    metrics_text.append("Status: ", style="muted")
    metrics_text.append(batch_result.status.upper(), style=f"bold {status_style}")

    console.print(
        Panel(
            metrics_text,
            title="[bold white]Summary[/bold white]",
            border_style=status_style,
            box=ROUNDED,
            padding=(0, 2),
        )
    )

    notice_box = Panel(
        "[bold yellow]Notice:[/bold yellow] Answers filled in browser. The tool does not submit. Review and submit manually in Blackboard.",
        border_style="warning",
        box=ROUNDED,
        padding=(0, 2),
    )
    console.print(notice_box)
    console.print()


def print_auth_status(bb_user: str | None = None, model: str = DEFAULT_GEMINI_MODEL) -> None:
    """Print current authentication status."""
    table = Table(
        title="[bold white]Auth Status[/bold white]",
        box=SIMPLE_HEAVY,
        border_style="table.border",
        header_style="table.header",
        show_lines=True,
    )

    table.add_column("Service", style="primary", width=20)
    table.add_column("Account", style="bold_white", width=25)
    table.add_column("Status", width=20)
    table.add_column("Details", style="muted", width=30)

    resolved_bb_user = bb_user or Credentials.get_default_bb_username()
    if resolved_bb_user:
        try:
            _, masked_pwd = Credentials.show_credentials_bb(resolved_bb_user)
            table.add_row(
                "Blackboard",
                resolved_bb_user,
                "[badge.success] CONFIGURED [/]",
                f"Password: {masked_pwd}",
            )
        except Exception:
            table.add_row(
                "Blackboard",
                resolved_bb_user,
                "[badge.warning] PARTIAL [/]",
                "Password missing in keyring",
            )
    else:
        table.add_row(
            "Blackboard",
            "(None)",
            "[badge.error] NOT SET [/]",
            "Run 'qs auth setup'",
        )

    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    stored_key = None
    if not (env_key and env_key.strip()):
        with contextlib.suppress(Exception):
            stored_key = keyring.get_password(API_SERVICE_NAME, API_KEY_ACCOUNT)

    if (env_key and env_key.strip()) or (stored_key and stored_key.strip()):
        try:
            masked_key = Credentials.show_api_key()
            table.add_row(
                "Gemini API",
                model,
                "[badge.success] CONFIGURED [/]",
                f"Key: {masked_key[:8]}...{masked_key[-4:]}",
            )
        except Exception:
            table.add_row(
                "Gemini API",
                model,
                "[badge.error] NOT SET [/]",
                "GEMINI_API_KEY missing",
            )
    else:
        table.add_row(
            "Gemini API",
            model,
            "[badge.error] NOT SET [/]",
            "GEMINI_API_KEY missing",
        )

    console.print(table)


def run_auth_setup() -> None:
    """Interactive credential setup."""
    console.print(
        Panel(
            "[bold white]Setup Credentials[/bold white]",
            border_style="primary",
            box=ROUNDED,
        )
    )

    console.print("\n[primary]Blackboard[/primary]")
    existing_bb = Credentials.get_default_bb_username()
    bb_user = Prompt.ask(
        "Username",
        default=existing_bb or "",
        console=console,
    ).strip()

    if bb_user:
        bb_pass = Prompt.ask(
            "Password",
            password=True,
            console=console,
        )
        if bb_pass:
            Credentials.set_default_bb_username(bb_user)
            try:
                Credentials.set_credentials_bb(bb_user, bb_pass)
            except Exception:
                Credentials.update_credentials_bb(bb_user, bb_pass)
            console.print(f"[success]Saved Blackboard credentials for '{bb_user}'.[/]")

    console.print("\n[primary]Gemini API Key[/primary]")
    existing_has_key = False
    try:
        Credentials.get_api_key()
        existing_has_key = True
    except Exception:
        pass

    prompt_msg = "API Key (min 39 chars)"
    if existing_has_key:
        prompt_msg += " [muted](leave blank to keep current)[/muted]"

    gemini_key = Prompt.ask(
        prompt_msg,
        password=True,
        console=console,
    ).strip()

    if gemini_key:
        try:
            Credentials.set_api_key(gemini_key)
            console.print("[success]Saved Gemini API key.[/]")
        except Exception as e:
            console.print(f"[error]Failed to save API key: {e}[/]")
    elif existing_has_key:
        console.print("[muted]Kept existing API key.[/]")

    console.print("\n[success]Setup done.[/]\n")


def run_auth_clear() -> None:
    """Clear all credentials."""
    if not Confirm.ask("Clear all saved credentials?", console=console):
        console.print("[muted]Cancelled.[/]")
        return

    bb_user = Credentials.get_default_bb_username()
    if bb_user:
        with contextlib.suppress(Exception):
            Credentials.delete_credentials_bb(bb_user)
        Credentials.delete_default_bb_username()
        console.print(f"[success]Removed Blackboard credentials for '{bb_user}'.[/]")
    else:
        console.print("[muted]No Blackboard credentials found.[/]")

    try:
        Credentials.delete_api_key()
        console.print("[success]Removed Gemini API key.[/]")
    except Exception:
        console.print("[muted]No Gemini API key found in keyring.[/]")

    console.print("[success]Credentials cleared.[/]")


def run_auth_update() -> None:
    """Update credentials."""
    console.print("[primary]Update Credentials[/primary]\n")
    service_choice = Prompt.ask(
        "Update target",
        choices=["blackboard", "gemini", "both"],
        default="both",
        console=console,
    )

    if service_choice in ("blackboard", "both"):
        existing_bb = Credentials.get_default_bb_username()
        bb_user = Prompt.ask(
            "Blackboard username", default=existing_bb or "", console=console
        ).strip()
        if bb_user:
            bb_pass = Prompt.ask("New password", password=True, console=console)
            if bb_pass:
                Credentials.set_default_bb_username(bb_user)
                try:
                    Credentials.update_credentials_bb(bb_user, bb_pass)
                except Exception:
                    Credentials.set_credentials_bb(bb_user, bb_pass)
                console.print(f"[success]Updated Blackboard credentials for '{bb_user}'.[/]")

    if service_choice in ("gemini", "both"):
        gemini_key = Prompt.ask("New Gemini API key", password=True, console=console).strip()
        if gemini_key:
            Credentials.set_api_key(gemini_key)
            console.print("[success]Updated Gemini API key.[/]")
