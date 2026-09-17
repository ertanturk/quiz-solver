"""Logging configuration for Quiz Solver."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from qs.config import DEFAULT_DATE_FORMAT, DEFAULT_LOG_FORMAT

console = Console(stderr=True)


def setup_logger(
    name: str = "qs",
    log_file: Path | str | None = None,
    level: int | str | None = None,
    verbose: bool = False,
) -> logging.Logger:
    """Configure and return the root package logger.

    In normal CLI mode (verbose=False), console logging is restricted to WARNING/ERROR
    so the terminal UX remains clean. Full debug logs appear when verbose=True or
    are directed to an optional log file.
    """
    logger = logging.getLogger(name)

    if verbose:
        effective_level = logging.DEBUG
        console_level = logging.DEBUG
    elif level is not None:
        effective_level = (
            getattr(logging, str(level).upper(), logging.INFO) if isinstance(level, str) else level
        )
        console_level = effective_level
    else:
        effective_level = logging.INFO
        console_level = logging.WARNING

    logger.setLevel(effective_level)

    # Update existing console handlers if already attached
    for h in logger.handlers:
        if isinstance(h, RichHandler):
            h.setLevel(console_level)

    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        rich_handler = RichHandler(
            console=console,
            show_time=False,
            show_path=verbose,
            rich_tracebacks=True,
            markup=True,
        )
        rich_handler.setLevel(console_level)
        logger.addHandler(rich_handler)

    if log_file is not None:
        path = Path(log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        )
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "qs") -> logging.Logger:
    """Retrieve logger for the given name, ensuring root 'qs' logger is initialized."""
    root_qs = logging.getLogger("qs")
    if not root_qs.handlers:
        setup_logger("qs")

    logger = logging.getLogger(name)
    if name != "qs" and name.startswith("qs."):
        logger.propagate = True

    return logger


logger = get_logger()
