"""Quiz Solver Application orchestration and service layer."""

from __future__ import annotations

from qs.app.service import QuizSolverService, solve_quiz

__all__ = [
    "QuizSolverService",
    "solve_quiz",
]
