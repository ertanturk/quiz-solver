"""CLI Theme and Rich styling definitions for Quiz Solver."""

from __future__ import annotations

from rich.console import Console
from rich.style import Style
from rich.theme import Theme

# Color palette inspired by MEF University & Blackboard Ultra
COLOR_PRIMARY = "#00A3A6"  # MEF Teal
COLOR_ACCENT = "#6366F1"  # Indigo
COLOR_SUCCESS = "#10B981"  # Emerald Green
COLOR_WARNING = "#F59E0B"  # Amber
COLOR_ERROR = "#EF4444"  # Crimson Red
COLOR_MUTED = "#9CA3AF"  # Slate Gray
COLOR_BORDER = "#374151"  # Border Gray

CLI_THEME = Theme(
    {
        "primary": f"bold {COLOR_PRIMARY}",
        "primary.subtle": COLOR_PRIMARY,
        "accent": f"bold {COLOR_ACCENT}",
        "success": f"bold {COLOR_SUCCESS}",
        "warning": f"bold {COLOR_WARNING}",
        "error": f"bold {COLOR_ERROR}",
        "muted": COLOR_MUTED,
        "bold_white": "bold white",
        "badge.success": f"bold white on {COLOR_SUCCESS}",
        "badge.warning": f"bold white on {COLOR_WARNING}",
        "badge.error": f"bold white on {COLOR_ERROR}",
        "badge.info": f"bold white on {COLOR_PRIMARY}",
        "table.header": f"bold {COLOR_PRIMARY}",
        "table.border": COLOR_BORDER,
        "panel.border": COLOR_PRIMARY,
    }
)

# Shared console instance
console = Console(theme=CLI_THEME)

STYLE_PRIMARY = Style(color=COLOR_PRIMARY, bold=True)
STYLE_SUCCESS = Style(color=COLOR_SUCCESS, bold=True)
STYLE_ERROR = Style(color=COLOR_ERROR, bold=True)
STYLE_WARNING = Style(color=COLOR_WARNING, bold=True)
STYLE_MUTED = Style(color=COLOR_MUTED)
