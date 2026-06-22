"""Matplotlib theme helpers synced with CustomTkinter appearance."""

from __future__ import annotations

from typing import Literal

ThemeMode = Literal["dark", "light"]

DARK_THEME = {
    "bg_figure": "#242424",
    "bg_axes": "#2b2b2b",
    "text": "#E0E0E0",
    "grid": "#4A4A4A",
    "spine": "#5A5A5A",
}

LIGHT_THEME = {
    "bg_figure": "#F5F5F5",
    "bg_axes": "#FFFFFF",
    "text": "#1A1A1A",
    "grid": "#CCCCCC",
    "spine": "#AAAAAA",
}

CHART_COLORS = {
    "data": "#00FFFF",
    "prediction": "#FF7F50",
    "loss": "#FF1493",
    "loss_faded": "#FF1493",
    "kde": "#7B68EE",
    "marker": "#FFD700",
    "test_data": "#FF6B6B",
}

LEGEND_KWARGS = {
    "loc": "upper right",
    "framealpha": 0.92,
    "facecolor": "#2b2b2b",
    "edgecolor": "#5A5A5A",
    "labelcolor": "#E0E0E0",
    "fontsize": 9,
}


def get_theme_colors(mode: ThemeMode) -> dict[str, str]:
    """Return color palette for the given appearance mode."""
    return DARK_THEME if mode == "dark" else LIGHT_THEME


def apply_plot_theme(fig: object, axes: list[object], mode: ThemeMode) -> None:
    """Apply background, grid, and label colors to a figure and its axes."""
    colors = get_theme_colors(mode)
    fig.patch.set_facecolor(colors["bg_figure"])

    for axis in axes:
        axis.set_facecolor(colors["bg_axes"])
        axis.tick_params(colors=colors["text"], labelcolor=colors["text"])
        axis.xaxis.label.set_color(colors["text"])
        axis.yaxis.label.set_color(colors["text"])
        axis.title.set_color(colors["text"])
        axis.grid(True, color=colors["grid"], alpha=0.35, linestyle="-", linewidth=0.6)
        for spine in axis.spines.values():
            spine.set_color(colors["spine"])
