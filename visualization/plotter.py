"""
Owner: Thành viên 3 - Visualization / Plot / Animation.

Matplotlib plotting helpers. Does not read CSV or train models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.stats import gaussian_kde

from core.preprocessing import PreprocessReport

from visualization.theme import CHART_COLORS, LEGEND_KWARGS, ThemeMode, apply_plot_theme, get_theme_colors


@dataclass(frozen=True)
class TrainingFrameView:
    """Data returned after drawing one training animation frame."""

    equation: str
    epoch: int
    loss: float
    weights: np.ndarray
    y_pred: np.ndarray
    residuals: np.ndarray
    x_visual: np.ndarray
    y_actual: np.ndarray


class PlotArtists:
    """Reusable matplotlib artists for training animation."""

    def __init__(self) -> None:
        self.scatter_data: Any = None
        self.line_pred: Any = None
        self.line_loss_full: Any = None
        self.line_loss_progress: Any = None
        self.loss_marker: Any = None
        self.loss_point: Any = None
        self.loss_annotation: Any = None
        self.loss_fill: Any = None
        self.residual_lines: LineCollection | None = None
        self.scatter_test: Any = None
        self.initialized = False
        self.fit_xlim: tuple[float, float] = (0.0, 1.0)
        self.fit_ylim: tuple[float, float] = (0.0, 1.0)
        self.loss_xlim: tuple[float, float] = (0.0, 1.0)
        self.loss_ylim: tuple[float, float] = (0.0, 1.0)


class PreviewArtists:
    """Reusable matplotlib artists for CSV data preview."""

    def __init__(self) -> None:
        self.scatter: Any = None
        self.scatter_missing: Any = None
        self.scatter_outliers: Any = None
        self.hist_bars: Any = None
        self.kde_line: Any = None
        self.initialized = False


def _visual_x_values(x_data: np.ndarray, visual_feature_index: int) -> np.ndarray:
    x_data = np.asarray(x_data, dtype=float)
    if x_data.ndim == 1:
        return x_data
    return x_data[:, visual_feature_index]


def _prediction_curve(
    model: object,
    x_data: np.ndarray,
    weights: np.ndarray,
    visual_feature_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_visual = _visual_x_values(x_data, visual_feature_index)
    x_min, x_max = float(x_visual.min()), float(x_visual.max())
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    x_line = np.linspace(x_min, x_max, 300)

    if np.asarray(x_data).ndim == 1:
        x_pred_input = x_line
    else:
        num_features = np.asarray(x_data).shape[1]
        x_pred_input = np.zeros((len(x_line), num_features), dtype=float)
        x_pred_input[:, visual_feature_index] = x_line
        for col in range(num_features):
            if col == visual_feature_index:
                continue
            x_pred_input[:, col] = float(np.median(np.asarray(x_data)[:, col]))

    y_line = model.predict_with_weights(x_pred_input, weights)
    return x_line, y_line


def _padding_limits(
    values: np.ndarray,
    ratio: float = 0.08,
    min_span: float = 1e-6,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    low = float(np.min(values))
    high = float(np.max(values))
    span = max(high - low, min_span)
    pad = span * ratio
    return low - pad, high + pad


def _style_fit_axis(
    ax_fit: Axes,
    x_visual: np.ndarray,
    y_actual: np.ndarray,
    y_curve_samples: list[np.ndarray],
    visual_name: str,
    theme_mode: ThemeMode,
) -> tuple[float, float, float, float]:
    """Apply standard scatter/fit axis styling with fixed limits."""
    colors = get_theme_colors(theme_mode)
    y_samples = [np.asarray(y_actual, dtype=float).ravel(), *y_curve_samples]
    y_all = np.concatenate(y_samples) if y_samples else np.asarray(y_actual, dtype=float)
    x_lo, x_hi = _padding_limits(x_visual)
    y_lo, y_hi = _padding_limits(y_all)

    ax_fit.set_xlim(x_lo, x_hi)
    ax_fit.set_ylim(y_lo, y_hi)
    ax_fit.set_xlabel(visual_name, fontsize=11, labelpad=8)
    ax_fit.set_ylabel("Target (y)", fontsize=11, labelpad=8)
    ax_fit.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_fit.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_fit.grid(True, which="major", color=colors["grid"], alpha=0.35, linestyle="-", linewidth=0.6)
    return x_lo, x_hi, y_lo, y_hi


def _style_loss_axis(
    ax_loss: Axes,
    epochs: np.ndarray,
    losses: np.ndarray,
    theme_mode: ThemeMode,
) -> tuple[float, float, float, float]:
    """Apply standard epoch vs MSE axis styling with fixed limits."""
    colors = get_theme_colors(theme_mode)
    max_epoch = float(np.max(epochs)) if len(epochs) else 1.0
    max_loss = float(np.max(losses)) if len(losses) else 1.0
    min_loss = float(np.min(losses)) if len(losses) else 0.0

    x_lo = 0.0
    x_hi = max(max_epoch, 1.0) * 1.02
    y_lo = 0.0 if min_loss >= 0 else min_loss * 1.05
    y_hi = max(max_loss * 1.12, max_loss + 1e-9, 1e-6)

    ax_loss.set_xlim(x_lo, x_hi)
    ax_loss.set_ylim(y_lo, y_hi)
    ax_loss.set_xlabel("Epoch", fontsize=11, labelpad=8)
    ax_loss.set_ylabel("MSE (Mean Squared Error)", fontsize=11, labelpad=8)
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax_loss.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_loss.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_loss.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
    ax_loss.grid(True, which="major", axis="both", color=colors["grid"], alpha=0.4, linestyle="-", linewidth=0.6)
    ax_loss.set_axisbelow(True)
    return x_lo, x_hi, y_lo, y_hi


def draw_empty_plot(
    fig: Figure,
    ax_fit: Axes,
    ax_loss: Axes,
    theme_mode: ThemeMode = "dark",
) -> None:
    """Draw placeholder axes before any data is loaded."""
    ax_fit.cla()
    ax_loss.cla()
    apply_plot_theme(fig, [ax_fit, ax_loss], theme_mode)
    ax_fit.set_title("Fit — Feature vs Target", fontsize=12, pad=10)
    ax_fit.set_xlabel("Feature")
    ax_fit.set_ylabel("Target")
    ax_loss.set_title("Training Loss vs Epoch", fontsize=12, pad=10)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE")
    ax_fit.text(
        0.5,
        0.5,
        "Load data and train to visualize",
        transform=ax_fit.transAxes,
        ha="center",
        va="center",
        color=CHART_COLORS["data"],
        alpha=0.8,
    )


def draw_csv_preview(
    x_visual: np.ndarray,
    y_target: np.ndarray,
    visual_name: str,
    target_name: str,
    ax_fit: Axes,
    ax_loss: Axes,
    fig: Figure,
    theme_mode: ThemeMode = "dark",
    artists: PreviewArtists | None = None,
    preprocess_report: PreprocessReport | None = None,
    x_visual_raw: np.ndarray | None = None,
    y_target_raw: np.ndarray | None = None,
    target_scale: float | None = None,
) -> PreviewArtists:
    """Draw scatter preview and target histogram with KDE overlay."""
    if artists is None:
        artists = PreviewArtists()

    if not artists.initialized:
        ax_fit.cla()
        ax_loss.cla()
        apply_plot_theme(fig, [ax_fit, ax_loss], theme_mode)
        artists.scatter_missing = ax_fit.scatter(
            [],
            [],
            c="#888888",
            alpha=0.55,
            s=32,
            marker="x",
            label="Dropped (missing)",
        )
        artists.scatter_outliers = ax_fit.scatter(
            [],
            [],
            c="#E57373",
            alpha=0.75,
            s=36,
            edgecolors="none",
            label="Dropped (outlier)",
        )
        artists.scatter = ax_fit.scatter(
            [],
            [],
            c=CHART_COLORS["data"],
            alpha=0.65,
            s=28,
            edgecolors="none",
            label="Clean data",
        )
        artists.hist_bars = None
        artists.kde_line, = ax_loss.plot([], [], color=CHART_COLORS["kde"], linewidth=2.0, label="KDE")
        artists.initialized = True
    else:
        apply_plot_theme(fig, [ax_fit, ax_loss], theme_mode)

    x_visual = np.asarray(x_visual, dtype=float)
    y_target = np.asarray(y_target, dtype=float)

    def _scale_y(value: float) -> float:
        if target_scale is not None and np.isfinite(value):
            return float(value) / target_scale
        return float(value)

    artists.scatter.set_offsets(np.column_stack([x_visual, y_target]))

    if preprocess_report is not None and x_visual_raw is not None and y_target_raw is not None:
        x_all = np.asarray(x_visual_raw, dtype=float)
        y_all = np.asarray(y_target_raw, dtype=float)
        missing_idx = preprocess_report.missing_mask
        outlier_idx = preprocess_report.outlier_mask

        missing_x, missing_y = [], []
        for xi, yi in zip(x_all[missing_idx], y_all[missing_idx]):
            if np.isfinite(xi) and np.isfinite(yi):
                missing_x.append(xi)
                missing_y.append(_scale_y(yi))
            elif np.isfinite(xi):
                missing_x.append(xi)
                missing_y.append(float(np.nanmin(y_target)) if len(y_target) else 0.0)
            elif np.isfinite(yi):
                missing_x.append(float(np.nanmin(x_visual)) if len(x_visual) else 0.0)
                missing_y.append(_scale_y(yi))

        outlier_x = x_all[outlier_idx]
        outlier_y = np.array([_scale_y(yi) for yi in y_all[outlier_idx]], dtype=float)
        outlier_keep = np.isfinite(outlier_x) & np.isfinite(outlier_y)
        artists.scatter_missing.set_offsets(
            np.column_stack([missing_x, missing_y]) if missing_x else np.empty((0, 2))
        )
        artists.scatter_outliers.set_offsets(
            np.column_stack([outlier_x[outlier_keep], outlier_y[outlier_keep]])
            if outlier_keep.any()
            else np.empty((0, 2))
        )
    else:
        artists.scatter_missing.set_offsets(np.empty((0, 2)))
        artists.scatter_outliers.set_offsets(np.empty((0, 2)))

    _style_fit_axis(ax_fit, x_visual, y_target, [y_target], visual_name, theme_mode)
    if target_scale is not None:
        ax_fit.set_ylabel("{} (÷{:.0e})".format(target_name, target_scale), fontsize=11, labelpad=8)
    title_suffix = " (after preprocess)" if preprocess_report is not None else ""
    ax_fit.set_title(
        "{} vs {} — {} points{}".format(visual_name, target_name, len(x_visual), title_suffix),
        fontsize=12,
        pad=10,
    )
    ax_fit.legend(**LEGEND_KWARGS)

    ax_loss.cla()
    apply_plot_theme(fig, [ax_loss], theme_mode)
    if len(y_target) > 1:
        ax_loss.hist(
            y_target,
            bins=min(30, max(10, len(y_target) // 3)),
            density=True,
            alpha=0.45,
            color=CHART_COLORS["data"],
            edgecolor="#5A5A5A",
            label="Histogram",
        )
        if len(y_target) >= 2 and np.std(y_target) > 0:
            kde = gaussian_kde(y_target)
            x_kde = np.linspace(y_target.min(), y_target.max(), 200)
            ax_loss.plot(x_kde, kde(x_kde), color=CHART_COLORS["kde"], linewidth=2.0, label="KDE")
    ax_loss.set_title("Distribution of {}".format(target_name), fontsize=12, pad=10)
    ax_loss.set_xlabel(target_name)
    ax_loss.set_ylabel("Density")
    ax_loss.legend(**LEGEND_KWARGS)

    fig.canvas.draw_idle()
    return artists


def init_training_artists(
    fig: Figure,
    ax_fit: Axes,
    ax_loss: Axes,
    model: object,
    x_data: np.ndarray,
    y_data: np.ndarray,
    visual_feature_index: int,
    feature_names: list[str],
    theme_mode: ThemeMode = "dark",
    x_test_visual: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    y_test_pred: np.ndarray | None = None,
) -> PlotArtists:
    """Create reusable artists for training animation."""
    artists = PlotArtists()
    ax_fit.cla()
    ax_loss.cla()
    apply_plot_theme(fig, [ax_fit, ax_loss], theme_mode)

    x_visual = _visual_x_values(x_data, visual_feature_index)
    y_data = np.asarray(y_data, dtype=float)
    visual_name = feature_names[visual_feature_index] if feature_names else "x"

    frames = model.frames
    epochs = np.array([item.epoch for item in frames], dtype=float)
    losses = np.array([item.loss for item in frames], dtype=float)

    curve_samples: list[np.ndarray] = []
    if frames:
        for weights in (frames[0].weights, frames[-1].weights):
            _, y_curve = _prediction_curve(model, x_data, weights, visual_feature_index)
            curve_samples.append(y_curve)
    if y_test is not None:
        curve_samples.append(np.asarray(y_test, dtype=float).ravel())
    if y_test_pred is not None:
        curve_samples.append(np.asarray(y_test_pred, dtype=float).ravel())

    x_lo, x_hi, y_lo, y_hi = _style_fit_axis(
        ax_fit, x_visual, y_data, curve_samples, visual_name, theme_mode
    )
    artists.fit_xlim = (x_lo, x_hi)
    artists.fit_ylim = (y_lo, y_hi)

    artists.scatter_data = ax_fit.scatter(
        x_visual,
        y_data,
        c=CHART_COLORS["data"],
        alpha=0.65,
        s=36,
        edgecolors="#1a1a1a",
        linewidths=0.4,
        label="Train data (80%)",
        zorder=3,
    )
    if x_test_visual is not None and y_test is not None and len(y_test) > 0:
        artists.scatter_test = ax_fit.scatter(
            np.asarray(x_test_visual, dtype=float),
            np.asarray(y_test, dtype=float),
            c=CHART_COLORS["test_data"],
            alpha=0.9,
            s=64,
            marker="D",
            edgecolors="#FFFFFF",
            linewidths=0.8,
            label="Test data (20%)",
            zorder=5,
        )
    else:
        artists.scatter_test = ax_fit.scatter([], [], s=0, label="_test")
    artists.line_pred, = ax_fit.plot(
        [],
        [],
        color=CHART_COLORS["prediction"],
        linewidth=2.8,
        label="Model fit",
        zorder=4,
    )
    artists.residual_lines = LineCollection(
        [], colors=CHART_COLORS["marker"], alpha=0.28, linewidths=0.9, zorder=2
    )
    ax_fit.add_collection(artists.residual_lines)
    ax_fit.set_title("Model Fit", fontsize=12, fontweight="bold", pad=10)
    ax_fit.legend(**LEGEND_KWARGS)

    lx_lo, lx_hi, ly_lo, ly_hi = _style_loss_axis(ax_loss, epochs, losses, theme_mode)
    artists.loss_xlim = (lx_lo, lx_hi)
    artists.loss_ylim = (ly_lo, ly_hi)

    artists.loss_fill = ax_loss.fill_between(
        epochs,
        losses,
        y2=ly_lo,
        color=CHART_COLORS["loss"],
        alpha=0.08,
        zorder=1,
    )
    artists.line_loss_full, = ax_loss.plot(
        epochs,
        losses,
        color=CHART_COLORS["loss"],
        alpha=0.28,
        linewidth=1.2,
        linestyle="--",
        label="Full training",
        zorder=2,
    )
    artists.line_loss_progress, = ax_loss.plot(
        [],
        [],
        color=CHART_COLORS["loss"],
        linewidth=2.8,
        solid_capstyle="round",
        label="Current run",
        zorder=3,
    )
    artists.loss_point = ax_loss.scatter(
        [],
        [],
        s=90,
        c=CHART_COLORS["marker"],
        edgecolors="#FFFFFF",
        linewidths=1.5,
        zorder=5,
        label="_current",
    )
    artists.loss_marker = ax_loss.axvline(
        0,
        color=CHART_COLORS["marker"],
        linestyle=":",
        alpha=0.55,
        linewidth=1.2,
        zorder=2,
    )
    colors = get_theme_colors(theme_mode)
    artists.loss_annotation = ax_loss.text(
        0.03,
        0.97,
        "",
        transform=ax_loss.transAxes,
        va="top",
        ha="left",
        color=colors["text"],
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#1e1e1e", edgecolor="#5A5A5A", alpha=0.92),
    )
    ax_loss.set_title("Training Loss", fontsize=12, fontweight="bold", pad=10)
    ax_loss.legend(**LEGEND_KWARGS)

    artists.initialized = True
    return artists


def draw_training_frame(
    model: object,
    x_data: np.ndarray,
    y_data: np.ndarray,
    frame_index: int,
    feature_names: list[str],
    visual_feature_index: int,
    ax_fit: Axes,
    ax_loss: Axes,
    fig: Figure,
    artists: PlotArtists,
    theme_mode: ThemeMode = "dark",
) -> TrainingFrameView:
    """Update training artists for a single frame and return point-level details."""
    if not artists.initialized:
        raise ValueError("Training artists must be initialized before drawing frames.")

    frames = model.frames
    if not frames:
        return TrainingFrameView(
            equation="",
            epoch=0,
            loss=0.0,
            weights=np.array([]),
            y_pred=np.array([]),
            residuals=np.array([]),
            x_visual=np.array([]),
            y_actual=np.array([]),
        )

    frame_index = max(0, min(frame_index, len(frames) - 1))
    frame = frames[frame_index]
    y_actual = np.asarray(y_data, dtype=float).ravel()
    y_pred = np.asarray(model.predict_with_weights(x_data, frame.weights), dtype=float).ravel()
    residuals = y_actual - y_pred
    x_visual = _visual_x_values(x_data, visual_feature_index)

    x_curve, y_curve = _prediction_curve(model, x_data, frame.weights, visual_feature_index)
    artists.line_pred.set_data(x_curve, y_curve)

    if artists.residual_lines is not None:
        segments = [
            [(float(xi), float(yi)), (float(xi), float(ypi))]
            for xi, yi, ypi in zip(x_visual, y_actual, y_pred)
        ]
        artists.residual_lines.set_segments(segments)

    epochs = np.array([item.epoch for item in frames], dtype=float)
    losses = np.array([item.loss for item in frames], dtype=float)
    progress_epochs = epochs[: frame_index + 1]
    progress_losses = losses[: frame_index + 1]
    artists.line_loss_progress.set_data(progress_epochs, progress_losses)

    current_epoch = float(epochs[frame_index])
    current_loss = float(frame.loss)
    artists.loss_marker.set_xdata([current_epoch, current_epoch])
    artists.loss_point.set_offsets(np.array([[current_epoch, current_loss]]))

    total_epochs = int(epochs[-1])
    progress_pct = (frame.epoch / total_epochs * 100.0) if total_epochs > 0 else 0.0
    artists.loss_annotation.set_text(
        "Epoch: {:d} / {:d}\nMSE: {:.4e}\nProgress: {:.1f}%".format(
            frame.epoch, total_epochs, current_loss, progress_pct
        )
    )

    visual_name = feature_names[visual_feature_index] if feature_names else "x"
    ax_fit.set_xlim(*artists.fit_xlim)
    ax_fit.set_ylim(*artists.fit_ylim)
    ax_fit.set_title(
        "Model Fit  ·  Epoch {:d}/{:d}".format(frame.epoch, total_epochs),
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    ax_loss.set_xlim(*artists.loss_xlim)
    ax_loss.set_ylim(*artists.loss_ylim)
    ax_loss.set_title(
        "Training Loss  ·  MSE = {:.4e}".format(current_loss),
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    fig.canvas.draw_idle()
    equation = model.equation(frame.weights, feature_names)
    return TrainingFrameView(
        equation=equation,
        epoch=frame.epoch,
        loss=frame.loss,
        weights=frame.weights.copy(),
        y_pred=y_pred,
        residuals=residuals,
        x_visual=x_visual,
        y_actual=y_actual,
    )
