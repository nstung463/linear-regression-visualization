"""
Owner: Thành viên 4 - GUI / Integration.

Bridge between GUI, data, model and visualization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.model import PolynomialRegressionGD
from data.csv_loader import (
    arrays_from_csv_selection,
    describe_numeric,
    read_csv,
    read_csv_numeric_columns,
)
from data.csv_loader import parse_xy_data


@dataclass(frozen=True)
class TrainingResult:
    """Output after training completes."""

    model: PolynomialRegressionGD
    x_data: np.ndarray
    y_data: np.ndarray
    feature_names: list[str]
    visual_feature_index: int


@dataclass(frozen=True)
class CsvPreviewData:
    """Raw CSV data loaded for preview."""

    path: str
    rows: list[dict[str, str]]
    numeric_names: list[str]
    numeric_data: dict[str, np.ndarray]


@dataclass(frozen=True)
class CsvPreviewResult:
    """Prepared arrays and stats for CSV scatter preview."""

    x_visual: np.ndarray
    y_target: np.ndarray
    n_valid: int
    n_dropped: int
    column_stats: dict[str, dict[str, float]]


def _validate_train_params(degree: int, lr: float, epochs: int) -> None:
    if degree < 1:
        raise ValueError("Degree must be at least 1.")
    if lr <= 0:
        raise ValueError("Learning rate must be positive.")
    if epochs <= 0:
        raise ValueError("Epochs must be positive.")


def _fit_model(
    x_data: np.ndarray,
    y_data: np.ndarray,
    degree: int,
    lr: float,
    epochs: int,
    save_every: int = 1,
) -> PolynomialRegressionGD:
    model = PolynomialRegressionGD(degree=degree, lr=lr, epochs=epochs, save_every=save_every)
    model.fit(x_data, y_data)
    return model


def load_csv_for_preview(path: str) -> CsvPreviewData:
    """Read a CSV file and extract numeric columns for preview."""
    rows = read_csv(path)
    numeric_names, numeric_data = read_csv_numeric_columns(path)
    if not numeric_names:
        raise ValueError("No numeric columns found in the CSV file.")
    return CsvPreviewData(
        path=path,
        rows=rows,
        numeric_names=numeric_names,
        numeric_data=numeric_data,
    )


def prepare_csv_preview(
    numeric_data: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    feature_names: list[str],
    target_name: str,
    visual_name: str,
) -> CsvPreviewResult:
    """Build preview arrays and column statistics without training."""
    if visual_name not in numeric_data:
        raise ValueError("Visual column not found: {}".format(visual_name))
    if target_name not in numeric_data:
        raise ValueError("Target column not found: {}".format(target_name))

    x_visual_raw = np.asarray(numeric_data[visual_name], dtype=float)
    y_target_raw = np.asarray(numeric_data[target_name], dtype=float)

    if feature_names:
        x_stack = np.column_stack([numeric_data[name] for name in feature_names])
        keep = (
            np.isfinite(x_visual_raw)
            & np.isfinite(y_target_raw)
            & np.all(np.isfinite(x_stack), axis=1)
        )
    else:
        keep = np.isfinite(x_visual_raw) & np.isfinite(y_target_raw)

    n_total = len(y_target_raw)
    n_valid = int(keep.sum())
    n_dropped = n_total - n_valid

    columns_for_stats = list(dict.fromkeys([visual_name, target_name] + feature_names))
    column_stats = describe_numeric(rows, columns_for_stats)

    return CsvPreviewResult(
        x_visual=x_visual_raw[keep],
        y_target=y_target_raw[keep],
        n_valid=n_valid,
        n_dropped=n_dropped,
        column_stats=column_stats,
    )


def train_from_text(
    raw_text: str,
    degree: int,
    lr: float,
    epochs: int,
    save_every: int = 1,
) -> TrainingResult:
    """Train a model from x,y text input."""
    _validate_train_params(degree, lr, epochs)
    x_values, y_values = parse_xy_data(raw_text)
    x_data = x_values.reshape(-1, 1)
    model = _fit_model(x_data, y_values, degree, lr, epochs, save_every)
    return TrainingResult(
        model=model,
        x_data=x_data,
        y_data=y_values,
        feature_names=["x"],
        visual_feature_index=0,
    )


def train_from_csv_selection(
    numeric_data: dict[str, np.ndarray],
    feature_names: list[str],
    target_name: str,
    visual_name: str,
    degree: int,
    lr: float,
    epochs: int,
    save_every: int = 1,
) -> TrainingResult:
    """Train a model from selected CSV columns."""
    _validate_train_params(degree, lr, epochs)
    if visual_name not in feature_names:
        raise ValueError("Visual column must be included in feature columns.")

    x_data, y_data = arrays_from_csv_selection(numeric_data, feature_names, target_name)
    if len(y_data) < 2:
        raise ValueError("Need at least 2 valid rows after filtering.")

    model = _fit_model(x_data, y_data, degree, lr, epochs, save_every)
    visual_feature_index = feature_names.index(visual_name)
    return TrainingResult(
        model=model,
        x_data=x_data,
        y_data=y_data,
        feature_names=feature_names,
        visual_feature_index=visual_feature_index,
    )
