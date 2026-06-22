"""
Owner: Thành viên 4 - GUI / Integration.

Bridge between GUI, data, model and visualization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.metrics import regression_metrics
from core.model import PolynomialRegressionGD
from core.preprocessing import (
    PreprocessOptions,
    PreprocessReport,
    drop_missing_rows,
    preprocess_dataset,
    remove_outliers_regression_residual,
    split_train_test,
)
from data.csv_loader import (
    arrays_from_csv_selection,
    describe_numeric,
    read_csv,
    read_csv_numeric_columns,
)
from data.csv_loader import parse_xy_data


@dataclass(frozen=True)
class TestEvaluationResult:
    """Metrics and predictions on a held-out test CSV."""

    x_test: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    metrics: dict[str, float]
    target_scale: float | None = None


@dataclass(frozen=True)
class TrainingResult:
    """Output after training completes."""

    model: PolynomialRegressionGD
    x_data: np.ndarray
    y_data: np.ndarray
    feature_names: list[str]
    target_name: str
    visual_feature_index: int
    preprocess_report: PreprocessReport | None = None
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    test_evaluation: TestEvaluationResult | None = None


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


@dataclass(frozen=True)
class PreprocessResult:
    """Cleaned training arrays after preprocessing."""

    x_data: np.ndarray
    y_data: np.ndarray
    report: PreprocessReport
    feature_names: list[str]
    target_name: str
    visual_name: str
    x_visual_raw: np.ndarray
    y_target_raw: np.ndarray
    target_scale: float | None = None


def _validate_train_params(degree: int, lr: float, epochs: int) -> None:
    if degree < 1:
        raise ValueError("Degree must be at least 1.")
    if lr <= 0:
        raise ValueError("Learning rate must be positive.")
    if epochs <= 0:
        raise ValueError("Epochs must be positive.")


def _raw_arrays_from_csv(
    numeric_data: dict[str, np.ndarray],
    feature_names: list[str],
    target_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not feature_names:
        raise ValueError("At least one feature column is required.")
    for name in feature_names:
        if name not in numeric_data:
            raise ValueError("Feature column not found: {}".format(name))
    if target_name not in numeric_data:
        raise ValueError("Target column not found: {}".format(target_name))

    x_values = np.column_stack([numeric_data[name] for name in feature_names])
    y_values = np.asarray(numeric_data[target_name], dtype=float)
    return x_values, y_values


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


def _clean_csv_arrays(
    x_data: np.ndarray,
    y_data: np.ndarray,
    iqr_multiplier: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, PreprocessReport]:
    """Drop non-finite rows and regression-residual outliers before training."""
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float).ravel()
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n_original = x.shape[0]

    x, y, missing_mask = drop_missing_rows(x, y)
    n_dropped_missing = int(missing_mask.sum())

    outlier_mask = np.zeros(n_original, dtype=bool)
    n_dropped_outliers = 0
    if len(y) > 0:
        x, y, outlier_mask_partial = remove_outliers_regression_residual(x, y, iqr_multiplier)
        n_dropped_outliers = int(outlier_mask_partial.sum())
        if n_dropped_missing > 0:
            surviving = ~missing_mask
            full_outlier_mask = np.zeros(n_original, dtype=bool)
            full_outlier_mask[np.where(surviving)[0][outlier_mask_partial]] = True
            outlier_mask = full_outlier_mask
        else:
            outlier_mask = outlier_mask_partial

    report = PreprocessReport(
        n_original=n_original,
        n_dropped_missing=n_dropped_missing,
        n_dropped_outliers=n_dropped_outliers,
        n_final=len(y),
        missing_mask=missing_mask,
        outlier_mask=outlier_mask,
        target_normalized=False,
        target_scale=None,
    )
    return x, y, report


def _evaluate_holdout(
    model: PolynomialRegressionGD,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_scale: float | None = None,
) -> TestEvaluationResult:
    weights = model.frames[-1].weights
    y_pred = model.predict_with_weights(x_test, weights)
    metrics = regression_metrics(y_test, y_pred)
    return TestEvaluationResult(
        x_test=x_test,
        y_test=y_test,
        y_pred=y_pred,
        metrics=metrics,
        target_scale=target_scale,
    )


def _train_with_holdout(
    x_data: np.ndarray,
    y_data: np.ndarray,
    degree: int,
    lr: float,
    epochs: int,
    save_every: int,
    target_scale: float | None = None,
) -> tuple[
    PolynomialRegressionGD,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    TestEvaluationResult | None,
]:
    """Train on 80% and evaluate on a 20% hold-out split."""
    if len(y_data) < 5:
        model = _fit_model(x_data, y_data, degree, lr, epochs, save_every)
        return model, x_data, y_data, None, None, None

    x_train, x_test, y_train, y_test = split_train_test(x_data, y_data)
    model = _fit_model(x_train, y_train, degree, lr, epochs, save_every)
    evaluation = _evaluate_holdout(model, x_test, y_test, target_scale=target_scale)
    return model, x_train, y_train, x_test, y_test, evaluation


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


def preprocess_csv_selection(
    numeric_data: dict[str, np.ndarray],
    feature_names: list[str],
    target_name: str,
    visual_name: str,
    options: PreprocessOptions,
) -> PreprocessResult:
    """Clean selected CSV columns before training."""
    if visual_name not in feature_names:
        raise ValueError("Visual column must be included in feature columns.")

    x_data, y_data = _raw_arrays_from_csv(numeric_data, feature_names, target_name)
    x_visual_raw = np.asarray(numeric_data[visual_name], dtype=float)
    y_target_raw = np.asarray(numeric_data[target_name], dtype=float)

    x_clean, y_clean, report = preprocess_dataset(x_data, y_data, options)
    if len(y_clean) < 2:
        raise ValueError("Need at least 2 rows after preprocessing.")

    return PreprocessResult(
        x_data=x_clean,
        y_data=y_clean,
        report=report,
        feature_names=feature_names,
        target_name=target_name,
        visual_name=visual_name,
        x_visual_raw=x_visual_raw,
        y_target_raw=y_target_raw,
        target_scale=report.target_scale,
    )


def evaluate_on_test_csv(
    model: PolynomialRegressionGD,
    test_path: str,
    feature_names: list[str],
    target_name: str,
    target_scale: float | None = None,
) -> TestEvaluationResult:
    """Load a test CSV and evaluate the trained model."""
    _, numeric_data = read_csv_numeric_columns(test_path)
    x_test, y_test = arrays_from_csv_selection(numeric_data, feature_names, target_name)
    if len(y_test) == 0:
        raise ValueError("Test CSV has no valid rows for the selected columns.")

    if target_scale is not None:
        y_test = y_test / target_scale

    weights = model.frames[-1].weights
    y_pred = model.predict_with_weights(x_test, weights)
    metrics = regression_metrics(y_test, y_pred)
    return TestEvaluationResult(
        x_test=x_test,
        y_test=y_test,
        y_pred=y_pred,
        metrics=metrics,
        target_scale=target_scale,
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
    model, x_train, y_train, x_test, y_test, test_evaluation = _train_with_holdout(
        x_data, y_values, degree, lr, epochs, save_every
    )
    return TrainingResult(
        model=model,
        x_data=x_train,
        y_data=y_train,
        feature_names=["x"],
        target_name="y",
        visual_feature_index=0,
        x_test=x_test,
        y_test=y_test,
        test_evaluation=test_evaluation,
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
    preprocessed: PreprocessResult | None = None,
    iqr_multiplier: float = 1.5,
) -> TrainingResult:
    """Train a model from selected CSV columns."""
    _validate_train_params(degree, lr, epochs)
    if visual_name not in feature_names:
        raise ValueError("Visual column must be included in feature columns.")

    preprocess_report: PreprocessReport | None = None
    if preprocessed is not None:
        x_data = preprocessed.x_data
        y_data = preprocessed.y_data
        preprocess_report = preprocessed.report
    else:
        x_raw, y_raw = arrays_from_csv_selection(numeric_data, feature_names, target_name)
        x_data, y_data, preprocess_report = _clean_csv_arrays(x_raw, y_raw, iqr_multiplier)

    if len(y_data) < 2:
        raise ValueError("Need at least 2 valid rows after filtering.")

    target_scale = None
    if preprocess_report and preprocess_report.target_normalized:
        target_scale = preprocess_report.target_scale

    model, x_train, y_train, x_test, y_test, test_evaluation = _train_with_holdout(
        x_data, y_data, degree, lr, epochs, save_every, target_scale=target_scale
    )
    visual_feature_index = feature_names.index(visual_name)

    return TrainingResult(
        model=model,
        x_data=x_train,
        y_data=y_train,
        feature_names=feature_names,
        target_name=target_name,
        visual_feature_index=visual_feature_index,
        preprocess_report=preprocess_report,
        x_test=x_test,
        y_test=y_test,
        test_evaluation=test_evaluation,
    )
