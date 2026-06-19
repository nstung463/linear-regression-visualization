"""
Owner: Thành viên 1 - Core AI / Model.

File này chứa các hàm tiền xử lý dữ liệu cho model.

Không import Tkinter.
Không import Matplotlib.

Input:
    - numpy array cần scale.

Output:
    - array đã standardize.
    - mean.
    - std.

TODO:
    - Copy hàm `standardize` từ prototype vào đây.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreprocessOptions:
    """Options for dataset cleaning before training."""

    drop_missing: bool = True
    remove_outliers: bool = True
    iqr_multiplier: float = 1.5
    normalize_target: bool = True
    target_scale: float = 1e9


@dataclass(frozen=True)
class PreprocessReport:
    """Summary of rows removed during preprocessing."""

    n_original: int
    n_dropped_missing: int
    n_dropped_outliers: int
    n_final: int
    missing_mask: np.ndarray
    outlier_mask: np.ndarray
    target_normalized: bool
    target_scale: float | None


def drop_missing_rows(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove rows with non-finite values in features or target."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    keep = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    return x[keep], y[keep], ~keep


def remove_outliers_iqr(
    x: np.ndarray,
    y: np.ndarray,
    multiplier: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove rows outside the IQR fence on any feature or target column."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    combined = np.column_stack([x, y.reshape(-1, 1)])
    keep = np.ones(combined.shape[0], dtype=bool)
    for col in range(combined.shape[1]):
        values = combined[:, col]
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        keep &= (values >= lower) & (values <= upper)

    return x[keep], y[keep], ~keep


def normalize_target(y: np.ndarray, scale: float = 1e9) -> tuple[np.ndarray, float]:
    """Normalize target by dividing by scale (default 1 billion for VND)."""
    y = np.asarray(y, dtype=float)
    return y / scale, scale


def preprocess_dataset(
    x: np.ndarray,
    y: np.ndarray,
    options: PreprocessOptions,
) -> tuple[np.ndarray, np.ndarray, PreprocessReport]:
    """Apply configured cleaning steps and return cleaned arrays with a report."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n_original = x.shape[0]

    missing_mask = np.zeros(n_original, dtype=bool)
    outlier_mask = np.zeros(n_original, dtype=bool)
    n_dropped_missing = 0
    n_dropped_outliers = 0

    if options.drop_missing:
        x, y, missing_mask = drop_missing_rows(x, y)
        n_dropped_missing = int(missing_mask.sum())

    if options.remove_outliers and len(y) > 0:
        x, y, outlier_mask_partial = remove_outliers_iqr(x, y, options.iqr_multiplier)
        n_dropped_outliers = int(outlier_mask_partial.sum())
        if n_dropped_missing > 0:
            surviving = ~missing_mask
            full_outlier_mask = np.zeros(n_original, dtype=bool)
            full_outlier_mask[np.where(surviving)[0][outlier_mask_partial]] = True
            outlier_mask = full_outlier_mask
        else:
            outlier_mask = outlier_mask_partial

    target_normalized = False
    actual_scale = None
    if options.normalize_target and len(y) > 0:
        y, actual_scale = normalize_target(y, options.target_scale)
        target_normalized = True

    report = PreprocessReport(
        n_original=n_original,
        n_dropped_missing=n_dropped_missing,
        n_dropped_outliers=n_dropped_outliers,
        n_final=len(y),
        missing_mask=missing_mask,
        outlier_mask=outlier_mask,
        target_normalized=target_normalized,
        target_scale=actual_scale,
    )
    return x, y, report


def standardize(values: np.ndarray) -> tuple[np.ndarray, float | np.ndarray, float | np.ndarray]:
    """
    Standardize values theo công thức: z = (x - mean) / std.

    Input:
        values: numpy array 1D hoặc 2D.

    Output:
        scaled_values: array sau khi scale.
        mean: mean của từng cột hoặc scalar.
        std: std của từng cột hoặc scalar. Nếu std = 0 thì đổi thành 1.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        mean = values.mean()
        std = values.std(ddof=0)
        if std == 0:
            std = 1.0
        return (values - mean) / std, mean, std

    if values.ndim == 2:
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=0)
        std = np.where(std == 0, 1.0, std)
        return (values - mean) / std, mean, std

    raise ValueError("standardize only supports 1D or 2D numpy arrays.")

