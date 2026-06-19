"""
Regression evaluation metrics on original-scale targets.
"""

from __future__ import annotations

import numpy as np

from core.model import PolynomialRegressionGD


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, and R² on original-scale values."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    errors = y_true - y_pred
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))

    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {"rmse": rmse, "mae": mae, "r2": r2, "mse": mse}


def evaluate_model(model: PolynomialRegressionGD, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Evaluate the final trained weights on a dataset."""
    if not model.frames:
        raise ValueError("Model has no training frames.")
    weights = model.frames[-1].weights
    y_pred = model.predict_with_weights(x, weights)
    return regression_metrics(y, y_pred)
