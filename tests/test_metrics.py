"""Tests for regression metrics."""

from __future__ import annotations

import numpy as np

from core.metrics import evaluate_model, regression_metrics
from core.model import PolynomialRegressionGD


def test_regression_metrics_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = regression_metrics(y, y)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_evaluate_model_uses_final_weights() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
    y = np.array([2.0, 4.0, 6.0, 8.0])
    model = PolynomialRegressionGD(degree=1, lr=0.05, epochs=200, save_every=50)
    model.fit(x, y)
    metrics = evaluate_model(model, x, y)
    assert metrics["r2"] > 0.99
