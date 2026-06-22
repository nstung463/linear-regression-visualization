"""Tests for preprocessing and CSV pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.controller import evaluate_on_test_csv, preprocess_csv_selection, train_from_csv_selection
from core.preprocessing import PreprocessOptions, normalize_target, preprocess_dataset
from data.csv_loader import read_csv_numeric_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data_csv" / "house_price_train_linear.csv"
TEST_PATH = PROJECT_ROOT / "data_csv" / "house_price_test_linear.csv"


def test_normalize_target_divides_by_scale() -> None:
    y = np.array([1e9, 2e9, 3e9])
    y_norm, scale = normalize_target(y, 1e9)
    assert np.allclose(y_norm, [1.0, 2.0, 3.0])
    assert scale == 1e9


def test_preprocess_with_normalize_applies_scaling() -> None:
    x = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1e9, 2e9, 3e9])
    options = PreprocessOptions(
        drop_missing=False,
        remove_outliers=False,
        normalize_target=True,
        target_scale=1e9,
    )
    _, y_clean, report = preprocess_dataset(x, y, options)
    assert report.target_normalized is True
    assert report.target_scale == 1e9
    assert np.allclose(y_clean, [1.0, 2.0, 3.0])


def test_preprocess_train_csv_drops_missing_and_outliers() -> None:
    assert TRAIN_PATH.exists(), "Train preset CSV is required for this test."
    _, numeric_data = read_csv_numeric_columns(str(TRAIN_PATH))
    options = PreprocessOptions(drop_missing=True, remove_outliers=True, iqr_multiplier=1.5, normalize_target=True)
    result = preprocess_csv_selection(
        numeric_data,
        ["Dien_Tich_m2"],
        "Gia_Nha_VND",
        "Dien_Tich_m2",
        options,
    )
    report = result.report
    assert report.n_original == 100
    assert report.n_dropped_missing == 5
    assert report.n_dropped_outliers == 5
    assert report.n_final == 90
    assert len(result.y_data) == 90
    assert report.target_normalized is True


def test_preprocess_dataset_masks_align_with_original_rows() -> None:
    x = np.array([[1.0], [2.0], [3.0], [100.0], [4.0]])
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    options = PreprocessOptions(drop_missing=True, remove_outliers=True, iqr_multiplier=1.5, normalize_target=False)
    _, _, report = preprocess_dataset(x, y, options)
    assert report.n_original == 5
    assert report.n_dropped_missing == 1
    assert bool(report.missing_mask[2])


def test_train_and_evaluate_on_house_price_presets() -> None:
    assert TRAIN_PATH.exists() and TEST_PATH.exists()
    _, train_data = read_csv_numeric_columns(str(TRAIN_PATH))
    options = PreprocessOptions(drop_missing=True, remove_outliers=True, iqr_multiplier=1.5, normalize_target=True)
    preprocessed = preprocess_csv_selection(
        train_data,
        ["Dien_Tich_m2"],
        "Gia_Nha_VND",
        "Dien_Tich_m2",
        options,
    )
    training = train_from_csv_selection(
        train_data,
        ["Dien_Tich_m2"],
        "Gia_Nha_VND",
        "Dien_Tich_m2",
        degree=1,
        lr=0.03,
        epochs=300,
        save_every=50,
        preprocessed=preprocessed,
    )
    evaluation = evaluate_on_test_csv(
        training.model,
        str(TEST_PATH),
        ["Dien_Tich_m2"],
        "Gia_Nha_VND",
        target_scale=preprocessed.target_scale,
    )
    metrics = evaluation.metrics
    assert metrics["r2"] > 0.99
    assert len(evaluation.y_pred) == 10


def test_evaluate_on_test_csv_shape() -> None:
    assert TRAIN_PATH.exists() and TEST_PATH.exists()
    _, train_data = read_csv_numeric_columns(str(TRAIN_PATH))
    preprocessed = preprocess_csv_selection(
        train_data,
        ["Dien_Tich_m2"],
        "Gia_Nha_VND",
        "Dien_Tich_m2",
        PreprocessOptions(normalize_target=True),
    )
    training = train_from_csv_selection(
        train_data,
        ["Dien_Tich_m2"],
        "Gia_Nha_VND",
        "Dien_Tich_m2",
        degree=1,
        lr=0.03,
        epochs=100,
        save_every=25,
        preprocessed=preprocessed,
    )
    evaluation = evaluate_on_test_csv(
        training.model,
        str(TEST_PATH),
        ["Dien_Tich_m2"],
        "Gia_Nha_VND",
        target_scale=preprocessed.target_scale,
    )
    assert evaluation.x_test.shape == (10, 1)
    assert evaluation.y_test.shape == (10,)
    assert evaluation.y_pred.shape == (10,)
