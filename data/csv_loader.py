"""CSV and text input helpers for the linear-regression visualizer.

Scope for Thanh vien 2:
- Parse text data in x,y format.
- Read CSV files.
- Detect/filter numeric columns.
- Convert selected feature/target columns into numpy arrays.
"""

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def parse_xy_data(raw_text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse text in x,y format into numpy arrays.

    Empty lines and comment lines starting with # are ignored. A header line
    such as x,y is allowed.
    """
    x_values = []
    y_values = []

    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            raise ValueError("Line {} must be in x,y format.".format(line_number))

        if parts[0].lower() == "x" and parts[1].lower() == "y":
            continue

        try:
            x_values.append(float(parts[0]))
            y_values.append(float(parts[1]))
        except ValueError as exc:
            raise ValueError("Line {} contains non-numeric data.".format(line_number)) from exc

    if len(x_values) < 2:
        raise ValueError("Need at least 2 data points.")

    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def parse_xy_text(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for parse_xy_data."""
    return parse_xy_data(text)


def read_csv(path: str) -> List[Dict[str, str]]:
    """Read a CSV file into a list of row dictionaries."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError("CSV file not found: {}".format(csv_path))

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return [dict(row) for row in csv.DictReader(file_obj)]


def to_float(value: object) -> float:
    """Convert a value to float, returning NaN when conversion fails."""
    if value is None:
        return np.nan
    text = str(value).strip().replace(",", "")
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def numeric_columns(
    rows: Sequence[Dict[str, str]], min_valid_ratio: float = 0.8
) -> List[str]:
    """Return columns that are numeric for at least min_valid_ratio rows."""
    if not rows:
        return []

    result = []
    for column in rows[0].keys():
        values = np.asarray([to_float(row.get(column)) for row in rows], dtype=float)
        if float(np.isfinite(values).mean()) >= min_valid_ratio:
            result.append(column)
    return result


def read_csv_numeric_columns(file_path: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Read a CSV file and return all numeric columns as numpy arrays."""
    rows = read_csv(file_path)
    names = numeric_columns(rows)
    data = {
        name: np.asarray([to_float(row.get(name)) for row in rows], dtype=float)
        for name in names
    }
    return names, data


def arrays_from_csv_selection(
    numeric_data: Dict[str, np.ndarray],
    feature_names: List[str],
    target_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert selected feature/target columns to clean numpy arrays."""
    if not feature_names:
        raise ValueError("At least one feature column is required.")
    for name in feature_names:
        if name not in numeric_data:
            raise ValueError("Feature column not found: {}".format(name))
    if target_name not in numeric_data:
        raise ValueError("Target column not found: {}".format(target_name))

    x_values = np.column_stack([numeric_data[name] for name in feature_names])
    y_values = np.asarray(numeric_data[target_name], dtype=float)
    keep = np.isfinite(y_values) & np.all(np.isfinite(x_values), axis=1)
    return x_values[keep], y_values[keep]


def feature_target_arrays(
    path: str, feature: str = "total_sqft", target: str = "price"
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a CSV and return one feature column and one target column."""
    numeric_names, numeric_data = read_csv_numeric_columns(path)
    if feature not in numeric_names:
        raise ValueError("Feature column is not numeric: {}".format(feature))
    if target not in numeric_names:
        raise ValueError("Target column is not numeric: {}".format(target))
    return arrays_from_csv_selection(numeric_data, [feature], target)


def describe_numeric(
    rows: Sequence[Dict[str, str]], columns: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    """Return simple summary statistics for numeric columns."""
    summary = {}
    for column in columns:
        values = np.asarray([to_float(row.get(column)) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        summary[column] = {
            "count": float(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parents[1] / "Bengaluru_House_Data.csv"
    if not csv_path.exists():
        print("Dataset not found:", csv_path)
        print("Place Bengaluru_House_Data.csv in the project root to preview CSV loading.")
        raise SystemExit(0)

    rows = read_csv(str(csv_path))
    numeric_names, numeric_data = read_csv_numeric_columns(str(csv_path))
    x_values, y_values = arrays_from_csv_selection(numeric_data, ["total_sqft"], "price")

    print("Dataset:", csv_path)
    print("Rows:", len(rows))
    print("Numeric columns:", ", ".join(numeric_names))
    print("Example feature/target arrays:")
    print("  feature: total_sqft")
    print("  target: price")
    print("  X shape:", x_values.shape)
    print("  y shape:", y_values.shape)
    print("Numeric summary:")
    for column, stats in describe_numeric(rows, numeric_names).items():
        print(
            "  {column}: count={count:.0f}, mean={mean:.2f}, std={std:.2f}, "
            "min={min:.2f}, max={max:.2f}".format(column=column, **stats)
        )
