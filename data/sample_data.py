"""
Owner: Thành viên 2 - Data Input / CSV / Sample.

File này tạo dataset mẫu để demo.

Không import Tkinter.
Không import Matplotlib.

Input:
    - Tên sample dataset.

Output:
    - Chuỗi text có format CSV mini: header `x,y` và các dòng data.

TODO:
    - Copy `make_sample_dataset`, `make_sample_data`, `format_xy_rows` từ prototype vào đây.
"""

from __future__ import annotations

import numpy as np


SAMPLE_DATASET_NAMES = (
    "Linear",
    "Quadratic",
    "Cubic",
    "Quartic",
    "Sine wave",
    "Wide quadratic",
)


def format_xy_rows(x: np.ndarray, y: np.ndarray) -> str:
    """
    Convert x, y arrays thành text.

    Input:
        x: numpy array 1D.
        y: numpy array 1D.

    Output:
        Chuỗi có header `x,y`.
    """
    return "x,y\n" + "\n".join(f"{xi:.3f},{yi:.3f}" for xi, yi in zip(x, y))


def make_sample_dataset(name: str) -> str:
    """
    Tạo sample dataset theo tên.

    Input:
        name: Một giá trị trong SAMPLE_DATASET_NAMES.

    Output:
        Chuỗi data dạng x,y để đưa vào Text widget.
    """
    rng = np.random.default_rng(7)
    x = np.linspace(-5, 5, 34)

    if name == "Linear":
        y = 3.2 * x + 7 + rng.normal(0, 1.1, size=len(x))
    elif name == "Quadratic":
        y = 1.4 * x**2 - 2.2 * x + 4 + rng.normal(0, 1.8, size=len(x))
    elif name == "Cubic":
        y = 0.35 * x**3 - 1.8 * x**2 + 2.4 * x + 8 + rng.normal(0, 4.2, size=len(x))
    elif name == "Quartic":
        y = -0.18 * x**4 + 0.75 * x**3 + 2.4 * x**2 - 5 * x + 15 + rng.normal(0, 5.5, size=len(x))
    elif name == "Sine wave":
        y = 11 * np.sin(1.25 * x) + 0.9 * x + rng.normal(0, 1.0, size=len(x))
    elif name == "Wide quadratic":
        x = np.linspace(-12, 12, 42)
        y = 0.42 * x**2 - 1.3 * x + 6 + rng.normal(0, 3.8, size=len(x))
    else:
        raise ValueError(f"Unknown sample dataset: {name}")

    return format_xy_rows(x, y)


def make_sample_data() -> str:
    """
    Tạo sample mặc định.

    Input:
        Không có.

    Output:
        Chuỗi data sample mặc định, nên dùng Cubic.
    """
    return make_sample_dataset("Cubic")
