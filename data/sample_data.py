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
    raise NotImplementedError("Thành viên 2 copy hàm format_xy_rows từ prototype.")


def make_sample_dataset(name: str) -> str:
    """
    Tạo sample dataset theo tên.

    Input:
        name: Một giá trị trong SAMPLE_DATASET_NAMES.

    Output:
        Chuỗi data dạng x,y để đưa vào Text widget.
    """
    raise NotImplementedError("Thành viên 2 copy hàm make_sample_dataset từ prototype.")


def make_sample_data() -> str:
    """
    Tạo sample mặc định.

    Input:
        Không có.

    Output:
        Chuỗi data sample mặc định, nên dùng Cubic.
    """
    raise NotImplementedError("Thành viên 2 copy hàm make_sample_data từ prototype.")

