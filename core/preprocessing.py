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

import numpy as np


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
    raise NotImplementedError("Thành viên 1 copy và kiểm tra hàm standardize từ prototype.")

