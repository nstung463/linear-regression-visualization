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

