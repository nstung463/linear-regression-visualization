"""
Owner: Thành viên 3 - Visualization / Plot / Animation.

File này phụ trách vẽ Matplotlib.

Không đọc CSV trong file này.
Không train model trong file này.
Không tạo Tkinter widget mới trong file này, chỉ nhận axes/canvas từ GUI.

Input:
    - model đã train.
    - x_data, y_data.
    - frame_index.
    - feature_names.
    - visual_feature_index.
    - ax_fit, ax_loss của Matplotlib.

Output:
    - Cập nhật axes để GUI redraw canvas.

TODO:
    - Copy `_draw_empty_plot` và `draw_frame` từ prototype.
    - Tách thành function độc lập.
"""

from __future__ import annotations

import numpy as np


def draw_empty_plot(ax_fit: object, ax_loss: object) -> None:
    """
    Vẽ plot rỗng khi chưa train.

    Input:
        ax_fit: Matplotlib axis bên trái.
        ax_loss: Matplotlib axis bên phải.

    Output:
        Axes được clear và set title/label/grid.
    """
    raise NotImplementedError("Thành viên 3 copy logic _draw_empty_plot từ prototype.")


def draw_training_frame(
    model: object,
    x_data: np.ndarray,
    y_data: np.ndarray,
    frame_index: int,
    feature_names: list[str],
    visual_feature_index: int,
    ax_fit: object,
    ax_loss: object,
) -> None:
    """
    Vẽ một frame training.

    Input:
        model: Model đã train, có `frames`, `predict_with_weights`, `equation`.
        x_data: Feature data.
        y_data: Target data.
        frame_index: Index frame cần vẽ.
        feature_names: Tên các feature.
        visual_feature_index: Feature dùng làm trục x khi vẽ.
        ax_fit: Axis vẽ data và prediction.
        ax_loss: Axis vẽ loss curve.

    Output:
        Hai axes được cập nhật.
    """
    raise NotImplementedError("Thành viên 3 copy và tách logic draw_frame từ prototype.")

