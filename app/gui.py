"""
Owner: Thành viên 4 - GUI / Integration.

File này phụ trách giao diện Tkinter.

Không viết thuật toán Gradient Descent trong file này.
Không viết logic đọc CSV chi tiết trong file này.
Không viết logic vẽ plot chi tiết trong file này.

Input:
    - Text data `x,y` từ Text widget.
    - CSV file path từ filedialog.
    - Degree, learning rate, epochs, speed từ Spinbox.
    - Target, features, visual x từ Combobox/Listbox.

Output:
    - App window.
    - Gọi controller để train model.
    - Gọi visualization để cập nhật plot.

State cần quản lý:
    - self.model
    - self.x_data
    - self.y_data
    - self.feature_names
    - self.visual_feature_index
    - self.frame_index
    - self.playing

TODO:
    - Copy layout Tkinter từ prototype vào đây.
    - Thay các đoạn xử lý trực tiếp bằng hàm trong `app.controller`.
    - Thay đoạn vẽ plot bằng hàm trong `visualization.plotter`.
"""

from __future__ import annotations

import tkinter as tk


class PolynomialRegressionApp(tk.Tk):
    """
    Main GUI application.

    Input:
        Hành động của người dùng từ Tkinter widgets.

    Output:
        Ứng dụng desktop tương tác được.
    """

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("Thành viên 4 sẽ copy và tách UI từ prototype vào đây.")

