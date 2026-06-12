"""
Owner: Thành viên 2 - Data Input / CSV / Sample.

File này xử lý input data.

Không import Tkinter.
Không import Matplotlib.
Không train model trong file này.

Input:
    - CSV file path.
    - Raw text dạng `x,y`.
    - Feature names và target name.

Output:
    - numpy arrays x, y đã clean.
    - danh sách cột numeric.
    - dict numeric_data.

TODO:
    - Copy các hàm đọc CSV và parse text từ prototype vào đây.
"""

from __future__ import annotations

import numpy as np


def read_csv_numeric_columns(file_path: str) -> tuple[list[str], dict[str, np.ndarray]]:
    """
    Đọc CSV và lấy các cột numeric.

    Input:
        file_path: Đường dẫn file CSV.

    Output:
        numeric_names: Danh sách tên cột numeric.
        numeric_data: Dict tên cột -> numpy array, có thể có NaN.
    """
    raise NotImplementedError("Thành viên 2 copy hàm read_csv_numeric_columns từ prototype.")


def arrays_from_csv_selection(
    numeric_data: dict[str, np.ndarray],
    feature_names: list[str],
    target_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lấy X, y từ các cột user chọn.

    Input:
        numeric_data: Dict tên cột -> numpy array.
        feature_names: Các cột làm X.
        target_name: Cột làm y.

    Output:
        x: Matrix feature, shape (n_samples, n_features).
        y: Target vector, shape (n_samples,).

    Rule:
        - Bỏ dòng nào có NaN.
        - Nếu còn ít hơn 3 dòng thì raise ValueError.
    """
    raise NotImplementedError("Thành viên 2 copy hàm arrays_from_csv_selection từ prototype.")


def parse_xy_data(raw_text: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse text data dạng `x,y`.

    Input:
        raw_text: Nhiều dòng, mỗi dòng có dạng `x,y` hoặc `x;y`.

    Output:
        x: numpy array 1D.
        y: numpy array 1D.

    Rule:
        - Cho phép dòng header `x,y`.
        - Cần ít nhất 3 điểm dữ liệu.
        - Dòng sai format thì raise ValueError có nói rõ line number.
    """
    raise NotImplementedError("Thành viên 2 copy hàm parse_xy_data từ prototype.")

