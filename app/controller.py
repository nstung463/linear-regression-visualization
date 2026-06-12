"""
Owner: Thành viên 4 - GUI / Integration.

File này làm cầu nối giữa GUI, data, model và visualization.

Lý do có controller:
    - GUI không bị quá dài.
    - Logic train/import/validate tập trung một chỗ.
    - Dễ merge code của các thành viên hơn.

Input:
    - Raw text data từ GUI.
    - CSV numeric data từ data module.
    - Config train: degree, lr, epochs.
    - Feature names và target name.

Output:
    - x_data, y_data đã clean.
    - model đã fit.
    - metadata để GUI vẽ đúng feature.

TODO:
    - Viết hàm train_from_text.
    - Viết hàm train_from_csv_selection.
    - Validate degree >= 1, epochs > 0, lr > 0.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrainingResult:
    """
    Output chuẩn sau khi train.

    Attributes:
        model: Model đã train xong.
        x_data: Feature matrix hoặc vector.
        y_data: Target vector.
        feature_names: Tên các feature dùng để train.
        visual_feature_index: Index của feature dùng để vẽ trục x.
    """

    model: object
    x_data: np.ndarray
    y_data: np.ndarray
    feature_names: list[str]
    visual_feature_index: int


def train_from_text(raw_text: str, degree: int, lr: float, epochs: int) -> TrainingResult:
    """
    Train model từ text dạng `x,y`.

    Input:
        raw_text: Chuỗi nhiều dòng, mỗi dòng có dạng `x,y`.
        degree: Bậc đa thức.
        lr: Learning rate.
        epochs: Số epoch train.

    Output:
        TrainingResult gồm model, x_data, y_data và metadata.
    """
    raise NotImplementedError("Thành viên 4 tích hợp data.parse_xy_data và core.PolynomialRegressionGD ở đây.")


def train_from_csv_selection(
    numeric_data: dict[str, np.ndarray],
    feature_names: list[str],
    target_name: str,
    visual_name: str,
    degree: int,
    lr: float,
    epochs: int,
) -> TrainingResult:
    """
    Train model từ CSV đã đọc sẵn.

    Input:
        numeric_data: Dict tên cột -> numpy array numeric.
        feature_names: Danh sách cột dùng làm X train.
        target_name: Cột y.
        visual_name: Cột X dùng để vẽ trên plot.
        degree: Bậc đa thức.
        lr: Learning rate.
        epochs: Số epoch train.

    Output:
        TrainingResult gồm model, x_data, y_data và metadata.
    """
    raise NotImplementedError("Thành viên 4 tích hợp data.arrays_from_csv_selection và core.PolynomialRegressionGD ở đây.")

