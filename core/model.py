"""
Owner: Thành viên 1 - Core AI / Model.

File này chứa thuật toán Polynomial Regression bằng Gradient Descent.

Không import Tkinter.
Không import Matplotlib.

Input chính:
    - x: numpy array 1D hoặc 2D.
    - y: numpy array 1D.
    - degree: bậc đa thức.
    - lr: learning rate.
    - epochs: số epoch.

Output chính:
    - self.weights: weights sau training.
    - self.frames: list TrainingFrame để animation.
    - prediction từ `predict_with_weights`.

TODO:
    - Copy class `PolynomialRegressionGD` từ prototype vào đây.
    - Import `TrainingFrame` từ `core.training`.
    - Import `standardize` từ `core.preprocessing`.
    - Đảm bảo model không phụ thuộc GUI.
"""

from __future__ import annotations

import numpy as np


class PolynomialRegressionGD:
    """
    Polynomial Regression trained by Gradient Descent.

    Public methods cần giữ ổn định để các thành viên khác gọi:
        - fit(x, y)
        - predict_with_weights(x, weights)
        - equation(weights, feature_names=None)
    """

    def __init__(self, degree: int, lr: float = 0.03, epochs: int = 500, save_every: int = 1) -> None:
        """
        Input:
            degree: Bậc đa thức.
            lr: Learning rate.
            epochs: Số vòng lặp train.
            save_every: Mỗi bao nhiêu epoch thì lưu một TrainingFrame.

        Output:
            Object model mới, chưa train.
        """
        raise NotImplementedError("Thành viên 1 implement constructor từ prototype.")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PolynomialRegressionGD":
        """
        Train model.

        Input:
            x: Feature vector/matrix.
            y: Target vector.

        Output:
            self, sau khi đã có weights và frames.
        """
        raise NotImplementedError("Thành viên 1 implement fit từ prototype.")

    def predict_with_weights(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Predict y bằng một bộ weights cụ thể.

        Input:
            x: Feature vector/matrix cần predict.
            weights: Weights tại một frame.

        Output:
            y_pred: numpy array prediction đã unscale về đơn vị gốc.
        """
        raise NotImplementedError("Thành viên 1 implement predict từ prototype.")

    def equation(self, weights: np.ndarray, feature_names: list[str] | None = None) -> str:
        """
        Tạo text phương trình để hiển thị trên plot.

        Input:
            weights: Vector weights.
            feature_names: Tên feature nếu có.

        Output:
            Chuỗi equation.
        """
        raise NotImplementedError("Thành viên 1 implement equation từ prototype.")

