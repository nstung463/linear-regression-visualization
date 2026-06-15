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

from core.preprocessing import standardize
from core.training import TrainingFrame


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
        if degree < 1:
            raise ValueError("degree must be at least 1")

        self.degree = degree
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.save_every = max(1, int(save_every))

        self.weights: np.ndarray = np.array([])
        self.frames: list[TrainingFrame] = []
        self._x_mean: np.ndarray | float | None = None
        self._x_std: np.ndarray | float | None = None
        self._y_mean: float | None = None
        self._y_std: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PolynomialRegressionGD":
        """
        Train model.

        Input:
            x: Feature vector/matrix.
            y: Target vector.

        Output:
            self, sau khi đã có weights và frames.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D numpy array")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")

        x_std, self._x_mean, self._x_std = standardize(x)
        y_std, self._y_mean, self._y_std = standardize(y)

        X = self._make_design_matrix(x_std)
        self.weights = np.zeros(X.shape[1], dtype=float)
        self.frames = []

        n_samples = X.shape[0]
        for epoch in range(self.epochs + 1):
            predictions = X @ self.weights
            residuals = predictions - y_std
            loss = float(np.mean(residuals**2))

            if epoch % self.save_every == 0 or epoch == self.epochs:
                self.frames.append(TrainingFrame(epoch=epoch, weights=self.weights.copy(), loss=loss))

            if epoch == self.epochs:
                break

            gradient = (2.0 / n_samples) * (X.T @ residuals)
            self.weights = self.weights - self.lr * gradient

        return self

    def predict_with_weights(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Predict y bằng một bộ weights cụ thể.

        Input:
            x: Feature vector/matrix cần predict.
            weights: Weights tại một frame.

        Output:
            y_pred: numpy array prediction đã unscale về đơn vị gốc.
        """
        if self._x_mean is None or self._x_std is None or self._y_mean is None or self._y_std is None:
            raise ValueError("Model must be fit before calling predict_with_weights.")

        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D numpy array")

        if x.shape[1] != np.asarray(self._x_mean).shape[0]:
            raise ValueError("Input x must have the same number of features used during fit.")

        x_std = (x - self._x_mean) / self._x_std
        predictions_std = self._make_design_matrix(x_std) @ np.asarray(weights, dtype=float)
        return predictions_std * float(self._y_std) + float(self._y_mean)

    def equation(self, weights: np.ndarray, feature_names: list[str] | None = None) -> str:
        """
        Tạo text phương trình để hiển thị trên plot.

        Input:
            weights: Vector weights.
            feature_names: Tên feature nếu có.

        Output:
            Chuỗi equation.
        """
        weights = np.asarray(weights, dtype=float).ravel()
        num_features = (weights.shape[0] - 1) // self.degree
        if weights.shape[0] != 1 + num_features * self.degree:
            raise ValueError("Weights length does not match model degree and feature count.")

        if feature_names is None:
            feature_names = [f"x{i + 1}" for i in range(num_features)]

        terms: list[str] = []
        intercept = float(weights[0])
        terms.append(f"{intercept:.3f}")

        index = 1
        for power in range(1, self.degree + 1):
            for feature_index in range(num_features):
                coefficient = float(weights[index])
                index += 1
                if abs(coefficient) < 1e-12:
                    continue

                name = feature_names[feature_index]
                power_suffix = "" if power == 1 else f"^{power}"
                sign = " + " if coefficient >= 0 else " - "
                absolute_value = abs(coefficient)
                term = f"{absolute_value:.3f}*{name}{power_suffix}"
                terms.append(sign + term)

        equation = "y = " + "".join(terms)
        return equation

    def _make_design_matrix(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("Internal design matrix builder requires 2D input.")

        columns = [np.ones((x.shape[0], 1), dtype=float)]
        for power in range(1, self.degree + 1):
            columns.append(x**power)

        return np.hstack(columns)

