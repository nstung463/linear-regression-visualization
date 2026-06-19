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

        self.weights: np.ndarray = np.array([]) # khởi tạo weights ban đầu là empty array
        self.frames: list[TrainingFrame] = [] # khởi tạo list frames ban đầu là empty list
        self._x_mean: np.ndarray | float | None = None
        self._x_std: np.ndarray | float | None = None # khởi tạo x_std ban đầu là None
        self._y_mean: float | None = None # khởi tạo y_mean ban đầu là None
        self._y_std: float | None = None # khởi tạo y_std ban đầu là None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PolynomialRegressionGD": # fit là hàm để train model
        """
        Train model.

        Input:
            x: Feature vector/matrix.
            y: Target vector.

        Output:
            self, sau khi đã có weights và frames.
        """
        # chuyển đổi x và y thành numpy array và flatten cho dễ xử lý
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        # kiểm tra xem x có phải là 1D hoặc 2D array không
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D numpy array")

        # kiểm tra xem x và y có cùng số lượng samples không
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")

        # chuẩn hóa x và y
        x_std, self._x_mean, self._x_std = standardize(x)
        y_std, self._y_mean, self._y_std = standardize(y)

        # tạo design matrix để tính toán weights
        X = self._make_design_matrix(x_std)
        self.weights = np.zeros(X.shape[1], dtype=float) # khởi tạo weights ban đầu là 0
        self.frames = [] # khởi tạo list frames ban đầu là empty

        # lặp qua từng epoch để training
        n_samples = X.shape[0]
        for epoch in range(self.epochs + 1):
            # tính toán predictions (predictions là giá trị dự đoán của model)
            predictions = X @ self.weights

            # tính toán residuals (residuals là sự khác biệt giữa predictions và y_std)
            residuals = predictions - y_std # residuals = predictions - y_std
            
            # tính toán loss (loss là giá trị sai số trung bình của model)
            loss = float(np.mean(residuals**2))

            # lưu frame nếu đạt điều kiện save_every hoặc là epoch cuối cùng
            if epoch % self.save_every == 0 or epoch == self.epochs:
                self.frames.append(TrainingFrame(epoch=epoch, weights=self.weights.copy(), loss=loss))

            # nếu đạt đến epoch cuối cùng thì break
            if epoch == self.epochs:
                break

            # tính toán gradient (gradient là đạo hàm của loss theo weights)
            gradient = (2.0 / n_samples) * (X.T @ residuals) # X.T là transpose của X (X.T là ma trận chuyển vị của X)
            # @ là phép nhân ma trận
            # T là phép chuyển vị ma trận
            # residuals là vector sai số giữa predictions và y_std
            # n_samples là số lượng samples
            # 2.0 / n_samples là hệ số điều chỉnh
            # gradient là đạo hàm của loss theo weights
            # self.weights là vector weights hiện tại
            self.weights = self.weights - self.lr * gradient # cập nhật weights bằng cách trừ đi gradient * learning rate

        return self # trả về self sau khi training hoàn tất

    def predict_with_weights(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Predict y bằng một bộ weights cụ thể.

        Input:
            x: Feature vector/matrix cần predict.
            weights: Weights tại một frame.

        Output:
            y_pred: numpy array prediction đã unscale về đơn vị gốc.
        """
        # kiểm tra xem model có được train không
        if self._x_mean is None or self._x_std is None or self._y_mean is None or self._y_std is None:
            raise ValueError("Model must be fit before calling predict_with_weights.")

        # chuyển đổi x thành numpy array và flatten cho dễ xử lý
        x = np.asarray(x, dtype=float)
        # kiểm tra xem x có phải là 1D hoặc 2D array không
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        # kiểm tra xem x có phải là 2D array không
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D numpy array")

        # kiểm tra xem x có cùng số lượng features với khi train không
        if x.shape[1] != np.asarray(self._x_mean).shape[0]: # np.asarray(self._x_mean).shape[0] là số lượng features của x khi train
            raise ValueError("Input x must have the same number of features used during fit.")

        # chuẩn hóa x
        x_std = (x - self._x_mean) / self._x_std # chuẩn hóa x bằng cách trừ đi mean và chia cho x_std (x_std là độ lệch chuẩn của x khi train)
        # tính toán predictions
        predictions_std = self._make_design_matrix(x_std) @ np.asarray(weights, dtype=float) # tính toán predictions bằng cách nhân design matrix với weights (design matrix là ma trận được tạo bởi các cột là x^i với i từ 0 đến degree)
        # unscale predictions về đơn vị gốc
        return predictions_std * float(self._y_std) + float(self._y_mean) # unscale predictions về đơn vị gốc bằng cách nhân với y_std và cộng với mean (y_std là độ lệch chuẩn của y khi train)

    def equation(self, weights: np.ndarray, feature_names: list[str] | None = None) -> str:
        """
        Tạo text phương trình để hiển thị trên plot.    

        Input:
            weights: Vector weights.
            feature_names: Tên feature nếu có.

        Output:
            Chuỗi equation.
        """
        # chuyển đổi weights thành numpy array và flatten cho dễ xử lý
        weights = np.asarray(weights, dtype=float).ravel()
        # kiểm tra xem weights có phải là 1D array không
        num_features = (weights.shape[0] - 1) // self.degree
        # kiểm tra xem weights có phải là 1D array không
        if weights.shape[0] != 1 + num_features * self.degree:
            raise ValueError("Weights length does not match model degree and feature count.") # nếu không khớp thì raise error

        # kiểm tra xem feature_names có phải là list không
        if feature_names is None:
            feature_names = [f"x{i + 1}" for i in range(num_features)] # nếu không có thì tạo list feature_names với tên là x1, x2, ..., xn

        terms: list[str] = [] # khởi tạo list terms ban đầu là empty
        intercept = float(weights[0]) # intercept là hệ số tự do
        terms.append(f"{intercept:.3f}") # thêm intercept vào list terms    

        index = 1 # index là index của weights
        # lặp qua từng power để tính toán terms (terms là list các term của phương trình)
        for power in range(1, self.degree + 1):
            # lặp qua từng feature_index để tính toán terms (feature_index là index của feature)
            for feature_index in range(num_features):
                # kiểm tra xem coefficient có phải là 0 không   (coefficient là hệ số của feature_index)
                coefficient = float(weights[index])
                index += 1 # tăng index lên 1
                if abs(coefficient) < 1e-12: # nếu coefficient nhỏ hơn 1e-12 thì continue
                    continue # nếu coefficient nhỏ hơn 1e-12 thì continue
                name = feature_names[feature_index] # name là tên của feature_index
                power_suffix = "" if power == 1 else f"^{power}" # power_suffix là suffix của power
                sign = " + " if coefficient >= 0 else " - " # sign là dấu của coefficient
                absolute_value = abs(coefficient) # absolute_value là giá trị tuyệt đối của coefficient
                term = f"{absolute_value:.3f}*{name}{power_suffix}" # term là term của feature_index
                terms.append(sign + term) # thêm term vào list terms    (sign + term là term của feature_index)

        equation = "y = " + "".join(terms) # equation là phương trình của model (y = intercept + term1 + term2 + ...)
        return equation

    def _make_design_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        Tạo design matrix để tính toán weights.

        Input:
            x: Feature vector/matrix.

        Output:
            Design matrix.
        """
        
        # chuyển đổi x thành numpy array và flatten cho dễ xử lý
        x = np.asarray(x, dtype=float)
        # kiểm tra xem x có phải là 2D array không
        if x.ndim != 2:
            raise ValueError("Internal design matrix builder requires 2D input.") # nếu không phải thì raise error

        columns = [np.ones((x.shape[0], 1), dtype=float)] # khởi tạo list columns ban đầu là [1, 1, ..., 1]
        for power in range(1, self.degree + 1):
            columns.append(x**power) # thêm x^power vào list columns

        return np.hstack(columns) # hstack là hàm để nối các array theo chiều ngang (np.hstack là hàm để nối các array theo chiều ngang)
