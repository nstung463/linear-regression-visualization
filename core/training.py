"""
Owner: Thành viên 1 - Core AI / Model.

File này chứa các object liên quan đến quá trình training.

Input:
    - epoch: số thứ tự vòng lặp training.
    - weights: vector trọng số tại epoch đó.
    - loss: MSE tại epoch đó.

Output:
    - TrainingFrame để visualization vẽ animation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrainingFrame:
    """
    Một frame trong quá trình AI học.

    Attributes:
        epoch: Epoch hiện tại.
        weights: Copy của weights tại epoch này.
        loss: Mean Squared Error tại epoch này.
    """

    epoch: int
    weights: np.ndarray
    loss: float

