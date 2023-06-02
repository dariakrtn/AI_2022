from typing import Tuple
import numpy as np


class Line():
    def __init__(
        self,
        points: np.ndarray
    ) -> None:

        self.k = None
        self.b = None
        self.points = points

    def estimate_params(self) -> None:
        points_num = len(self.points)
        if points_num > 2:
            raise NotImplementedError
        elif points_num < 2:
            raise ValueError(f"Not enough points. Must be at least 2, but got {len(self.points)}")
        else:
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]

            self.k = (y1 - y2) / (x1 - x2 + 0.000001)
            self.b = y2 - self.k * x2


    def divide_points(
        self,
        points: np.ndarray,
        eps: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    
        distance = np.abs(self.k * points[:, 0] - points[:, 1] + self.b) / np.sqrt(self.k ** 2 + 1 + 0.00001)
        return points[distance <= eps], points[distance > eps]