import numpy as np
import matplotlib.pyplot as plt

from line import Line
from typing import Dict

class RANSAC():
    def __init__(self) -> None:
        
        self.iter_num: int = 100
        self.inlin_thrsh: float = 0.8
        self.epsilon: float = 0.1
        self.best_params: dict = {}
        self.inliers:  list = []
        self.outliers: list = []
        self.score: int = 0
        self.points: np.ndarray = None

    def set_case(
        self,
        points: np.ndarray,
        iter_num: int = 100,
        inline_threshold: float = 0.8,
        epsilon: float = 0.1
    ) -> None:

        self.points = points
        self.iter_num = iter_num
        self.inlin_thrsh = inline_threshold
        self.epsilon = epsilon


    def clear_case(self) -> None:
        self.best_params = {}
        self.inliers = []
        self.outliers = []
        self.points = []
        self.score = 0

    def fit(self) -> Dict[str, float]:

        for i in range(self.iter_num):
            rnd_points_idx = np.random.randint(0, len(self.points), 2)
            point1 = self.points[rnd_points_idx[0]]
            point2 = self.points[rnd_points_idx[1]]
            line = Line(np.array([point1, point2]))
            line.estimate_params()
            inliers, outliers = line.divide_points(self.points, self.epsilon)
            curr_outliers_ratio = len(outliers) / len(self.points)

            if not (len(self.inliers) == 0 and len(self.outliers) == 0):
                best_outliers_ratio = len(self.outliers) / len(self.points)

                if best_outliers_ratio < curr_outliers_ratio:
                    continue

            self.best_params = {
                "k": line.k,
                "b": line.b
            }

            self.inliers = inliers
            self.outliers = outliers

            if (1 - curr_outliers_ratio) > self.inlin_thrsh:
                break

        return self.best_params


    def draw(self, save_path: str) -> None:
        plt.figure(figsize=(15, 10))
        plt.scatter(self.inliers[:, 0], self.inliers[:, 1], c="blue", label="Inliers")
        plt.scatter(self.outliers[:, 0], self.outliers[:, 1], c="red", label="Outliers")
        xmin = min(self.inliers[:, 0].min(), self.outliers[:, 0].min())
        xmax = max(self.inliers[:, 0].max(), self.outliers[:, 0].max())
        line_x = np.linspace(xmin, xmax, 2)
        line_y = self.best_params['k'] * line_x + self.best_params['b']
        plt.plot(line_x, line_y, c='green', label='Estimated line')
        plt.grid()
        plt.legend()
        plt.title("RANSAC estimation example")
        plt.xlabel("X-Axis")
        plt.ylabel("Y-Axis")
        plt.savefig(save_path, dpi=300)