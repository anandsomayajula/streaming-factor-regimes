import numpy as np

class OnlineFactorEstimator:
    """
    Tracks an estimate of the top principal component (factor)
    using Oja's rule.
    """
    def __init__(self, num_assets: int, step_size: float = 0.01, seed: int = 0):
        rng = np.random.default_rng(seed)
        w = rng.normal(size=num_assets)
        self.w = w / np.linalg.norm(w)
        self.step_size = step_size

    def update(self, x: np.ndarray):
        """
        One Oja update with a new return vector x.
        """
        y = x @ self.w          
        self.w = self.w + self.step_size * y * (x - y * self.w)
        self.w /= np.linalg.norm(self.w)

    def get_factor(self) -> np.ndarray:
        return self.w.copy()
