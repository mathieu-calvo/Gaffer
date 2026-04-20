"""Ridge-regression baseline.

Kept as a sanity anchor in the benchmark — if a boosted tree doesn't clearly beat
a linear model on this data, that's a signal the feature pipeline is off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class RidgePredictor:
    name = "ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        self._model = Ridge(alpha=alpha)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgePredictor:
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)
