"""LightGBM quantile-regression predictor.

Trains three LightGBM models (at α=0.1, 0.5, 0.9 by default) so the app can
surface 80%-coverage prediction intervals alongside the point forecast.

Quantile regression minimises the pinball loss, which is asymmetric — it
explicitly teaches the model how badly errors on the high side vs low side
should be penalised. Coupled with boosted trees this gives conditional
intervals that tighten for well-observed players and widen for noisy ones
(rotations, injuries, rookies) — exactly the behaviour we want to surface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


class LgbmQuantilePredictor:
    name = "lightgbm_quantile"

    def __init__(
        self,
        quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 30,
        random_state: int = 42,
    ) -> None:
        self._quantiles = quantiles
        self._models: dict[float, LGBMRegressor] = {
            q: LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
            for q in quantiles
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LgbmQuantilePredictor:
        for model in self._models.values():
            model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point prediction uses the median quantile (typically 0.5)."""
        median_q = sorted(self._quantiles)[len(self._quantiles) // 2]
        return self._models[median_q].predict(X)

    def predict_interval(
        self, X: pd.DataFrame, quantiles: tuple[float, float] = (0.1, 0.9)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) predictions for the requested quantile pair."""
        lower_q, upper_q = quantiles
        if lower_q not in self._models or upper_q not in self._models:
            raise ValueError(
                f"Quantiles {quantiles} not among trained quantiles {self._quantiles}"
            )
        return self._models[lower_q].predict(X), self._models[upper_q].predict(X)
