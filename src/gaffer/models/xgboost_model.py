"""XGBoost per-position regressor."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


class XgbPredictor:
    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self._model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
            tree_method="hist",
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XgbPredictor:
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_
