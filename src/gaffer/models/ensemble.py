"""Position-routed ensemble.

Holds one trained predictor per position and dispatches `predict` based on the
`position` column of the incoming frame. Decouples the optimiser (which sees
predictions, one per player-GW) from the per-position training regime.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from gaffer.domain.enums import Position
from gaffer.models.base import PointsPredictor


class PositionEnsemble:
    """Collection of per-position predictors sharing the same input schema.

    `position_column` defaults to `"position"`. Rows whose position isn't in the
    trained set raise a KeyError — callers should pre-filter.
    """

    def __init__(
        self,
        factory: Callable[[Position], PointsPredictor],
        position_column: str = "position",
    ) -> None:
        self._factory = factory
        self._position_col = position_column
        self._models: dict[Position, PointsPredictor] = {}
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PositionEnsemble:
        if self._position_col not in X.columns:
            raise KeyError(
                f"Expected column {self._position_col!r} in X; got {list(X.columns)[:5]}..."
            )
        self._feature_names = self._numeric_features(X).columns.tolist()
        for pos_value in X[self._position_col].unique():
            position = Position(pos_value)
            mask = (X[self._position_col] == pos_value).values
            model = self._factory(position)
            X_pos = self._numeric_features(X.iloc[mask])
            model.fit(X_pos, y.iloc[mask])
            self._models[position] = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(X), dtype=float)
        for pos_value in X[self._position_col].unique():
            position = Position(pos_value)
            if position not in self._models:
                raise KeyError(f"No trained model for position {position}")
            mask = (X[self._position_col] == pos_value).values
            X_pos = self._numeric_features(X.iloc[mask])
            out[mask] = self._models[position].predict(X_pos)
        return out

    def predict_interval(
        self, X: pd.DataFrame, quantiles: tuple[float, float] = (0.1, 0.9)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Route quantile-interval predictions per position; leaf models must implement it."""
        lower = np.empty(len(X), dtype=float)
        upper = np.empty(len(X), dtype=float)
        for pos_value in X[self._position_col].unique():
            position = Position(pos_value)
            model = self._models[position]
            if not hasattr(model, "predict_interval"):
                raise AttributeError(
                    f"Leaf model for {position} has no predict_interval"
                )
            mask = (X[self._position_col] == pos_value).values
            X_pos = self._numeric_features(X.iloc[mask])
            lo, hi = model.predict_interval(X_pos, quantiles=quantiles)
            lower[mask] = lo
            upper[mask] = hi
        return lower, upper

    @property
    def feature_names_in_(self) -> list[str]:
        return list(self._feature_names)

    def _numeric_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        # Drop the routing column and any non-numeric columns that LightGBM/XGBoost
        # can't ingest directly (team/opponent_team strings, kickoff dates, etc.).
        numeric = frame.drop(columns=[self._position_col], errors="ignore").select_dtypes(
            include=["number", "bool"]
        )
        if self._feature_names:
            # Lock to the feature set seen at fit-time so train/inference stay aligned.
            return numeric.reindex(columns=self._feature_names)
        return numeric

    def models(self) -> dict[Position, PointsPredictor]:
        return dict(self._models)
