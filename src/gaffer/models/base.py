"""Points-predictor protocol and shared types.

Any object implementing `fit` + `predict` (and optionally `predict_interval`) can
plug into the training harness and ensemble router. Concrete implementations live
in `ridge.py`, `xgboost_model.py`, `lightgbm_model.py`, `quantile.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class PointsPredictor(Protocol):
    """Minimal interface for a per-position points regressor."""

    @property
    def name(self) -> str:
        """Human-readable model identifier used in benchmark tables."""
        ...

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PointsPredictor:
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


@runtime_checkable
class QuantilePredictor(Protocol):
    """Optional interface for predictors that emit quantile-based intervals."""

    def predict_interval(
        self, X: pd.DataFrame, quantiles: tuple[float, float] = (0.1, 0.9)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) arrays for the requested quantiles."""
        ...


@dataclass(frozen=True)
class CvFoldResult:
    """Performance of a single predictor on one CV fold."""

    model_name: str
    fold: int
    rmse: float
    mae: float
    n_test: int


@dataclass(frozen=True)
class CvSummary:
    """Aggregated CV result across folds for one predictor on one position."""

    model_name: str
    position: str
    mean_rmse: float
    std_rmse: float
    mean_mae: float
    folds: list[CvFoldResult]
