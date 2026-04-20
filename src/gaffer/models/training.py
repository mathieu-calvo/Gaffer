"""Training harness — blocked time-series cross-validation by season.

Blocks by season rather than random split because the target is temporal: FPL
meta and player careers drift year-over-year, so a random split leaks future
information into training. One season held out at a time simulates the
production setting where we train on all past data and predict the current year.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from gaffer.models.base import CvFoldResult, CvSummary, PointsPredictor


def season_block_splits(
    seasons: pd.Series, min_train_seasons: int = 2
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) arrays: test = each season after the warm-up.

    The first `min_train_seasons` seasons are used to seed the first training
    fold; each subsequent season becomes a test fold while everything before it
    becomes train. Produces a walk-forward split.
    """
    unique = sorted(seasons.unique())
    if len(unique) <= min_train_seasons:
        raise ValueError(
            f"Need > {min_train_seasons} seasons to form at least one fold; "
            f"got {len(unique)}"
        )
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(min_train_seasons, len(unique)):
        train_seasons = unique[:i]
        test_season = unique[i]
        train_mask = seasons.isin(train_seasons).to_numpy()
        test_mask = (seasons == test_season).to_numpy()
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return splits


def evaluate_predictor(
    factory: Callable[[], PointsPredictor],
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    position_label: str = "all",
    min_train_seasons: int = 2,
) -> CvSummary:
    """Run blocked time-series CV for one predictor and return a summary.

    `factory` is called fresh per fold so no leakage between fits. `position_label`
    is metadata only — used when this function is called per-position.
    """
    folds: list[CvFoldResult] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        season_block_splits(seasons, min_train_seasons=min_train_seasons)
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        folds.append(
            CvFoldResult(
                model_name=model.name,
                fold=fold_idx,
                rmse=rmse,
                mae=mae,
                n_test=len(test_idx),
            )
        )

    rmses = np.array([f.rmse for f in folds])
    maes = np.array([f.mae for f in folds])
    return CvSummary(
        model_name=folds[0].model_name,
        position=position_label,
        mean_rmse=float(rmses.mean()),
        std_rmse=float(rmses.std()),
        mean_mae=float(maes.mean()),
        folds=folds,
    )


def benchmark_predictors(
    factories: dict[str, Callable[[], PointsPredictor]],
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    position_label: str = "all",
    min_train_seasons: int = 2,
) -> pd.DataFrame:
    """Benchmark multiple predictor factories; return a ranked summary DataFrame."""
    rows: list[dict[str, object]] = []
    for key, factory in factories.items():
        summary = evaluate_predictor(
            factory, X, y, seasons, position_label=position_label,
            min_train_seasons=min_train_seasons,
        )
        rows.append(
            {
                "model": key,
                "position": position_label,
                "mean_rmse": summary.mean_rmse,
                "std_rmse": summary.std_rmse,
                "mean_mae": summary.mean_mae,
                "n_folds": len(summary.folds),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_rmse").reset_index(drop=True)
