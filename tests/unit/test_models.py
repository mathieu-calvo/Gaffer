"""Predictor + ensemble + CV tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gaffer.domain.enums import Position
from gaffer.models.base import (
    CvSummary,
    PointsPredictor,
    QuantilePredictor,
)
from gaffer.models.ensemble import PositionEnsemble
from gaffer.models.lightgbm_model import LgbmPredictor
from gaffer.models.quantile import LgbmQuantilePredictor
from gaffer.models.ridge import RidgePredictor
from gaffer.models.training import (
    benchmark_predictors,
    evaluate_predictor,
    season_block_splits,
)
from gaffer.models.xgboost_model import XgbPredictor


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "f3": rng.normal(size=n),
    })
    y = pd.Series(0.5 * X["f1"] - 0.3 * X["f2"] + rng.normal(scale=0.5, size=n))
    return X, y


@pytest.fixture
def position_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    rows = []
    targets = []
    for pos in (Position.GKP, Position.DEF, Position.MID, Position.FWD):
        for _ in range(50):
            f1, f2 = rng.normal(), rng.normal()
            rows.append({"position": pos.value, "f1": f1, "f2": f2})
            targets.append(f1 + f2)
    X = pd.DataFrame(rows)
    y = pd.Series(targets)
    return X, y


@pytest.fixture
def season_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(2)
    seasons = pd.Series(np.repeat(["20-21", "21-22", "22-23", "23-24"], 50))
    X = pd.DataFrame({
        "f1": rng.normal(size=200),
        "f2": rng.normal(size=200),
    })
    y = pd.Series(X["f1"] + 0.2 * X["f2"] + rng.normal(scale=0.5, size=200))
    return X, y, seasons


class TestPredictorProtocols:
    @pytest.mark.parametrize("factory", [
        lambda: RidgePredictor(),
        lambda: XgbPredictor(n_estimators=10),
        lambda: LgbmPredictor(n_estimators=10),
        lambda: LgbmQuantilePredictor(n_estimators=10),
    ])
    def test_implements_points_predictor(self, factory):
        assert isinstance(factory(), PointsPredictor)

    def test_quantile_implements_quantile_protocol(self):
        assert isinstance(LgbmQuantilePredictor(n_estimators=10), QuantilePredictor)


class TestPredictors:
    def test_ridge_fits_and_predicts(self, regression_data):
        X, y = regression_data
        model = RidgePredictor().fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_xgb_fits_and_predicts(self, regression_data):
        X, y = regression_data
        model = XgbPredictor(n_estimators=20).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_lgbm_fits_and_predicts(self, regression_data):
        X, y = regression_data
        model = LgbmPredictor(n_estimators=20).fit(X, y)
        assert model.predict(X).shape == (len(X),)


class TestQuantilePredictor:
    def test_intervals_bracket_point_prediction(self, regression_data):
        X, y = regression_data
        model = LgbmQuantilePredictor(n_estimators=30).fit(X, y)
        median = model.predict(X)
        lo, hi = model.predict_interval(X, quantiles=(0.1, 0.9))
        # Quantile regression with finite trees doesn't strictly guarantee
        # lo<=median<=hi pointwise, but on average it must hold.
        assert lo.mean() < median.mean() < hi.mean()
        assert lo.shape == hi.shape == (len(X),)

    def test_unknown_quantile_raises(self, regression_data):
        X, y = regression_data
        model = LgbmQuantilePredictor(n_estimators=10).fit(X, y)
        with pytest.raises(ValueError, match="Quantiles"):
            model.predict_interval(X, quantiles=(0.05, 0.95))


class TestPositionEnsemble:
    def test_routes_to_per_position_models(self, position_data):
        X, y = position_data
        ensemble = PositionEnsemble(factory=lambda pos: RidgePredictor())
        ensemble.fit(X, y)

        models = ensemble.models()
        assert set(models.keys()) == set(Position)

    def test_predict_shape_matches_input(self, position_data):
        X, y = position_data
        ensemble = PositionEnsemble(factory=lambda pos: RidgePredictor()).fit(X, y)
        preds = ensemble.predict(X)
        assert preds.shape == (len(X),)

    def test_missing_position_column_raises(self, regression_data):
        X, y = regression_data  # no `position` column
        with pytest.raises(KeyError, match="position"):
            PositionEnsemble(factory=lambda pos: RidgePredictor()).fit(X, y)

    def test_unseen_position_at_predict_raises(self, position_data):
        X, y = position_data
        ensemble = PositionEnsemble(factory=lambda pos: RidgePredictor())
        # Train on a single position.
        mask = X["position"] == Position.MID.value
        ensemble.fit(X.loc[mask], y.loc[mask])
        with pytest.raises(KeyError, match="position"):
            ensemble.predict(X)  # contains GKP/DEF/FWD that weren't trained


class TestSeasonBlockSplits:
    def test_walk_forward_split_count(self):
        seasons = pd.Series(["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10)
        splits = season_block_splits(seasons, min_train_seasons=2)
        # 4 seasons, 2 warm-up → 2 folds (one per season after warm-up).
        assert len(splits) == 2

    def test_test_set_is_one_season(self):
        seasons = pd.Series(["A"] * 5 + ["B"] * 5 + ["C"] * 5)
        splits = season_block_splits(seasons, min_train_seasons=2)
        train_idx, test_idx = splits[0]
        assert set(seasons.iloc[test_idx]) == {"C"}
        assert set(seasons.iloc[train_idx]) == {"A", "B"}

    def test_too_few_seasons_raises(self):
        with pytest.raises(ValueError, match="seasons"):
            season_block_splits(pd.Series(["A", "A", "B"]), min_train_seasons=2)


class TestEvaluatePredictor:
    def test_returns_summary(self, season_data):
        X, y, seasons = season_data
        summary = evaluate_predictor(
            factory=RidgePredictor, X=X, y=y, seasons=seasons,
            position_label="ALL", min_train_seasons=2,
        )
        assert isinstance(summary, CvSummary)
        assert summary.position == "ALL"
        assert len(summary.folds) == 2  # 4 seasons, 2 warm-up

    def test_benchmark_table_sorted_by_rmse(self, season_data):
        X, y, seasons = season_data
        df = benchmark_predictors(
            factories={
                "ridge": RidgePredictor,
                "lgbm": lambda: LgbmPredictor(n_estimators=20),
            },
            X=X, y=y, seasons=seasons,
        )
        assert list(df.columns) == [
            "model", "position", "mean_rmse", "std_rmse", "mean_mae", "n_folds",
        ]
        assert df["mean_rmse"].is_monotonic_increasing
