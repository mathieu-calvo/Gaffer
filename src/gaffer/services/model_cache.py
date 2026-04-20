"""Joblib-backed train-or-load helper for the per-position ensemble.

Training the full ensemble on 8 seasons of data takes ~30s. The Streamlit app
calls this on every cold start, so caching to disk between runs (and using
`@st.cache_resource` on top) keeps interactive latency tolerable.
"""

from __future__ import annotations

import joblib

from gaffer.config import settings
from gaffer.domain.enums import Position
from gaffer.models.ensemble import PositionEnsemble
from gaffer.models.lightgbm_model import LgbmPredictor
from gaffer.models.quantile import LgbmQuantilePredictor
from gaffer.providers.base import FplDataProvider, HistoricalDataProvider
from gaffer.services.prediction_service import build_training_set

_MODEL_DIR = settings.cache_dir / "models"
_POINT_PATH = _MODEL_DIR / "ensemble_point.joblib"
_QUANTILE_PATH = _MODEL_DIR / "ensemble_quantile.joblib"


def _default_factory(_pos: Position) -> LgbmPredictor:
    return LgbmPredictor(n_estimators=300, learning_rate=0.05)


def _default_quantile_factory(_pos: Position) -> LgbmQuantilePredictor:
    return LgbmQuantilePredictor(n_estimators=300, learning_rate=0.05)


def train_or_load_ensembles(
    fpl: FplDataProvider,
    historical: HistoricalDataProvider,
    force_retrain: bool = False,
) -> tuple[PositionEnsemble, PositionEnsemble]:
    """Return (point_ensemble, quantile_ensemble), training and persisting if needed.

    Both ensembles share the same training matrix; they're trained in series
    rather than in parallel because LightGBM already saturates available cores.
    """
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not force_retrain and _POINT_PATH.exists() and _QUANTILE_PATH.exists():
        return joblib.load(_POINT_PATH), joblib.load(_QUANTILE_PATH)

    td = build_training_set(fpl, historical)
    X = td.X.dropna()
    y = td.y.loc[X.index]

    point = PositionEnsemble(factory=_default_factory).fit(X, y)
    quantile = PositionEnsemble(factory=_default_quantile_factory).fit(X, y)

    joblib.dump(point, _POINT_PATH)
    joblib.dump(quantile, _QUANTILE_PATH)
    return point, quantile
