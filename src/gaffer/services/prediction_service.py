"""End-to-end prediction pipeline: raw data → trained ensemble → projections.

Orchestrates providers (live FPL API + historical CSV), feature engineering, and
model training. Two public entry points:

- :func:`build_training_set` — assemble the combined historical + current-season
  training matrix with EWMA rolling features and a T+1-shifted target. Ported
  from `FantasyApiModel.prepare_train_test_sets`.
- :func:`build_inference_set` — for each upcoming fixture, produce the feature
  vector by joining each player's latest rolling snapshot with the fixture's
  home/away + FDR information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gaffer.config import settings
from gaffer.features.constants import (
    FIXTURE_FEATURES,
    PLAYER_ID_COL,
    PLAYER_METADATA_FEATURES,
    TEAM_METADATA_FEATURES,
)
from gaffer.features.engineering import feature_engineer, reformat_dates, reformat_fdr
from gaffer.features.rolling import compute_rolling
from gaffer.models.base import PointsPredictor, QuantilePredictor
from gaffer.providers.base import FplDataProvider, HistoricalDataProvider


@dataclass(frozen=True)
class TrainingData:
    """Output of :func:`build_training_set`.

    `X` holds features known before each gameweek (rolled EWMA + fixture context);
    `y` is the realised FPL points; `seasons` is aligned to `X.index` and is what
    the blocked CV splitter uses to form folds.
    """

    X: pd.DataFrame
    y: pd.Series
    seasons: pd.Series


def build_training_set(
    fpl: FplDataProvider,
    historical: HistoricalDataProvider,
    alpha: float | None = None,
) -> TrainingData:
    """Combine historical + current-season data and produce the training matrix.

    The returned `X` has:
    - Rolling player/team/FPL feature EWMAs shifted by one gameweek (so GW N's
      row only sees stats from GW 1..N-1).
    - Current-GW fixture metadata (position, home/away, FDR, team, opponent).
    The target `y` is the GW's realised `total_points`.
    """
    alpha = alpha if alpha is not None else settings.ewma_alpha
    gwdf = pd.concat([historical.get_historical_gwdata(), fpl.get_player_histories()])
    engineered = feature_engineer(gwdf)

    rolling = compute_rolling(engineered, alpha=alpha).rolling

    t_features = (
        engineered.reset_index()
        .loc[
            :,
            FIXTURE_FEATURES
            + TEAM_METADATA_FEATURES
            + PLAYER_ID_COL
            + PLAYER_METADATA_FEATURES
            + ["total_points"],
        ]
        .rename(columns={"total_points": "fpl_points"})
        .drop_duplicates(subset=["name", "kickoff_date"], keep="last")
        .set_index(["name", "kickoff_date"])
        .drop(columns=["season"])
    )

    rolling_unique = rolling.dropna(how="all")
    rolling_unique = rolling_unique[~rolling_unique.index.duplicated(keep="last")]
    combined = rolling_unique.join(t_features, how="inner")

    season_lookup = (
        engineered.reset_index()
        .drop_duplicates(subset=["name", "kickoff_date"], keep="last")
        .set_index(["name", "kickoff_date"])["season"]
    )
    seasons = season_lookup.reindex(combined.index)

    y = combined["fpl_points"]
    X = combined.drop(columns=["fpl_points"])
    return TrainingData(X=X, y=y, seasons=seasons)


def build_inference_set(
    fpl: FplDataProvider,
    historical: HistoricalDataProvider,
    alpha: float | None = None,
    horizon_gws: int = 5,
) -> pd.DataFrame:
    """Feature matrix for upcoming fixtures over the next `horizon_gws` gameweeks.

    Each player's most recent rolling snapshot is joined against every upcoming
    fixture for their team. Home and away perspectives are materialised
    separately (FPL API exposes a single row per fixture).
    """
    alpha = alpha if alpha is not None else settings.ewma_alpha
    bootstrap = fpl.get_bootstrap()

    # Combined history for rolling
    gwdf = pd.concat([historical.get_historical_gwdata(), fpl.get_player_histories()])
    engineered = feature_engineer(gwdf)
    last_game = compute_rolling(engineered, alpha=alpha).last_game

    # Upcoming fixtures
    fixtures = fpl.get_fixtures()
    fixtures = fixtures[~fixtures["finished"]]
    fixtures = fixtures.loc[
        :,
        [
            "id", "event", "kickoff_time", "team_h", "team_a",
            "team_h_difficulty", "team_a_difficulty",
        ],
    ]
    fixtures = reformat_dates(fixtures)

    # Limit horizon
    min_gw = int(fixtures["event"].min())
    fixtures = fixtures[fixtures["event"] < min_gw + horizon_gws]

    # Materialise home + away perspectives.
    home = fixtures.assign(was_home=True)
    away = fixtures.assign(was_home=False)
    per_perspective = pd.concat([home, away], ignore_index=True)
    per_perspective["team"] = np.where(
        per_perspective["was_home"], per_perspective["team_h"], per_perspective["team_a"]
    )
    per_perspective["opponent_team"] = np.where(
        per_perspective["was_home"], per_perspective["team_a"], per_perspective["team_h"]
    )
    per_perspective = reformat_fdr(per_perspective)
    per_perspective = per_perspective.drop(columns=["id"])

    # Join latest player stats for every player on each team's matches.
    player_meta = pd.DataFrame(
        {
            "id": list(bootstrap.id_to_player_name.keys()),
            "name": list(bootstrap.id_to_player_name.values()),
        }
    )
    player_meta["team"] = player_meta["id"].map(bootstrap.player_id_to_team_name)
    player_meta["position"] = (
        player_meta["id"]
        .map(bootstrap.id_to_player_position)
        .map(bootstrap.id_to_position_short)
    )
    player_meta["position"] = player_meta["position"].replace("GK", "GKP")
    latest = player_meta.set_index("name").join(last_game, how="inner").reset_index()
    latest["price"] = (
        latest["id"].map(bootstrap.elements_df.set_index("id")["now_cost"]) / 10.0
    )

    merged = per_perspective.merge(latest, on="team", how="inner")
    merged = merged.sort_values(["kickoff_date", "name"]).set_index(
        ["name", "kickoff_date"]
    )
    return merged


@dataclass(frozen=True)
class Projections:
    """Fitted-model projections plus the player metadata the optimiser needs.

    `projections` is indexed by (player_id, gameweek) with columns
    `expected_points`, `lower_80`, `upper_80`. `players` is indexed by player_id
    with columns `name, team, position, price` — the exact shape consumed by
    :func:`gaffer.optimizer.milp.solve`.
    """

    projections: pd.DataFrame
    players: pd.DataFrame


def predict_projections(
    fpl: FplDataProvider,
    historical: HistoricalDataProvider,
    point_model: PointsPredictor,
    quantile_model: QuantilePredictor | None = None,
    horizon_gws: int = 5,
    alpha: float | None = None,
) -> Projections:
    """Build the inference set, predict, and reshape into optimiser inputs.

    A row in the inference set is one (player, fixture) — a player with two
    matches in a single GW (e.g., double gameweeks) appears twice and we sum
    expected points per player-GW.
    """
    inference = build_inference_set(fpl, historical, alpha=alpha, horizon_gws=horizon_gws)
    feature_cols = _ensure_feature_columns(inference, point_model)

    X = inference[feature_cols]
    inference = inference.assign(expected_points=point_model.predict(X))

    if quantile_model is not None:
        lower, upper = quantile_model.predict_interval(X, quantiles=(0.1, 0.9))
        inference = inference.assign(lower_80=lower, upper_80=upper)
    else:
        inference = inference.assign(
            lower_80=inference["expected_points"],
            upper_80=inference["expected_points"],
        )

    grouped = (
        inference.reset_index()
        .groupby(["id", "event"], as_index=True)[
            ["expected_points", "lower_80", "upper_80"]
        ]
        .sum()
    )
    grouped.index.set_names(["player_id", "gameweek"], inplace=True)

    players = (
        inference.reset_index()
        .drop_duplicates("id")
        .set_index("id")[["name", "team", "position", "price"]]
        .sort_index()
    )
    players.index.set_names("player_id", inplace=True)
    return Projections(projections=grouped, players=players)


def _ensure_feature_columns(
    inference: pd.DataFrame, model: PointsPredictor
) -> list[str]:
    """Pick feature columns to feed the model.

    Always includes `position` (the PositionEnsemble routes on it) and otherwise
    keeps numeric/bool columns minus obvious metadata.
    """
    drop = {"id", "event", "team_h", "team_a", "team", "opponent_team"}
    cols = [
        c for c in inference.columns
        if c not in drop and (c == "position" or inference[c].dtype.kind in "fib")
    ]
    return cols
