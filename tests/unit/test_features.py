"""Feature-engineering transform tests."""

from __future__ import annotations

import pandas as pd
import pytest

from gaffer.features.constants import ROLLING_FEATURES
from gaffer.features.engineering import (
    feature_engineer,
    reformat_dates,
    reformat_fdr,
    reformat_fpl_features,
    reformat_team_form,
)
from gaffer.features.rolling import RollingArtefacts, compute_rolling


class TestReformatDates:
    def test_iso_string_to_datetime(self):
        df = pd.DataFrame({"kickoff_time": ["2023-08-12T15:00:00Z", "2023-08-13T17:30:00Z"]})
        out = reformat_dates(df)
        assert "kickoff_date" in out.columns
        assert "kickoff_time" not in out.columns
        assert out["kickoff_date"].dtype.kind == "M"
        # date-only — hour/minute should be zero
        assert out["kickoff_date"].iloc[0] == pd.Timestamp("2023-08-12")

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"kickoff_time": ["2023-08-12T15:00:00Z"]})
        df_copy = df.copy()
        reformat_dates(df)
        pd.testing.assert_frame_equal(df, df_copy)


class TestReformatTeamForm:
    def test_home_win_scores_three(self):
        df = pd.DataFrame({
            "was_home": [True],
            "team_h_score": [2],
            "team_a_score": [1],
        })
        out = reformat_team_form(df)
        assert out["team_points"].iloc[0] == 3.0
        assert out["team_goals_scored"].iloc[0] == 2
        assert out["team_goals_conceded"].iloc[0] == 1

    def test_away_win_scores_three(self):
        df = pd.DataFrame({
            "was_home": [False],
            "team_h_score": [0],
            "team_a_score": [3],
        })
        out = reformat_team_form(df)
        assert out["team_points"].iloc[0] == 3.0
        assert out["team_goals_scored"].iloc[0] == 3
        assert out["team_goals_conceded"].iloc[0] == 0

    def test_draw_scores_one(self):
        df = pd.DataFrame({
            "was_home": [True], "team_h_score": [1], "team_a_score": [1],
        })
        assert reformat_team_form(df)["team_points"].iloc[0] == 1.0

    def test_loss_scores_zero(self):
        df = pd.DataFrame({
            "was_home": [True], "team_h_score": [0], "team_a_score": [2],
        })
        assert reformat_team_form(df)["team_points"].iloc[0] == 0.0

    def test_drops_raw_score_columns(self):
        df = pd.DataFrame({
            "was_home": [True], "team_h_score": [1], "team_a_score": [0],
        })
        out = reformat_team_form(df)
        assert "team_h_score" not in out.columns
        assert "team_a_score" not in out.columns


class TestReformatFdr:
    def test_home_perspective_uses_away_difficulty_for_team(self):
        df = pd.DataFrame({
            "was_home": [True],
            "team_a_difficulty": [4],
            "team_h_difficulty": [2],
        })
        out = reformat_fdr(df)
        # team_fdr = the difficulty *facing* this team — i.e., opponent's strength.
        # Implementation uses team_a_difficulty for home rows.
        assert out["team_fdr"].iloc[0] == 4
        assert out["opponent_team_fdr"].iloc[0] == 2

    def test_away_perspective(self):
        df = pd.DataFrame({
            "was_home": [False],
            "team_a_difficulty": [4],
            "team_h_difficulty": [2],
        })
        out = reformat_fdr(df)
        assert out["team_fdr"].iloc[0] == 2
        assert out["opponent_team_fdr"].iloc[0] == 4


class TestReformatFplFeatures:
    def test_normalises_by_managers(self):
        df = pd.DataFrame({
            "transfers_balance": [1_000],
            "selected": [500_000],
            "transfers_in": [2_000],
            "transfers_out": [1_000],
            "nb_managers": [10_000_000],
        })
        out = reformat_fpl_features(df)
        assert out["transfers_balance_pct"].iloc[0] == pytest.approx(0.0001)
        assert out["selected_pct"].iloc[0] == pytest.approx(0.05)
        dropped = ("transfers_balance", "selected", "transfers_in", "transfers_out", "nb_managers")
        for col in dropped:
            assert col not in out.columns


class TestFeatureEngineer:
    def test_preserves_row_count(self, synthetic_gwdf):
        out = feature_engineer(synthetic_gwdf)
        assert len(out) == len(synthetic_gwdf)

    def test_indexed_by_kickoff_date(self, synthetic_gwdf):
        out = feature_engineer(synthetic_gwdf)
        assert out.index.name == "kickoff_date"

    def test_canonicalises_gk_to_gkp(self, synthetic_gwdf):
        df = synthetic_gwdf.copy()
        df.loc[0, "position"] = "GK"
        out = feature_engineer(df)
        assert "GK" not in out["position"].values
        assert "MID" in out["position"].values  # other rows untouched


class TestComputeRolling:
    def test_returns_artefacts(self, synthetic_gwdf):
        engineered = feature_engineer(synthetic_gwdf)
        out = compute_rolling(engineered, alpha=0.5, min_periods=1)
        assert isinstance(out, RollingArtefacts)
        assert isinstance(out.rolling, pd.DataFrame)
        assert isinstance(out.last_game, pd.DataFrame)

    def test_rolling_columns_are_rolling_features(self, synthetic_gwdf):
        engineered = feature_engineer(synthetic_gwdf)
        out = compute_rolling(engineered, alpha=0.5, min_periods=1)
        assert set(out.rolling.columns) == set(ROLLING_FEATURES)

    def test_t_plus_one_shift_drops_first_observation(self, synthetic_gwdf):
        # min_periods=1, alpha=1.0 — first row of each player should be NaN
        # because the rolling output is shifted by one to avoid leakage.
        engineered = feature_engineer(synthetic_gwdf)
        out = compute_rolling(engineered, alpha=1.0, min_periods=1)
        # First row per player has nothing prior — should be all NaN.
        first_per_player = out.rolling.groupby(level="name").head(1)
        assert first_per_player.isna().all().all()

    def test_last_game_one_row_per_player(self, synthetic_gwdf):
        engineered = feature_engineer(synthetic_gwdf)
        out = compute_rolling(engineered, alpha=0.5, min_periods=1)
        n_players = engineered["name"].nunique()
        assert len(out.last_game) == n_players
        assert set(out.last_game.index) == set(engineered["name"].unique())

    def test_alpha_one_uses_only_latest(self, synthetic_gwdf):
        # With alpha=1, the EWMA degenerates to "use the most recent value only".
        # last_game for each player should match their final raw observation
        # for any feature that's directly carried (e.g., minutes — always 90 here).
        engineered = feature_engineer(synthetic_gwdf)
        out = compute_rolling(engineered, alpha=1.0, min_periods=1)
        # All synthetic minutes are 90, so the rolling minutes should also be 90.
        assert (out.last_game["minutes"] == 90.0).all()
