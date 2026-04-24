"""Player Projections — searchable table of every player with intervals + fixtures."""

from __future__ import annotations

import streamlit as st

from gaffer.config import settings
from gaffer.providers.fpl_api import LiveFplApiProvider
from gaffer.providers.historical_csv import HistoricalCsvProvider
from gaffer.services.model_cache import train_or_load_ensembles
from gaffer.services.prediction_service import predict_projections
from gaffer.ui.components import render_player_table

st.set_page_config(
    page_title="Gaffer · Player Projections",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading data + ensemble…")
def load_projections(horizon: int):
    fpl = LiveFplApiProvider()
    historical = HistoricalCsvProvider()
    point, quantile = train_or_load_ensembles(fpl, historical)
    proj = predict_projections(
        fpl=fpl, historical=historical,
        point_model=point, quantile_model=quantile,
        horizon_gws=horizon,
    )
    return proj, fpl.get_current_gw(), fpl.get_fixtures()


st.title("📈 Player Projections")
st.markdown(
    "Every available player with point estimates and 80% prediction intervals "
    "(LightGBM quantile regression at 0.1 / 0.9). Click a row to see that "
    "player's upcoming fixtures."
)

with st.sidebar:
    horizon = st.slider("Horizon (gameweeks)", 1, 5, settings.horizon)

try:
    proj, current_gw, fixtures = load_projections(horizon)
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load projections: {exc}")
    st.stop()

target_gw = st.selectbox(
    "Gameweek for headline projection",
    options=sorted(proj.projections.index.get_level_values("gameweek").unique()),
    index=0,
)
selected_pid = render_player_table(
    proj.players,
    proj.projections,
    gameweek=int(target_gw),
    horizon_total=True,
    selectable=True,
    key="proj_table",
)

st.divider()

upcoming = fixtures[~fixtures["finished"]].sort_values("kickoff_time")

if selected_pid is not None:
    player_row = proj.players.loc[selected_pid]
    team = str(player_row["team"])
    player_fixtures = upcoming[
        (upcoming["team_h"] == team) | (upcoming["team_a"] == team)
    ].copy()
    player_fixtures["Opponent"] = player_fixtures.apply(
        lambda r: r["team_a"] if r["team_h"] == team else r["team_h"], axis=1
    )
    player_fixtures["Venue"] = player_fixtures.apply(
        lambda r: "H" if r["team_h"] == team else "A", axis=1
    )
    player_fixtures["FDR"] = player_fixtures.apply(
        lambda r: r["team_h_difficulty"] if r["team_h"] == team else r["team_a_difficulty"],
        axis=1,
    )
    player_fixtures = player_fixtures[
        ["event", "kickoff_time", "Opponent", "Venue", "FDR"]
    ].rename(columns={"event": "GW", "kickoff_time": "Kickoff"})

    st.subheader(
        f"Upcoming fixtures · {player_row['name']} ({team}) · "
        f"{len(player_fixtures)} match{'es' if len(player_fixtures) != 1 else ''}"
    )
    st.dataframe(player_fixtures, hide_index=True, use_container_width=True)
else:
    st.subheader("Upcoming fixtures (league)")
    league = upcoming.head(20)[
        ["event", "kickoff_time", "team_h", "team_a",
         "team_h_difficulty", "team_a_difficulty"]
    ].rename(
        columns={
            "event": "GW",
            "kickoff_time": "Kickoff",
            "team_h": "Home",
            "team_a": "Away",
            "team_h_difficulty": "FDR (H)",
            "team_a_difficulty": "FDR (A)",
        }
    )
    st.dataframe(league, hide_index=True, use_container_width=True)
    st.caption("Click a player row above to filter fixtures to that player only.")
