"""Player Projections — searchable table of every player with intervals + fixtures."""

from __future__ import annotations

import pandas as pd
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
    "(LightGBM quantile regression at 0.1 / 0.9)."
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
render_player_table(proj.players, proj.projections, gameweek=int(target_gw), horizon_total=True)

st.divider()
st.subheader("Upcoming fixtures (next 3)")
upcoming = fixtures[~fixtures["finished"]].sort_values("kickoff_time").head(20)[
    ["event", "kickoff_time", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
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
st.dataframe(upcoming, hide_index=True, use_container_width=True)
