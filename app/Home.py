"""Optimal Squad — the recommended 15-man squad and starting XI for the upcoming gameweek(s)."""

from __future__ import annotations

import streamlit as st

from gaffer.config import settings
from gaffer.providers.fpl_api import LiveFplApiProvider
from gaffer.providers.historical_csv import HistoricalCsvProvider
from gaffer.services.model_cache import train_or_load_ensembles
from gaffer.services.optimization_service import optimize_squad
from gaffer.services.prediction_service import predict_projections
from gaffer.ui.components import render_pitch, render_player_table

st.set_page_config(
    page_title="Gaffer · Optimal Squad",
    page_icon="⚽",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading FPL data + training ensemble (one-off)…")
def load_models_and_projections(horizon: int):
    fpl = LiveFplApiProvider()
    historical = HistoricalCsvProvider()
    point, quantile = train_or_load_ensembles(fpl, historical)
    proj = predict_projections(
        fpl=fpl, historical=historical,
        point_model=point, quantile_model=quantile,
        horizon_gws=horizon,
    )
    current_gw = fpl.get_current_gw()
    return proj, current_gw


@st.cache_data(show_spinner="Solving MILP…")
def solve_optimal_squad(horizon: int, bench_weight: float, current_gw: int, _proj):
    result = optimize_squad(
        projections=_proj.projections,
        players=_proj.players,
        start_gw=current_gw,
        horizon=horizon,
        bench_weight=bench_weight,
    )
    return result


st.title("⚽ Gaffer — Optimal Squad")
st.markdown(
    "Per-position ensemble × multi-gameweek MILP. Adjust the horizon to balance "
    "fixture lookahead against solver time."
)

with st.sidebar:
    st.header("Solver options")
    horizon = st.slider("Horizon (gameweeks)", 1, 5, settings.horizon)
    bench_weight = st.slider(
        "Bench weight in objective", 0.0, 1.0, 0.10, 0.05,
        help="Higher = more bench depth at the cost of starter quality.",
    )
    if st.button("Force retrain models", help="Useful after pushing model code changes"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

try:
    proj, current_gw = load_models_and_projections(horizon)
    result = solve_optimal_squad(horizon, bench_weight, current_gw, proj)
except FileNotFoundError as exc:
    st.error(
        f"Historical CSV not found: {exc}. Place "
        "`clean_merged_gwdf_2016_to_2024.csv` under `data/historical/`."
    )
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load data or train models: {exc}")
    st.stop()

st.success(
    f"Solved · status: {result.solver_status} · total expected points "
    f"(net of hits): {result.total_expected_points:.1f} over {len(result.plans)} GW"
)

tabs = st.tabs([f"GW {plan.gameweek}" for plan in result.plans])
for tab, plan in zip(tabs, result.plans, strict=True):
    with tab:
        col1, col2 = st.columns([2, 1])
        with col1:
            render_pitch(plan, proj.players, proj.projections)
        with col2:
            cap_row = proj.players.loc[plan.captain_id]
            vc_row = proj.players.loc[plan.vice_captain_id]
            st.metric("Expected points (net)", f"{plan.expected_points:.1f}")
            st.metric(
                "Squad value",
                f"£{proj.players.loc[plan.squad_ids, 'price'].sum():.1f}m",
            )
            st.markdown(f"**Captain:** {cap_row['name']} ({cap_row['team']})")
            st.markdown(f"**Vice:** {vc_row['name']} ({vc_row['team']})")

st.divider()
st.subheader("Per-player projections")
render_player_table(
    proj.players, proj.projections,
    gameweek=current_gw, horizon_total=True,
)
