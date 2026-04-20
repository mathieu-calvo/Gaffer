"""Transfer Planner — recommends transfers from a user-supplied current squad."""

from __future__ import annotations

import streamlit as st

from gaffer.config import settings
from gaffer.providers.fpl_api import LiveFplApiProvider
from gaffer.providers.historical_csv import HistoricalCsvProvider
from gaffer.services.model_cache import train_or_load_ensembles
from gaffer.services.optimization_service import optimize_squad
from gaffer.services.prediction_service import predict_projections
from gaffer.ui.components import render_pitch, squad_id_picker

st.set_page_config(
    page_title="Gaffer · Transfer Planner",
    page_icon="🔁",
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
    return proj, fpl.get_current_gw()


st.title("🔁 Transfer Planner")
st.markdown(
    "Pick your current 15-man squad and the optimiser recommends transfers, "
    "trading raw expected points against the 4-point hit cost above your free "
    "transfer allowance."
)

with st.sidebar:
    horizon = st.slider("Planning horizon (gameweeks)", 1, 5, settings.horizon)
    bank = st.number_input("Money in the bank (£m)", min_value=0.0, value=0.0, step=0.1)
    free_transfers = st.slider("Free transfers this GW", 0, 5, 1)

try:
    proj, current_gw = load_projections(horizon)
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load projections: {exc}")
    st.stop()

selected_ids = squad_id_picker(proj.players, key="transfer_squad")
if len(selected_ids) != 15:
    st.info(f"Select your current 15-man squad. Selected so far: {len(selected_ids)}/15.")
    st.stop()

with st.spinner("Solving transfer plan…"):
    result = optimize_squad(
        projections=proj.projections,
        players=proj.players,
        start_gw=current_gw,
        horizon=horizon,
        initial_squad_ids=selected_ids,
        bank=bank,
        free_transfers=free_transfers,
    )

st.success(
    f"Solver status: {result.solver_status} · "
    f"net expected points over horizon: {result.total_expected_points:.1f}"
)

for plan in result.plans:
    st.subheader(f"GW {plan.gameweek}")
    if plan.transfers_in:
        in_names = [proj.players.loc[pid, "name"] for pid in plan.transfers_in]
        out_names = [proj.players.loc[pid, "name"] for pid in plan.transfers_out]
        st.markdown(
            f"**Transfers:** {len(plan.transfers_in)} "
            f"(hit cost: −{plan.hit_cost} pts)"
        )
        col_in, col_out = st.columns(2)
        with col_in:
            st.markdown("**IN**")
            for name in in_names:
                st.markdown(f"- {name}")
        with col_out:
            st.markdown("**OUT**")
            for name in out_names:
                st.markdown(f"- {name}")
    else:
        st.markdown("_No transfers — the existing squad is already optimal._")
    render_pitch(plan, proj.players, proj.projections, title=None)
