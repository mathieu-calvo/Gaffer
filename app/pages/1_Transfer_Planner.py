"""Transfer Planner — recommends transfers from a user-supplied current squad."""

from __future__ import annotations

import streamlit as st

from gaffer.config import settings
from gaffer.providers.fpl_api import LiveFplApiProvider
from gaffer.providers.historical_csv import HistoricalCsvProvider
from gaffer.services.model_cache import train_or_load_ensembles
from gaffer.services.optimization_service import optimize_squad
from gaffer.services.prediction_service import predict_projections
from gaffer.ui.components import build_player_label_map, render_pitch, squad_id_picker

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


# Apply any squad/bank prefill from a prior "Import squad" click before widgets
# are instantiated — Streamlit forbids mutating a widget's session_state key
# after the widget has been rendered in the current run.
if "_tp_bank_pending" in st.session_state:
    st.session_state["tp_bank"] = st.session_state.pop("_tp_bank_pending")

st.title("🔁 Transfer Planner")
st.markdown(
    "Pick your current 15-man squad and the optimiser recommends transfers, "
    "trading raw expected points against the 4-point hit cost above your free "
    "transfer allowance."
)

with st.sidebar:
    horizon = st.slider("Planning horizon (gameweeks)", 1, 5, settings.horizon)
    bank = st.number_input(
        "Money in the bank (£m)", min_value=0.0, step=0.1, key="tp_bank"
    )
    free_transfers = st.slider(
        "Free transfers this GW", 0, 5, key="tp_free_transfers",
        value=st.session_state.get("tp_free_transfers", 1),
    )

try:
    proj, current_gw = load_projections(horizon)
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load projections: {exc}")
    st.stop()

with st.expander("Import squad by FPL manager ID", expanded=False):
    st.caption(
        "Your manager id is in the URL of your FPL team page "
        "(e.g. fantasy.premierleague.com/entry/**1234567**/event/12)."
    )
    col_id, col_btn = st.columns([3, 1])
    with col_id:
        manager_id = st.text_input("FPL manager ID", key="tp_manager_id", value="")
    with col_btn:
        st.write("")
        import_clicked = st.button("Import squad", use_container_width=True)

    if import_clicked:
        if not manager_id.strip().isdigit():
            st.error("Please enter a numeric FPL manager ID.")
        else:
            try:
                fpl = LiveFplApiProvider()
                entry = fpl.get_manager_entry(int(manager_id))
                pick_gw = int(entry.get("current_event") or current_gw)
                picked_ids = fpl.get_manager_picks(int(manager_id), pick_gw)
                label_map = build_player_label_map(proj.players)
                id_to_label = {v: k for k, v in label_map.items()}
                missing = [pid for pid in picked_ids if pid not in id_to_label]
                squad_labels = [id_to_label[pid] for pid in picked_ids if pid in id_to_label]
                st.session_state["transfer_squad"] = squad_labels
                bank_raw = entry.get("last_deadline_bank")
                if bank_raw is not None:
                    st.session_state["_tp_bank_pending"] = round(float(bank_raw) / 10.0, 1)
                msg = (
                    f"Imported {len(squad_labels)} players from GW {pick_gw} "
                    f"for manager `{entry.get('player_first_name', '')} "
                    f"{entry.get('player_last_name', '')}`."
                )
                if missing:
                    msg += f" {len(missing)} player(s) not available in current projections."
                st.success(msg)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not import squad: {exc}")

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
