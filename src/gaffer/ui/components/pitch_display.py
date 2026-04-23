"""Streamlit wrapper around the pitch figure."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from gaffer.optimizer.result import GameweekPlan
from gaffer.visualization.pitch import build_pitch_figure


def render_pitch(
    plan: GameweekPlan,
    players: pd.DataFrame,
    projections: pd.DataFrame,
    title: str | None = None,
) -> None:
    """Render the pitch + captain armband + bench summary for one gameweek plan.

    `players` is indexed by player_id (cols name, team, position, price).
    `projections` is indexed by (player_id, gameweek) with `expected_points`.
    Players with a blank gameweek (no fixture in `plan.gameweek`) are shown
    with 0 xPts rather than crashing the render.
    """
    def _xp(pid: int) -> float:
        if (pid, plan.gameweek) in projections.index:
            return float(projections.loc[(pid, plan.gameweek), "expected_points"])
        return 0.0

    def _team_short(pid: int) -> str:
        if "team_short" in players.columns:
            short = players.loc[pid, "team_short"]
            if isinstance(short, str) and short:
                return short
        return str(players.loc[pid, "team"])[:3].upper()

    xi_payload = [
        {
            "id": int(pid),
            "name": str(players.loc[pid, "name"]),
            "team": str(players.loc[pid, "team"]),
            "team_short": _team_short(pid),
            "position": str(players.loc[pid, "position"]),
            "expected_points": _xp(pid),
        }
        for pid in plan.xi_ids
    ]
    fig = build_pitch_figure(
        xi_payload,
        captain_id=plan.captain_id,
        vice_captain_id=plan.vice_captain_id,
        title=title or f"Starting XI · GW {plan.gameweek}",
    )
    st.plotly_chart(fig, use_container_width=True)

    bench_rows = [
        {
            "Slot": ["GK", "Sub 1", "Sub 2", "Sub 3"][i],
            "Name": str(players.loc[pid, "name"]),
            "Pos": str(players.loc[pid, "position"]),
            "Team": str(players.loc[pid, "team"]),
            "xPts": _xp(pid),
        }
        for i, pid in enumerate(plan.bench_ids)
    ]
    st.markdown("**Bench**")
    st.dataframe(
        pd.DataFrame(bench_rows),
        hide_index=True,
        use_container_width=True,
    )
