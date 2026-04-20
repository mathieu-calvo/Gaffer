"""Sortable / searchable player projections table for the Streamlit pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_player_table(
    players: pd.DataFrame,
    projections: pd.DataFrame,
    gameweek: int,
    horizon_total: bool = False,
) -> None:
    """Render a sortable, filterable table of player projections.

    Args:
        players: Index=player_id, columns name/team/position/price.
        projections: MultiIndex (player_id, gameweek), columns
            expected_points / lower_80 / upper_80.
        gameweek: GW to surface as the headline projection.
        horizon_total: If True, also show a sum across all gameweeks present in
            `projections` as a `xPts (horizon)` column.
    """
    if gameweek in projections.index.get_level_values("gameweek"):
        gw_proj = projections.xs(gameweek, level="gameweek")
    else:
        gw_proj = pd.DataFrame(
            index=players.index,
            columns=["expected_points", "lower_80", "upper_80"],
            dtype=float,
        ).fillna(0.0)

    table = players.join(gw_proj, how="left").rename(
        columns={
            "name": "Name",
            "team": "Team",
            "position": "Pos",
            "price": "£",
            "expected_points": f"xPts (GW{gameweek})",
            "lower_80": "10%",
            "upper_80": "90%",
        }
    )
    if horizon_total:
        horizon_ep = projections.groupby(level="player_id")["expected_points"].sum()
        table["xPts (horizon)"] = horizon_ep
    table = table.fillna(0.0)
    pos_filter = st.multiselect(
        "Position", options=sorted(table["Pos"].unique()), default=None
    )
    name_filter = st.text_input("Search name", "")
    if pos_filter:
        table = table[table["Pos"].isin(pos_filter)]
    if name_filter:
        table = table[table["Name"].str.contains(name_filter, case=False, na=False)]

    sort_col = f"xPts (GW{gameweek})" if not horizon_total else "xPts (horizon)"
    st.dataframe(
        table.sort_values(sort_col, ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "£": st.column_config.NumberColumn(format="£%.1fm"),
            f"xPts (GW{gameweek})": st.column_config.NumberColumn(format="%.2f"),
            "10%": st.column_config.NumberColumn(format="%.2f"),
            "90%": st.column_config.NumberColumn(format="%.2f"),
            **(
                {"xPts (horizon)": st.column_config.NumberColumn(format="%.2f")}
                if horizon_total else {}
            ),
        },
    )
