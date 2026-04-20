"""Multi-select widget for entering a current 15-man squad by player name."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def squad_id_picker(
    players: pd.DataFrame,
    label: str = "Your current squad (pick 15)",
    key: str = "squad_picker",
) -> list[int]:
    """Render a multiselect of player names; return the chosen player_ids.

    `players` is indexed by player_id. The widget displays "Name · Team · £p"
    for disambiguation but returns ids so downstream code can hand them
    directly to the optimiser.
    """
    label_to_id: dict[str, int] = {
        f"{row['name']} · {row['team']} · £{row['price']:.1f}m": int(pid)
        for pid, row in players.iterrows()
    }
    chosen_labels = st.multiselect(
        label,
        options=sorted(label_to_id.keys()),
        max_selections=15,
        key=key,
    )
    return [label_to_id[label] for label in chosen_labels]
