"""Streamlit widgets shared across pages."""

from gaffer.ui.components.pitch_display import render_pitch
from gaffer.ui.components.player_table import render_player_table
from gaffer.ui.components.squad_input import build_player_label_map, squad_id_picker

__all__ = [
    "build_player_label_map",
    "render_pitch",
    "render_player_table",
    "squad_id_picker",
]
