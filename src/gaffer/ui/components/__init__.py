"""Streamlit widgets shared across pages."""

from gaffer.ui.components.pitch_display import render_pitch
from gaffer.ui.components.player_table import render_player_table
from gaffer.ui.components.squad_input import squad_id_picker

__all__ = ["render_pitch", "render_player_table", "squad_id_picker"]
