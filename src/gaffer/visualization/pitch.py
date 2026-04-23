"""Pitch plot for the recommended starting XI.

Renders a stylised football pitch with eleven player markers arranged by
position. Each marker is a team-coloured disc (simplified jersey) with the
team's 3-letter short code inside, and the player's surname underneath.
Captain / vice-captain are suffixed with (C) / (V).

Built on plotly so it composes naturally with Streamlit's `st.plotly_chart`
and renders on Streamlit Community Cloud without extra system deps.
"""

from __future__ import annotations

from collections.abc import Iterable

import plotly.graph_objects as go

from gaffer.domain.enums import Position
from gaffer.visualization.team_colors import team_colors

PITCH_GREEN = "#1e7e34"
PITCH_LINE = "#ffffff"
POSITION_ROW_Y = {
    Position.GKP: 0.10,
    Position.DEF: 0.32,
    Position.MID: 0.58,
    Position.FWD: 0.82,
}


def _add_pitch_shapes(fig: go.Figure) -> None:
    """Draw the pitch background, halfway line, centre circle and penalty boxes."""
    fig.add_shape(
        type="rect", x0=0, x1=1, y0=0, y1=1,
        fillcolor=PITCH_GREEN, line=dict(color=PITCH_LINE, width=2),
        layer="below",
    )
    fig.add_shape(
        type="line", x0=0, x1=1, y0=0.5, y1=0.5,
        line=dict(color=PITCH_LINE, width=2),
        layer="below",
    )
    fig.add_shape(
        type="circle", x0=0.42, x1=0.58, y0=0.43, y1=0.57,
        line=dict(color=PITCH_LINE, width=2),
        layer="below",
    )
    fig.add_shape(
        type="rect", x0=0.25, x1=0.75, y0=0, y1=0.16,
        line=dict(color=PITCH_LINE, width=2),
        layer="below",
    )
    fig.add_shape(
        type="rect", x0=0.25, x1=0.75, y0=0.84, y1=1,
        line=dict(color=PITCH_LINE, width=2),
        layer="below",
    )


def _row_x_positions(n: int) -> list[float]:
    """Evenly space `n` markers across the pitch with breathing room at the edges."""
    if n <= 0:
        return []
    margin = 0.10
    if n == 1:
        return [0.5]
    step = (1.0 - 2 * margin) / (n - 1)
    return [margin + i * step for i in range(n)]


def _short_name(full_name: str) -> str:
    """Use the player's surname only, to keep the pitch readable on narrow screens."""
    parts = full_name.strip().split()
    if len(parts) <= 1:
        return full_name
    return parts[-1]


def _text_color_for(fill_hex: str) -> str:
    """Pick black or white text based on the fill's luminance for readability."""
    hex_clean = fill_hex.lstrip("#")
    r, g, b = int(hex_clean[0:2], 16), int(hex_clean[2:4], 16), int(hex_clean[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.6 else "#ffffff"


def build_pitch_figure(
    starting_xi: Iterable[dict],
    captain_id: int | None = None,
    vice_captain_id: int | None = None,
    title: str = "",
) -> go.Figure:
    """Return a plotly Figure of the starting XI laid out on a pitch.

    Args:
        starting_xi: Iterable of dicts with keys `id`, `name`, `position`
            (Position or str), `expected_points` (float), `team` (str), and
            `team_short` (3-letter code, optional — falls back to first 3 chars
            of `team`).
        captain_id: Player id wearing the armband (suffixes name with " (C)").
        vice_captain_id: Player id of the vice-captain (suffix " (V)").
        title: Optional figure title.
    """
    rows: dict[Position, list[dict]] = {pos: [] for pos in Position}
    for player in starting_xi:
        pos = player["position"]
        if isinstance(pos, str):
            pos = Position(pos)
        rows[pos].append(player)

    fig = go.Figure()
    _add_pitch_shapes(fig)

    # One scatter trace per team so each marker can have its own fill + edge
    # without falling back to a colorscale (plotly doesn't accept per-marker
    # edge colour in a single trace).
    per_team_traces: dict[str, dict] = {}

    # Captions below markers.
    caption_xs: list[float] = []
    caption_ys: list[float] = []
    caption_texts: list[str] = []

    for position, players in rows.items():
        if not players:
            continue
        x_positions = _row_x_positions(len(players))
        y = POSITION_ROW_Y[position]
        for x, player in zip(x_positions, players, strict=True):
            short = player.get("team_short") or player.get("team", "")[:3].upper()
            fill, edge = team_colors(short)
            text_color = _text_color_for(fill)
            trace = per_team_traces.setdefault(
                short,
                dict(
                    xs=[], ys=[], labels=[], hovers=[], text_colors=[],
                    fill=fill, edge=edge,
                ),
            )
            trace["xs"].append(x)
            trace["ys"].append(y)
            trace["labels"].append(short)
            trace["text_colors"].append(text_color)
            trace["hovers"].append(
                f"<b>{player['name']}</b><br>"
                f"{position.value} · {player.get('team', '')}<br>"
                f"xPts: {player['expected_points']:.2f}"
            )

            caption = _short_name(player["name"])
            if player["id"] == captain_id:
                caption += " (C)"
            elif player["id"] == vice_captain_id:
                caption += " (V)"
            caption_xs.append(x)
            caption_ys.append(y)
            caption_texts.append(caption)

    for short, trace in per_team_traces.items():
        fig.add_trace(go.Scatter(
            x=trace["xs"], y=trace["ys"],
            mode="markers+text",
            marker=dict(
                size=38,
                color=trace["fill"],
                line=dict(color=trace["edge"], width=2),
                symbol="circle",
            ),
            text=trace["labels"],
            textposition="middle center",
            textfont=dict(color=trace["text_colors"][0], size=10, family="Arial Black"),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=trace["hovers"],
            name=short,
            showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=caption_xs,
        y=[y - 0.045 for y in caption_ys],
        mode="text",
        text=caption_texts,
        textposition="bottom center",
        textfont=dict(color="white", size=10),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_xaxes(visible=False, range=[-0.05, 1.05])
    fig.update_yaxes(visible=False, range=[-0.05, 1.05])
    fig.update_layout(
        title=title or None,
        plot_bgcolor=PITCH_GREEN,
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        height=520,
        autosize=True,
    )
    return fig
