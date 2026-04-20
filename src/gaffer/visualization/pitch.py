"""Pitch plot for the recommended starting XI.

Renders a stylised football pitch with eleven player markers arranged by
position. Each marker shows the player's name, expected points, and a captain
armband when applicable.

Built on plotly so it composes naturally with Streamlit's `st.plotly_chart`
and renders on Streamlit Community Cloud without extra system deps.
"""

from __future__ import annotations

from collections.abc import Iterable

import plotly.graph_objects as go

from gaffer.domain.enums import Position

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
    )
    fig.add_shape(
        type="line", x0=0, x1=1, y0=0.5, y1=0.5,
        line=dict(color=PITCH_LINE, width=2),
    )
    fig.add_shape(
        type="circle", x0=0.42, x1=0.58, y0=0.43, y1=0.57,
        line=dict(color=PITCH_LINE, width=2),
    )
    # Penalty boxes (bottom = home side, top = away)
    fig.add_shape(
        type="rect", x0=0.25, x1=0.75, y0=0, y1=0.16,
        line=dict(color=PITCH_LINE, width=2),
    )
    fig.add_shape(
        type="rect", x0=0.25, x1=0.75, y0=0.84, y1=1,
        line=dict(color=PITCH_LINE, width=2),
    )


def _row_x_positions(n: int) -> list[float]:
    """Evenly space `n` markers across the pitch with breathing room at the edges."""
    if n <= 0:
        return []
    margin = 0.12
    if n == 1:
        return [0.5]
    step = (1.0 - 2 * margin) / (n - 1)
    return [margin + i * step for i in range(n)]


def build_pitch_figure(
    starting_xi: Iterable[dict],
    captain_id: int | None = None,
    vice_captain_id: int | None = None,
    title: str = "",
) -> go.Figure:
    """Return a plotly Figure of the starting XI laid out on a pitch.

    Args:
        starting_xi: Iterable of dicts with keys `id`, `name`, `position`
            (Position or str), `expected_points` (float), `team` (str).
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

    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    hovers: list[str] = []
    for position, players in rows.items():
        if not players:
            continue
        x_positions = _row_x_positions(len(players))
        y = POSITION_ROW_Y[position]
        for x, player in zip(x_positions, players, strict=True):
            xs.append(x)
            ys.append(y)
            label = player["name"]
            if player["id"] == captain_id:
                label += " (C)"
            elif player["id"] == vice_captain_id:
                label += " (V)"
            labels.append(label)
            hovers.append(
                f"<b>{player['name']}</b><br>"
                f"{position.value} · {player.get('team', '')}<br>"
                f"xPts: {player['expected_points']:.2f}"
            )

    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(size=36, color="#37003c", line=dict(color="white", width=2)),
        text=labels,
        textposition="bottom center",
        textfont=dict(color="white", size=11),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hovers,
        showlegend=False,
    ))

    fig.update_xaxes(visible=False, range=[-0.05, 1.05])
    fig.update_yaxes(visible=False, range=[-0.05, 1.05])
    fig.update_layout(
        title=title or None,
        plot_bgcolor=PITCH_GREEN,
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=40 if title else 20, b=20),
        height=520,
    )
    return fig
