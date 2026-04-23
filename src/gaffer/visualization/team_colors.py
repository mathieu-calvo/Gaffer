"""Premier League team primary / secondary colours for the simplified jersey.

Keys are the FPL team short codes (e.g., "ARS"). We don't ship real jersey SVGs
— the pitch just uses the primary colour for the marker fill and the secondary
for the border, which is enough visual separation to read the XI at a glance.

If a team isn't in the map (e.g., newly promoted side the season after), the
fallback is the FPL brand purple + white border. Add entries as needed.
"""

from __future__ import annotations

PL_TEAM_COLORS: dict[str, tuple[str, str]] = {
    "ARS": ("#EF0107", "#FFFFFF"),
    "AVL": ("#670E36", "#95BFE5"),
    "BHA": ("#0057B8", "#FFCD00"),
    "BOU": ("#DA291C", "#000000"),
    "BRE": ("#E30613", "#FFFFFF"),
    "BUR": ("#6C1D45", "#99D6EA"),
    "CHE": ("#034694", "#FFFFFF"),
    "CRY": ("#1B458F", "#C4122E"),
    "EVE": ("#003399", "#FFFFFF"),
    "FUL": ("#000000", "#FFFFFF"),
    "IPS": ("#005BAB", "#FFFFFF"),
    "LEE": ("#FFFFFF", "#1D428A"),
    "LEI": ("#003090", "#FDBE11"),
    "LIV": ("#C8102E", "#F6EB61"),
    "LUT": ("#F78F1E", "#FFFFFF"),
    "MCI": ("#6CABDD", "#1C2C5B"),
    "MUN": ("#DA291C", "#FBE122"),
    "NEW": ("#241F20", "#FFFFFF"),
    "NFO": ("#DD0000", "#FFFFFF"),
    "SHU": ("#EE2737", "#000000"),
    "SOU": ("#D71920", "#FFFFFF"),
    "SUN": ("#EB172B", "#FFFFFF"),
    "TOT": ("#132257", "#FFFFFF"),
    "WHU": ("#7A263A", "#1BB1E7"),
    "WOL": ("#FDB913", "#231F20"),
}

DEFAULT_COLOR = ("#37003c", "#FFFFFF")


def team_colors(short_code: str) -> tuple[str, str]:
    """Return (fill, edge) colours for a team short code, with a sensible fallback."""
    return PL_TEAM_COLORS.get(short_code, DEFAULT_COLOR)
