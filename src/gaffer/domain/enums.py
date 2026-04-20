"""Domain enumerations."""

from __future__ import annotations

from enum import Enum


class Position(str, Enum):  # noqa: UP042 - StrEnum not available on 3.10
    """FPL player position. Values match the live API's `singular_name_short`."""

    GKP = "GKP"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"

    @classmethod
    def from_fpl(cls, value: str) -> Position:
        """Parse a raw FPL position string, tolerating the legacy `GK` alias."""
        if value == "GK":
            return cls.GKP
        return cls(value)


class Formation(tuple, Enum):
    """Valid FPL starting-XI formations as (DEF, MID, FWD) triples.

    Goalkeeper count is always 1; total outfielders always 10. Formations outside
    this set are invalid (minimum 3 DEF, 2 MID, 1 FWD; max 5 DEF, 5 MID, 3 FWD).
    """

    F_3_4_3 = (3, 4, 3)
    F_3_5_2 = (3, 5, 2)
    F_4_3_3 = (4, 3, 3)
    F_4_4_2 = (4, 4, 2)
    F_4_5_1 = (4, 5, 1)
    F_5_2_3 = (5, 2, 3)
    F_5_3_2 = (5, 3, 2)
    F_5_4_1 = (5, 4, 1)

    @property
    def defenders(self) -> int:
        return self.value[0]

    @property
    def midfielders(self) -> int:
        return self.value[1]

    @property
    def forwards(self) -> int:
        return self.value[2]

    @classmethod
    def from_counts(cls, defenders: int, midfielders: int, forwards: int) -> Formation:
        """Resolve a formation from its (DEF, MID, FWD) triple; raises if invalid."""
        return cls((defenders, midfielders, forwards))
