"""FPL rule constants used across domain validation and the MILP optimiser."""

from __future__ import annotations

from dataclasses import dataclass

from gaffer.domain.enums import Position


@dataclass(frozen=True)
class FplRules:
    """Structural rules of the Fantasy Premier League game.

    Used by the Squad validator and the MILP as canonical constants. Mutating
    these is pointless — they are the game definition — but keeping them in one
    place means a rule change (e.g., budget raised to 105m) edits in exactly one
    spot.
    """

    budget: float = 100.0
    squad_size: int = 15
    xi_size: int = 11
    bench_size: int = 4
    max_per_club: int = 3

    squad_quota: dict[Position, int] = None  # type: ignore[assignment]
    formation_min: dict[Position, int] = None  # type: ignore[assignment]
    formation_max: dict[Position, int] = None  # type: ignore[assignment]

    transfer_hit_cost: int = 4
    default_free_transfers: int = 1

    def __post_init__(self) -> None:
        if self.squad_quota is None:
            object.__setattr__(
                self,
                "squad_quota",
                {Position.GKP: 2, Position.DEF: 5, Position.MID: 5, Position.FWD: 3},
            )
        if self.formation_min is None:
            object.__setattr__(
                self,
                "formation_min",
                {Position.GKP: 1, Position.DEF: 3, Position.MID: 2, Position.FWD: 1},
            )
        if self.formation_max is None:
            object.__setattr__(
                self,
                "formation_max",
                {Position.GKP: 1, Position.DEF: 5, Position.MID: 5, Position.FWD: 3},
            )


FPL_RULES = FplRules()
