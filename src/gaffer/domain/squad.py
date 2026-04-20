"""Squad, XI, and Bench models with FPL rule validation."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, model_validator

from gaffer.domain.constraints import FPL_RULES
from gaffer.domain.enums import Formation, Position
from gaffer.domain.player import Player


def _count_positions(players: list[Player]) -> dict[Position, int]:
    counts = Counter(p.position for p in players)
    return {pos: counts.get(pos, 0) for pos in Position}


def _count_clubs(players: list[Player]) -> dict[str, int]:
    return dict(Counter(p.team for p in players))


class Squad(BaseModel):
    """A full 15-man FPL squad. Validates positions, budget, and club cap."""

    players: list[Player]

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate(self) -> Squad:
        rules = FPL_RULES
        if len(self.players) != rules.squad_size:
            raise ValueError(
                f"Squad must contain exactly {rules.squad_size} players, got {len(self.players)}"
            )

        counts = _count_positions(self.players)
        for pos, required in rules.squad_quota.items():
            if counts[pos] != required:
                raise ValueError(
                    f"Squad needs {required} {pos.value} but has {counts[pos]}"
                )

        total_price = sum(p.price for p in self.players)
        if total_price > rules.budget + 1e-6:
            raise ValueError(
                f"Squad price {total_price:.1f} exceeds budget {rules.budget:.1f}"
            )

        club_counts = _count_clubs(self.players)
        over_cap = {
            club: n for club, n in club_counts.items() if n > rules.max_per_club
        }
        if over_cap:
            raise ValueError(
                f"Club cap {rules.max_per_club} exceeded: {over_cap}"
            )

        ids = [p.id for p in self.players]
        if len(set(ids)) != len(ids):
            raise ValueError("Squad contains duplicate players")

        return self

    @property
    def total_price(self) -> float:
        return round(sum(p.price for p in self.players), 1)

    def by_position(self, position: Position) -> list[Player]:
        return [p for p in self.players if p.position == position]


class XI(BaseModel):
    """A starting eleven. Validates against allowed FPL formations."""

    players: list[Player]

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate(self) -> XI:
        rules = FPL_RULES
        if len(self.players) != rules.xi_size:
            raise ValueError(
                f"XI must contain exactly {rules.xi_size} players, got {len(self.players)}"
            )

        counts = _count_positions(self.players)
        if counts[Position.GKP] != 1:
            raise ValueError(f"XI needs exactly 1 GKP, got {counts[Position.GKP]}")

        # Validate formation via the Formation enum (raises if invalid triple).
        Formation.from_counts(
            counts[Position.DEF], counts[Position.MID], counts[Position.FWD]
        )
        return self

    @property
    def formation(self) -> Formation:
        counts = _count_positions(self.players)
        return Formation.from_counts(
            counts[Position.DEF], counts[Position.MID], counts[Position.FWD]
        )


class Bench(BaseModel):
    """The four-man bench. Order matters: `players[0]` is the substitute goalkeeper;
    `players[1:4]` are the outfield subs in auto-substitution priority order.
    """

    players: list[Player]

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate(self) -> Bench:
        rules = FPL_RULES
        if len(self.players) != rules.bench_size:
            raise ValueError(
                f"Bench must contain exactly {rules.bench_size} players, got {len(self.players)}"
            )
        if self.players[0].position != Position.GKP:
            raise ValueError("First bench slot must be the substitute goalkeeper")
        if any(p.position == Position.GKP for p in self.players[1:]):
            raise ValueError("Only the first bench slot may be a goalkeeper")
        return self


class SquadSelection(BaseModel):
    """The full output of the optimiser: squad, XI, bench, captain, vice-captain."""

    squad: Squad
    xi: XI
    bench: Bench
    captain: Player
    vice_captain: Player

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate(self) -> SquadSelection:
        squad_ids = {p.id for p in self.squad.players}
        xi_ids = {p.id for p in self.xi.players}
        bench_ids = {p.id for p in self.bench.players}

        if xi_ids | bench_ids != squad_ids:
            raise ValueError("XI + Bench must exactly partition the Squad")
        if xi_ids & bench_ids:
            raise ValueError("XI and Bench overlap")
        if self.captain.id not in xi_ids:
            raise ValueError("Captain must be in the starting XI")
        if self.vice_captain.id not in xi_ids:
            raise ValueError("Vice-captain must be in the starting XI")
        if self.captain.id == self.vice_captain.id:
            raise ValueError("Captain and vice-captain must be different players")
        return self
