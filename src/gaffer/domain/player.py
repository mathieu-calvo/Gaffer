"""Player and projection domain models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from gaffer.domain.enums import Position


class Player(BaseModel):
    """A single FPL player. Prices are stored in millions (e.g., 7.5 = £7.5m)."""

    id: int
    name: str
    team: str
    position: Position
    price: float = Field(gt=0.0)
    chance_of_playing: int | None = Field(default=None, ge=0, le=100)

    model_config = {"frozen": True}


class PlayerProjection(BaseModel):
    """A model's expected-points forecast for one player in one gameweek.

    `lower_80` and `upper_80` are 10th / 90th percentile estimates from the
    quantile-regression model. Equal to `expected_points` when intervals aren't
    available.
    """

    player: Player
    gameweek: int = Field(ge=1, le=38)
    expected_points: float
    lower_80: float
    upper_80: float

    model_config = {"frozen": True}

    def __init__(self, **data):  # type: ignore[no-untyped-def]
        super().__init__(**data)
        if not (self.lower_80 <= self.expected_points <= self.upper_80):
            raise ValueError(
                "Prediction interval must satisfy lower_80 <= expected_points <= upper_80, "
                f"got ({self.lower_80}, {self.expected_points}, {self.upper_80})"
            )
