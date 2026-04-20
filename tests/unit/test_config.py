"""Settings / config tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gaffer.config import Settings


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.data_source in ("live", "csv")
        assert s.horizon == 3
        assert 0 < s.ewma_alpha <= 1.0

    def test_horizon_bounded(self):
        with pytest.raises(ValidationError):
            Settings(horizon=10)

    def test_solver_time_limit_min(self):
        with pytest.raises(ValidationError):
            Settings(solver_time_limit=1)

    def test_ewma_alpha_bounded(self):
        with pytest.raises(ValidationError):
            Settings(ewma_alpha=1.5)
