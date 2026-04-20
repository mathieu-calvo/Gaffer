"""Factory for data providers, selected by `settings.data_source`."""

from __future__ import annotations

from gaffer.config import settings
from gaffer.providers.base import FplDataProvider, HistoricalDataProvider
from gaffer.providers.fpl_api import LiveFplApiProvider
from gaffer.providers.historical_csv import HistoricalCsvProvider


def get_fpl_provider(season: str = "2024-25") -> FplDataProvider:
    """Return a live FPL data provider. CSV source has no live equivalent."""
    if settings.data_source == "csv":
        raise ValueError(
            "CSV data source has no live FPL provider. Set GAFFER_DATA_SOURCE=live."
        )
    return LiveFplApiProvider(season=season)


def get_historical_provider() -> HistoricalDataProvider:
    """Return the historical multi-season training data provider."""
    return HistoricalCsvProvider()
