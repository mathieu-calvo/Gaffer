"""Historical data provider — reads bundled multi-season CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gaffer.config import settings


class HistoricalCsvProvider:
    """Wraps `clean_merged_gwdf_2016_to_2024.csv` as a `HistoricalDataProvider`."""

    name = "csv"

    def __init__(self, csv_path: Path | None = None) -> None:
        self.csv_path = csv_path or settings.historical_csv

    def get_historical_gwdata(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Historical CSV not found at {self.csv_path}. "
                "Place `clean_merged_gwdf_2016_to_2024.csv` under data/historical/."
            )
        return pd.read_csv(self.csv_path, low_memory=False, index_col=0)
