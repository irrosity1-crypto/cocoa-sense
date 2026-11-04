"""Forward-price providers.

``spot_proxy`` (default, free data only): the forward price for delivery at t+h is approximated
by the monthly average spot price at the decision month t. This ignores the futures basis
(contango/backwardation) and is the main simplification of the simulation.

``futures``: reads a user-supplied CSV of contract settlements with columns
``date, contract_month, settle`` (USD/tonne). No futures data is fetched by this project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd


class FuturesProvider(Protocol):
    """Returns the forward price known at the end of ``decision_date`` for delivery month."""

    mode: str

    def forward_price(self, decision_date: pd.Timestamp, delivery: pd.Timestamp) -> float:
        """Forward price in USD/tonne."""
        ...


class SpotProxyProvider:
    """F_t = monthly average spot price of the decision month."""

    mode = "spot_proxy"

    def __init__(self, price: pd.Series) -> None:
        self.price = price

    def forward_price(self, decision_date: pd.Timestamp, delivery: pd.Timestamp) -> float:
        """Spot price at the decision month (delivery month is ignored)."""
        return float(self.price.loc[decision_date])


class CsvFuturesProvider:
    """Settlement of the nearest contract expiring in or after the delivery month.

    Uses the last settlement on or before the end of the decision month, so only prices known
    at decision time are used.
    """

    mode = "futures"

    def __init__(self, path: Path) -> None:
        df = pd.read_csv(path, parse_dates=["date"])
        missing = {"date", "contract_month", "settle"} - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        df["contract_month"] = pd.to_datetime(df["contract_month"]).dt.to_period("M").dt.start_time
        self.df = df.sort_values("date")

    def forward_price(self, decision_date: pd.Timestamp, delivery: pd.Timestamp) -> float:
        """Settle of the first contract maturing in/after delivery, as of decision month end."""
        cutoff = decision_date + pd.offsets.MonthEnd(0)
        known = self.df[(self.df["date"] <= cutoff) & (self.df["contract_month"] >= delivery)]
        if known.empty:
            raise LookupError(
                f"no futures settle for delivery {delivery:%Y-%m} by {cutoff:%Y-%m-%d}"
            )
        nearest = known["contract_month"].min()
        row = known[known["contract_month"] == nearest].iloc[-1]
        return float(row["settle"])
