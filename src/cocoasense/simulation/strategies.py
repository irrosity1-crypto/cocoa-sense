"""Selling strategies. Each returns w, the share of the quantity sold forward at decision time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecisionContext:
    """Everything known at the end of the decision month."""

    decision_date: pd.Timestamp
    horizon: int
    forward_price: float  # F_t, USD/tonne
    forecast: float  # model point forecast of the delivery-month price
    lower: float  # interval bounds of that forecast
    upper: float


class Strategy(Protocol):
    """A rule mapping a decision context to a forward share w in [0, 1]."""

    name: str

    def forward_share(self, ctx: DecisionContext) -> float:
        """Share sold forward."""
        ...


class SellForward:
    """Sell everything at the price available at decision time."""

    name = "sell_forward"

    def forward_share(self, ctx: DecisionContext) -> float:
        """Always 1."""
        return 1.0


class Wait:
    """Sell everything at the delivery-month price."""

    name = "wait"

    def forward_share(self, ctx: DecisionContext) -> float:
        """Always 0."""
        return 0.0


class Fixed5050:
    """Sell half forward, half at delivery."""

    name = "fixed_50_50"

    def forward_share(self, ctx: DecisionContext) -> float:
        """Always 0.5."""
        return 0.5


class ForecastDriven:
    """w = clip(0.5 + a·z, 0, 1) with z = (F_t − forecast) / interval half-width.

    If the model expects the price to fall below today's forward price (z > 0) it sells more
    forward; if it expects a rise (z < 0) it waits with more. Dividing by the interval
    half-width shrinks the bet when the model is uncertain. ``a = 0`` reproduces Fixed 50/50.
    """

    name = "forecast_driven"

    def __init__(self, a: float) -> None:
        if a < 0:
            raise ValueError("a must be non-negative")
        self.a = a

    def forward_share(self, ctx: DecisionContext) -> float:
        """Forecast- and uncertainty-scaled forward share."""
        half_width = (ctx.upper - ctx.lower) / 2
        if not np.isfinite(half_width) or half_width <= 0:
            return 0.5
        z = (ctx.forward_price - ctx.forecast) / half_width
        return float(np.clip(0.5 + self.a * z, 0.0, 1.0))


FIXED_STRATEGIES: tuple[Strategy, ...] = (SellForward(), Wait(), Fixed5050())
