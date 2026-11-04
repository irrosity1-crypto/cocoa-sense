"""Simulation engine: revenue of each strategy at every decision date.

Revenue for one decision (all amounts in USD):

    R = Q · [ w · F_t + (1 − w) · P_{t+h} ] − C_fin
    C_fin = r · (h / 12) · (1 − w) · Q · F_t

``C_fin`` is the cost of financing the unsold share until delivery at an annual rate r, valued
at the decision-date forward price so that it is known when the decision is made.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from cocoasense.simulation.futures_provider import FuturesProvider
from cocoasense.simulation.strategies import DecisionContext, Strategy


def revenue(
    w: float, forward: float, delivery: float, quantity: float, rate: float, horizon: int
) -> float:
    """Revenue of one decision (see module docstring)."""
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"w must be in [0, 1], got {w}")
    financing = rate * (horizon / 12) * (1 - w) * quantity * forward
    return quantity * (w * forward + (1 - w) * delivery) - financing


def simulate(
    forecasts: pd.DataFrame,
    strategies: Iterable[Strategy],
    provider: FuturesProvider,
    quantity: float,
    rate: float,
    delivery_override: dict[pd.Timestamp, float] | None = None,
) -> pd.DataFrame:
    """Run every strategy over a table of forecasts for one model and one horizon.

    ``forecasts`` needs ``decision_date, target_date, horizon, y_true, y_pred, lower, upper``.
    ``delivery_override`` replaces the realised delivery price (used for stress scenarios).
    Returns one row per decision date and strategy.
    """
    strategies = list(strategies)
    rows = []
    for rec in forecasts.itertuples(index=False):
        decision = pd.Timestamp(rec.decision_date)
        delivery_date = pd.Timestamp(rec.target_date)
        h = int(rec.horizon)
        forward = provider.forward_price(decision, delivery_date)
        delivered = float(rec.y_true)
        if delivery_override is not None:
            delivered = delivery_override[decision]
        ctx = DecisionContext(
            decision, h, forward, float(rec.y_pred), float(rec.lower), float(rec.upper)
        )
        for strat in strategies:
            w = strat.forward_share(ctx)
            r = revenue(w, forward, delivered, quantity, rate, h)
            rows.append(
                {
                    "decision_date": decision,
                    "target_date": delivery_date,
                    "horizon": h,
                    "strategy": strat.name,
                    "w": w,
                    "forward_price": forward,
                    "delivery_price": delivered,
                    "forecast": float(rec.y_pred),
                    "revenue": r,
                    "revenue_per_t": r / quantity,
                }
            )
    return pd.DataFrame(rows)


def scenario_delivery(
    forecasts: pd.DataFrame, provider: FuturesProvider, factor: float
) -> dict[pd.Timestamp, float]:
    """Hypothetical delivery price = factor × forward price at each decision date."""
    return {
        pd.Timestamp(r.decision_date): factor
        * provider.forward_price(pd.Timestamp(r.decision_date), pd.Timestamp(r.target_date))
        for r in forecasts.itertuples(index=False)
    }
