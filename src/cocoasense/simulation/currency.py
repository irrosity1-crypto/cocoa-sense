"""Cedi revenue view: the simulated US$ revenue expressed in Ghana cedis.

A reporting view only; no decision changes. Every delivery (forward and spot shares alike) is paid
in US dollars in the delivery month, so it is converted at that month's average GHS per US$ rate
(IMF). Deliveries without an observed rate are dropped, never extrapolated. Because all strategies
for a decision date share one delivery month and one rate, the per-date ranking of strategies is
the same as in US$; only the weighting across dates changes (later, weaker-cedi months count more).
"""

from __future__ import annotations

import pandas as pd


def to_cedis(results: pd.DataFrame, ghs_per_usd: pd.Series) -> pd.DataFrame:
    """Add ``ghs_per_usd``, ``revenue_ghs`` and ``revenue_ghs_per_t`` at the delivery-month rate.

    Rows whose delivery month has no observed exchange rate are removed.
    """
    fx = ghs_per_usd.dropna()
    out = results.assign(ghs_per_usd=results["target_date"].map(fx))
    out = out.dropna(subset=["ghs_per_usd"]).reset_index(drop=True)
    return out.assign(
        revenue_ghs=out["revenue"] * out["ghs_per_usd"],
        revenue_ghs_per_t=out["revenue_per_t"] * out["ghs_per_usd"],
    )


def annual_cedi(cedi: pd.DataFrame) -> pd.DataFrame:
    """Mean revenue per tonne in US$ and cedis, and the mean rate, by delivery year and strategy."""
    year = cedi["target_date"].dt.year.rename("delivery_year")
    return (
        cedi.groupby([year, "strategy"])
        .agg(
            n_deliveries=("revenue_per_t", "size"),
            mean_ghs_per_usd=("ghs_per_usd", "mean"),
            mean_revenue_usd_per_t=("revenue_per_t", "mean"),
            mean_revenue_ghs_per_t=("revenue_ghs_per_t", "mean"),
        )
        .reset_index()
    )
