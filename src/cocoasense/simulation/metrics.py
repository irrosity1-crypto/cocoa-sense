"""Strategy outcome metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

FIXED_NAMES = ("sell_forward", "wait", "fixed_50_50")


def worst_share_mean(values: np.ndarray, share: float = 0.10) -> float:
    """Mean of the worst ``share`` of outcomes (at least one observation)."""
    k = max(1, int(np.ceil(share * values.size)))
    return float(np.sort(values)[:k].mean())


def strategy_metrics(results: pd.DataFrame, value: str = "revenue_per_t") -> pd.DataFrame:
    """Per-strategy summary of a simulation run (``value``: revenue per tonne, US$ by default).

    * ``regret``: mean shortfall against the best of the fixed strategies at each date.
    * ``win_rate_vs_5050``: share of dates where the strategy earns strictly more than 50/50.
    """
    wide = results.pivot_table(index="decision_date", columns="strategy", values=value)
    fixed = [c for c in FIXED_NAMES if c in wide]
    best_fixed = wide[fixed].max(axis=1)
    rows = []
    for name in wide.columns:
        v = wide[name].to_numpy(float)
        rows.append(
            {
                "strategy": name,
                "n_dates": int(v.size),
                "mean_revenue_per_t": float(v.mean()),
                "std_revenue_per_t": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
                "worst10_mean_per_t": worst_share_mean(v),
                "min_revenue_per_t": float(v.min()),
                "regret_vs_best_fixed": float((best_fixed - wide[name]).mean()),
                "win_rate_vs_5050": float((wide[name] > wide["fixed_50_50"]).mean())
                if "fixed_50_50" in wide
                else float("nan"),
                "mean_w": float(results.loc[results["strategy"] == name, "w"].mean()),
            }
        )
    return pd.DataFrame(rows)


def metrics_by_period(
    results: pd.DataFrame,
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp | None]],
    value: str = "revenue_per_t",
) -> pd.DataFrame:
    """Strategy metrics for the full run and each period (by delivery month)."""
    parts = [strategy_metrics(results, value).assign(period="full")]
    for name, (start, end) in periods.items():
        mask = results["target_date"] >= start
        if end is not None:
            mask &= results["target_date"] <= end
        if mask.any():
            parts.append(strategy_metrics(results.loc[mask], value).assign(period=name))
    return pd.concat(parts, ignore_index=True)
