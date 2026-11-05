"""Success/failure scorecards, evidence-based confidence for each forecast, and plain-language text.

Success criteria (``configs/insights.yaml``): a forecast succeeds at X % if the actual price is
within X % of it; the direction is correct if the forecast and the actual moved the same way
from the decision-month price; the interval succeeds if it contains the actual price.

Confidence is estimated ex ante: for a forecast made at month t, only that model's earlier errors
at the same horizon whose target month was already known at t are used. Each past log error is
rescaled by current volatility / volatility when it was made, so the estimate widens in turbulent
markets. From these errors: p_within_5 / p_within_10 (chance the actual lands within 5 % / 10 %)
and p_up (chance the actual is above the decision-month price). Agreement between models adds a
second signal. The label (High / Medium / Low) combines both with fixed thresholds. These are
statements about past performance of similar forecasts, not guarantees.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cocoasense.config import InsightsConfig

NOT_ENOUGH = "Not enough history"
LABELS = ("High", "Medium", "Low", NOT_ENOUGH)


# ---------------------------------------------------------------- outcomes


def outcomes(preds: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """Add error columns and success flags to rows whose actual price is known."""
    out = preds[preds["y_true"].notna()].copy()
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["abs_pct_error"] = out["abs_error"] / out["y_true"] * 100
    for t in thresholds:
        out[f"within_{t:g}pct"] = (out["abs_pct_error"] <= t).astype(float)
    pred_move = np.sign(out["y_pred"] - out["last_obs"])
    act_move = np.sign(out["y_true"] - out["last_obs"])
    direction = (pred_move == act_move).astype(float)
    out["direction_correct"] = direction.where((pred_move != 0) & (act_move != 0))
    has_band = out["lower"].notna() & out["upper"].notna()
    inside = (out["y_true"] >= out["lower"]) & (out["y_true"] <= out["upper"])
    out["in_interval"] = inside.astype(float).where(has_band)
    return out


def scorecard(scored: pd.DataFrame, by: list[str], thresholds: list[float]) -> pd.DataFrame:
    """Success rates (%) and error sizes per group."""
    named: dict[str, tuple[str, str]] = {
        "n": ("y_true", "size"),
        "mape": ("abs_pct_error", "mean"),
        "mae": ("abs_error", "mean"),
        "direction_correct_pct": ("direction_correct", "mean"),
        "in_interval_pct": ("in_interval", "mean"),
    }
    for t in thresholds:
        named[f"within_{t:g}pct_pct"] = (f"within_{t:g}pct", "mean")
    table = scored.groupby(by, dropna=False).agg(**named).reset_index()
    pct_cols = [c for c in table if c.endswith("_pct")]
    table[pct_cols] = table[pct_cols] * 100
    return table


# -------------------------------------------------------------- confidence


def volatility(price: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of monthly log returns (known at the end of each month)."""
    out: pd.Series = np.log(price).diff().rolling(window, min_periods=max(3, window // 2)).std()
    return out


def model_spread(preds: pd.DataFrame) -> pd.Series:
    """(max - min) / last price of non-naive point forecasts per (decision_date, horizon), in %."""
    rest = preds[~preds["model"].str.startswith("naive")]
    g = rest.groupby(["decision_date", "horizon"])
    spread = (g["y_pred"].max() - g["y_pred"].min()) / g["last_obs"].first() * 100
    return spread.rename("model_spread_pct")


def label(p_within_10: float, spread: float, cfg: InsightsConfig) -> str:
    """High / Medium / Low from the estimated hit probability and model agreement."""
    if math.isnan(p_within_10):
        return NOT_ENOUGH
    spread = 0.0 if math.isnan(spread) else spread
    if p_within_10 < cfg.low.p_within_10 or spread > cfg.low.model_spread_pct:
        return "Low"
    if p_within_10 >= cfg.high.p_within_10 and spread <= cfg.high.model_spread_pct:
        return "High"
    return "Medium"


def ex_ante_confidence(
    rows: pd.DataFrame,
    history: pd.DataFrame,
    price: pd.Series,
    cfg: InsightsConfig,
) -> pd.DataFrame:
    """Confidence columns for ``rows`` using only errors in ``history`` known at each decision.

    ``history`` holds earlier forecasts with known actuals (same columns as ``rows``); an error
    counts for a row only if it is for the same model and horizon and its target month is on or
    before the row's decision month.
    """
    vol = volatility(price, cfg.volatility_window)
    past = history[history["y_true"].notna()].copy()
    past["log_err"] = np.log(past["y_true"] / past["y_pred"])
    past["vol"] = past["decision_date"].map(vol)
    past = past[past["vol"].notna()]
    groups = {k: g for k, g in past.groupby(["model", "horizon"])}
    keys = ("n_past_errors", "p_within_5", "p_within_10", "p_up", "vol_ratio")
    cols: dict[str, list[float]] = {k: [] for k in keys}
    for r in rows.itertuples(index=False):
        g = groups.get((r.model, r.horizon))
        known = g[g["target_date"] <= r.decision_date] if g is not None else past.iloc[:0]
        v_now = vol.get(r.decision_date, np.nan)
        if len(known) < cfg.min_past_errors or not np.isfinite(v_now):
            for k in keys:
                cols[k].append(float(len(known)) if k == "n_past_errors" else np.nan)
            continue
        ratio = v_now / known["vol"].to_numpy()
        e = known["log_err"].to_numpy() * ratio
        ape = np.abs(1 - np.exp(-e))  # |actual - forecast| / actual when actual = forecast * e^e
        cols["n_past_errors"].append(float(len(known)))
        cols["p_within_5"].append(float((ape <= 0.05).mean()))
        cols["p_within_10"].append(float((ape <= 0.10).mean()))
        cols["p_up"].append(float((r.y_pred * np.exp(e) > r.last_obs).mean()))
        cols["vol_ratio"].append(float(np.median(ratio)))
    out = rows.copy()
    for k, v in cols.items():
        out[k] = v
    spread = model_spread(rows)
    keys_dh = zip(out["decision_date"], out["horizon"], strict=True)
    out["model_spread_pct"] = [spread.get((d, h), np.nan) for d, h in keys_dh]
    out["confidence"] = [
        label(p, s, cfg) for p, s in zip(out["p_within_10"], out["model_spread_pct"], strict=True)
    ]
    return out


def confidence_check(scored: pd.DataFrame) -> pd.DataFrame:
    """Did the labels mean anything? Actual hit rates per confidence label (rows with outcomes)."""
    present = [lab for lab in LABELS if lab in set(scored["confidence"])]
    table = (
        scored.groupby("confidence")
        .agg(
            n=("y_true", "size"),
            predicted_p_within_10=("p_within_10", "mean"),
            actual_within_10=("within_10pct", "mean"),
            actual_within_5=("within_5pct", "mean"),
            mape=("abs_pct_error", "mean"),
        )
        .reindex(present)
        .reset_index()
    )
    for c in ("predicted_p_within_10", "actual_within_10", "actual_within_5"):
        table[c] = table[c] * 100
    return table


def probability_calibration(scored: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Reliability of p_up: predicted chance of a rise vs how often the price actually rose."""
    s = scored[scored["p_up"].notna() & (scored["y_true"] != scored["last_obs"])].copy()
    s["rose"] = (s["y_true"] > s["last_obs"]).astype(float)
    s["bin"] = pd.cut(s["p_up"], np.linspace(0, 1, bins + 1), include_lowest=True)
    table = (
        s.groupby("bin", observed=True)
        .agg(n=("rose", "size"), predicted=("p_up", "mean"), actual=("rose", "mean"))
        .reset_index()
    )
    table["bin"] = table["bin"].astype(str)
    return table


def brier(scored: pd.DataFrame) -> float:
    """Brier score of p_up (0 is perfect, 0.25 is a coin flip)."""
    s = scored[scored["p_up"].notna() & (scored["y_true"] != scored["last_obs"])]
    if s.empty:
        return float("nan")
    rose = (s["y_true"] > s["last_obs"]).astype(float)
    return float(((s["p_up"] - rose) ** 2).mean())


# --------------------------------------------------------------- language

ADVICE = {
    "High": "Confidence is HIGH: forecasts like this have usually been close.",
    "Medium": "Confidence is MEDIUM: useful as a guide, but expect a sizeable error.",
    "Low": "Confidence is LOW: treat this forecast with caution; a fixed selling split is safer "
    "than acting on it.",
}


DIRECTION_WARNING = (
    " Past up/down estimates like this one were no better than a coin flip, so do not rely on "
    "the direction."
)


def explain(row: pd.Series, model_label: str, direction_reliable: bool = True) -> str:
    """One paragraph in plain English describing a forecast and how far to trust it.

    ``direction_reliable=False`` (a Brier score of p_up no better than a coin flip) adds a
    warning after the up/down chance.
    """
    move = (row["y_pred"] / row["last_obs"] - 1) * 100
    target, base = f"{row['target_date']:%B %Y}", f"{row['decision_date']:%B %Y}"
    if abs(move) < 0.5:
        text = (
            f"{model_label} expects the {target} average price to stay close to the {base} "
            f"price of US${row['last_obs']:,.0f}/t (forecast US${row['y_pred']:,.0f}/t)."
        )
    else:
        way = "above" if move > 0 else "below"
        text = (
            f"{model_label} expects the {target} average price to be about "
            f"US${row['y_pred']:,.0f}/t, {abs(move):.1f} % {way} the {base} price of "
            f"US${row['last_obs']:,.0f}/t."
        )
    if pd.notna(row.get("lower")) and pd.notna(row.get("upper")):
        text += f" Its 80 % range is US${row['lower']:,.0f} – {row['upper']:,.0f}/t."
    if row.get("confidence", NOT_ENOUGH) == NOT_ENOUGH or pd.isna(row.get("p_within_10")):
        return text + " There are not yet enough past forecasts to estimate how reliable it is."
    text += (
        f" Judging by this model's earlier forecasts (adjusted for how volatile the market is "
        f"now), there is about a {row['p_within_10'] * 100:.0f} % chance the actual price lands "
        f"within ±10 % of the forecast and {row['p_within_5'] * 100:.0f} % within ±5 %. The "
        f"chance the price ends above the {base} level is about {row['p_up'] * 100:.0f} %."
    )
    if not direction_reliable:
        text += DIRECTION_WARNING
    spread = row.get("model_spread_pct")
    if pd.notna(spread):
        agree = "agree closely" if spread <= 5 else "disagree" if spread > 15 else "differ somewhat"
        text += f" The models {agree} (spread {spread:.1f} % of the price)."
    return text + " " + ADVICE[str(row["confidence"])]
