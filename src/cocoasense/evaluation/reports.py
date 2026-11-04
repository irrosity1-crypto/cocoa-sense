"""Write tidy tables as CSV and Markdown."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LEADERBOARD_COLS = [
    "model",
    "n",
    "mae",
    "rmse",
    "mape",
    "rel_mae",
    "directional_acc",
    "within_5pct",
    "within_10pct",
    "coverage",
    "interval_width_pct",
    "dm_stat",
    "dm_pvalue",
]


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Render a DataFrame as a GitHub Markdown table without extra dependencies."""

    def fmt(v: object) -> str:
        if isinstance(v, float | np.floating):
            if np.isnan(v):
                return "–"
            return format(float(v), floatfmt)
        if isinstance(v, pd.Timestamp):
            return v.strftime("%Y-%m")
        return str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body]) + "\n"


def leaderboard(table: pd.DataFrame, stage: str, period: str, horizon: int) -> pd.DataFrame:
    """Metrics for one stage/period/horizon sorted by MAE."""
    sel = table[
        (table["stage"] == stage) & (table["period"] == period) & (table["horizon"] == horizon)
    ]
    cols = [c for c in LEADERBOARD_COLS if c in sel]
    return sel.sort_values("mae")[cols].reset_index(drop=True)


def write_table(df: pd.DataFrame, path: Path, title: str | None = None) -> None:
    """Write ``path`` (.csv) and a sibling .md file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    md = (f"## {title}\n\n" if title else "") + to_markdown(df)
    path.with_suffix(".md").write_text(md, encoding="utf-8")
