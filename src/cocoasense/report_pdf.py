"""One-click PDF report: current forecasts, track record and key findings (three A4 pages).

Built with matplotlib only, from the pipeline's stored outputs; nothing is refitted. Used by
``cocoasense report-pdf`` and by the dashboard's download button.
"""

from __future__ import annotations

import io
import textwrap
from datetime import UTC, datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from cocoasense.config import ProjectConfig
from cocoasense.evaluation.insights import outcomes
from cocoasense.ingestion.fx_live import latest, read_daily

A4 = (8.27, 11.69)
INK, MUTED, ACCENT, UP, DOWN = "#1d1d1f", "#6b6b70", "#b8741a", "#13895f", "#c0392b"
LABELS = {
    "naive": "Naive",
    "arima": "ARIMA",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}
DISCLAIMER = (
    "Simulated results on public prices. These are not Ghana's or COCOBOD's actual gains or "
    "losses, and the forecasts are not trading advice. Frozen models were trained only on data up "
    "to December 2019; learning models are refitted each month on prices known at the time."
)
FINDINGS = [
    "No model reliably beat the naive 'price stays the same' forecast. ARIMA had the lowest error "
    "(6.87% at 1 month, 14.88% at 3 months), only 3.6% and 1.1% better than naive and not "
    "statistically significant.",
    "One month ahead, about 3 forecasts in 4 landed within ±10% of the actual price; three months "
    "ahead, about half. In calm 2020–2022 about 9 in 10 did.",
    "The 2023–25 surge broke the models: they had learned from 1991–2019 that high prices fall "
    "back, monthly volatility doubled (about 6.5% to 13.5%), and the causes (weather, disease, "
    "smuggling) are not in price data.",
    "The forecast-driven selling rule earned 0.8% less than a fixed 50/50 split. A volatility "
    "guard narrowed the gap to 0.4% but did not close it.",
    "Volatility-aware prediction ranges brought XGBoost's 80% range coverage from 72% to 84% "
    "(1 month): uncertainty can be measured honestly even when the price cannot be predicted.",
    "The cedi exchange rate did not improve forecasts. Refitting the models every month did not "
    "improve accuracy either.",
    "Conclusion: on public price data, machine learning does not improve forward-sales revenue. "
    "Its value lies in measuring uncertainty, which tells a seller when a fixed selling split is "
    "the safer choice.",
]


def _page(title: str, subtitle: str) -> plt.Figure:
    fig = plt.figure(figsize=A4)
    fig.text(0.07, 0.955, title, fontsize=18, weight="bold", color=INK)
    fig.text(0.07, 0.935, subtitle, fontsize=9.5, color=MUTED)
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.925, 0.925], color=ACCENT, lw=1.5))
    return fig


def _wrap(fig: plt.Figure, x: float, y: float, text: str, width: int = 95, **kw: Any) -> float:
    """Draw wrapped text; return the y position below it."""
    lines = textwrap.wrap(text, width)
    size = kw.get("fontsize", 9.5)
    colour = kw.get("color", INK)
    for i, line in enumerate(lines):
        fig.text(x, y - i * 0.0165, line, fontsize=size, color=colour)
    return y - len(lines) * 0.0165 - 0.01


def _table(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    cells: list[list[str]],
    header: list[str],
) -> None:
    ax = fig.add_axes(rect)
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=header, loc="upper left", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.45)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor("#f3ead9")
            cell.set_text_props(weight="bold")


def _forecast_cells(live: pd.DataFrame, last: float, rate: float | None) -> list[list[str]]:
    cells = []
    for (target, h), g in live.groupby(["target_date", "horizon"]):
        cons = g.loc[g["model"] != "naive", "y_pred"].mean()
        cells.append(
            [
                f"{pd.Timestamp(target):%b %Y}",
                f"{h} mo",
                "Consensus",
                f"{cons:,.0f}",
                f"{(cons / last - 1) * 100:+.1f}%",
                f"{cons * rate:,.0f}" if rate else "–",
                "",
            ]
        )
        for _, r in g.sort_values("model").iterrows():
            cells.append(
                [
                    "",
                    "",
                    LABELS.get(str(r["model"]), str(r["model"])),
                    f"{r['y_pred']:,.0f}",
                    f"{(r['y_pred'] / r['last_obs'] - 1) * 100:+.1f}%",
                    f"{r['y_pred'] * rate:,.0f}" if rate else "–",
                    str(r["confidence"]),
                ]
            )
    return cells


def _record_cells(scored: pd.DataFrame) -> list[list[str]]:
    cells = []
    for track in ("frozen", "learning"):
        for h in (1, 3):
            sub = scored[(scored["track"] == track) & (scored["horizon"] == h)]
            for model, g in sub.groupby("model"):
                hit = g["within_10pct"].mean()
                cells.append(
                    [
                        "Frozen" if track == "frozen" else "Learning",
                        f"{h} mo",
                        LABELS.get(str(model), str(model)),
                        f"{len(g)}",
                        f"{hit * 100:.0f}%",
                        f"{(1 - hit) * 100:.0f}%",
                        f"{g['within_5pct'].mean() * 100:.0f}%",
                    ]
                )
    return cells


def build_report(cfg: ProjectConfig) -> bytes:
    """Render the report and return the PDF bytes."""
    p = cfg.paths
    monthly = pd.read_parquet(p.interim / "monthly.parquet")
    monthly.index = pd.DatetimeIndex(monthly.index)
    price = monthly["price_usd_t"].dropna()
    rows = pd.read_parquet(p.forecasts / "insights_predictions.parquet")
    fx_live = latest(read_daily(p.artifacts / "fx" / "usd_ghs_daily.csv"))
    if fx_live is None and monthly["ghs_per_usd"].notna().any():
        s = monthly["ghs_per_usd"].dropna()
        fx_live = (pd.Timestamp(s.index[-1]), float(s.iloc[-1]), "IMF monthly average")
    rate = fx_live[1] if fx_live else None
    last, prev = float(price.iloc[-1]), float(price.iloc[-2])
    now = datetime.now(UTC)
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ------------------------------------------------------------ page 1
        fig = _page(
            "CocoaSense AI — forecast report",
            f"Generated {now:%d %B %Y} · data through {price.index[-1]:%B %Y}",
        )
        move = (last / prev - 1) * 100
        fig.text(0.07, 0.885, f"US${last:,.0f}/t", fontsize=26, weight="bold", color=INK)
        fig.text(
            0.40,
            0.89,
            f"{'up' if move >= 0 else 'down'} {abs(move):.1f}% on the month",
            fontsize=11,
            color=UP if move >= 0 else DOWN,
        )
        sub = f"{price.index[-1]:%B %Y} monthly average"
        if fx_live:
            sub += f" · about GH₵{last * fx_live[1]:,.0f}/t at GH₵{fx_live[1]:,.2f} per US$"
        fig.text(0.07, 0.862, sub, fontsize=9.5, color=MUTED)
        fig.text(0.07, 0.83, "Forecasts for the coming months", fontsize=12.5, weight="bold")
        live = rows[(rows["track"] == "learning") & rows["y_true"].isna()]
        cells = _forecast_cells(live, last, rate)
        if cells:
            height = min(0.62, 0.0235 * (len(cells) + 1))
            header = ["Month", "Ahead", "Model", "US$/t", "Change", "GH₵/t", "Confidence"]
            _table(fig, (0.07, 0.81 - height, 0.86, height), cells, header)
            y = 0.78 - height
        else:
            y = 0.8
        _wrap(
            fig,
            0.07,
            y,
            "Confidence comes from how close each model's earlier forecasts were, adjusted for how "
            "volatile the market is now. Low confidence means the actual price could easily land "
            "10% or more away from the forecast. Cedi values use today's exchange rate, because "
            "future exchange rates are not forecast.",
            fontsize=9,
            color=MUTED,
        )
        _wrap(fig, 0.07, 0.06, DISCLAIMER, width=110, fontsize=8, color=MUTED)
        pdf.savefig(fig)
        plt.close(fig)

        # ------------------------------------------------------------ page 2
        fig = _page("Track record", "How often past forecasts were right (2020 onwards)")
        scored = outcomes(rows, [5.0, 10.0])
        ax = fig.add_axes((0.1, 0.62, 0.84, 0.26))
        recent = price.loc[price.index >= price.index[-1] - pd.DateOffset(months=36)]
        ax.plot(recent.index, recent.values, color=INK, lw=1.8, label="Actual price")
        learn1 = scored[
            (scored["track"] == "learning")
            & (scored["horizon"] == 1)
            & (scored["model"] != "naive")
        ]
        cons = learn1.groupby("target_date")["y_pred"].mean()
        cons = cons.loc[cons.index >= recent.index[0]]
        ax.plot(
            cons.index, cons.values, color=ACCENT, lw=1.2, ls="--", label="Forecast a month earlier"
        )
        ax.set_ylabel("US$ per tonne")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
        ax.set_title(
            "Last three years: actual price and the consensus forecast", fontsize=10, loc="left"
        )
        cells = _record_cells(scored)
        fig.text(0.07, 0.57, "Success and failure rates", fontsize=12.5, weight="bold")
        if cells:
            height = min(0.48, 0.0215 * (len(cells) + 1))
            header = [
                "Models",
                "Ahead",
                "Model",
                "Forecasts",
                "Within ±10%",
                "Missed",
                "Within ±5%",
            ]
            _table(fig, (0.07, 0.555 - height, 0.86, height), cells, header)
        _wrap(fig, 0.07, 0.06, DISCLAIMER, width=110, fontsize=8, color=MUTED)
        pdf.savefig(fig)
        plt.close(fig)

        # ------------------------------------------------------------ page 3
        fig = _page("Key findings", "Held-out evaluation, January 2020 to the latest month")
        y = 0.89
        for finding in FINDINGS:
            fig.text(0.07, y, "•", fontsize=11, color=ACCENT)
            y = _wrap(fig, 0.095, y, finding, width=92) - 0.008
        _wrap(fig, 0.07, 0.06, DISCLAIMER, width=110, fontsize=8, color=MUTED)
        pdf.savefig(fig)
        plt.close(fig)
    return buf.getvalue()
