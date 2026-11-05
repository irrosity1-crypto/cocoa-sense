"""CocoaSense AI dashboard: Overview page. Run with ``uv run cocoasense dashboard``."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cocoasense.dashboard import data, ui

data.page_header("CocoaSense AI — Overview")

st.markdown(
    "CocoaSense AI forecasts the monthly world cocoa price 1 and 3 months ahead and tests "
    "whether those forecasts would have improved simulated forward-sales decisions for a cocoa "
    "exporter such as Ghana. Use the pages on the left to explore forecasts at any past date, "
    "compare models, and run the strategy simulator."
)
data.explainer(
    "**How to read this app.** **Live Forecast** shows what the models expect for the coming "
    "months, how confident to be, and a button to fetch the latest market prices. "
    "**Scorecard** counts how often past forecasts succeeded or failed. **Predicted vs Actual** "
    "puts every forecast next to the real price with its margin of error. **Forecast Explorer** "
    "replays what the system knew at any past date. **Model Performance** and **Strategy "
    "Simulator** hold the detailed statistics and the selling simulation. Two kinds of model "
    "appear: *frozen* models trained only on data up to 2019 (the strict, fair test) and "
    "*learning* models refitted every month on all prices known at the time."
)

monthly = data.monthly()
if monthly is None:
    data.missing("Validated price data", "validate")
    st.stop()

ev = data.cfg().evaluation
price = monthly["price_usd_t"].dropna()
fx = data.fx_latest()
rate = fx[1] if fx else None
last, prev = float(price.iloc[-1]), float(price.iloc[-2])
year = price.loc[price.index >= price.index[-1] - pd.DateOffset(months=11)]
ui.hero(
    f"US${last:,.0f}",
    "per tonne",
    (last / prev - 1) * 100,
    f"{price.index[-1]:%B %Y} average" + (f" · ≈ GH₵{last * rate:,.0f}/t" if rate else ""),
)
year_ago = price.get(price.index[-1] - pd.DateOffset(months=12))
fx_prev = data.fx_monthly()
fx_prev = fx_prev.loc[fx_prev.index <= price.index[-1] - pd.DateOffset(months=1)]
ui.cards(
    [
        ui.Card(
            "12-month change",
            f"{(last / year_ago - 1) * 100:+.1f}%" if year_ago else "–",
            sub=f"vs US${year_ago:,.0f} a year earlier" if year_ago else "",
        ),
        ui.Card("12-month range", f"{year.min():,.0f}–{year.max():,.0f}", sub="US$ per tonne"),
        ui.Card(
            "US$ to cedi",
            f"GH₵{rate:,.2f}" if rate else "–",
            ((rate / fx_prev.iloc[-1] - 1) * 100) if rate and not fx_prev.empty else None,
            sub=f"{fx[0]:%d %b %Y} · live rate" if fx else "",
            delta_suffix=" vs last month",
        ),
        ui.Card(
            "History",
            f"{len(price)} months",
            sub=f"{price.index.min():%b %Y} – {price.index.max():%b %Y} · trained to Dec 2019",
        ),
    ]
)

log_scale = st.toggle("Log scale", value=False, key="ov_log")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=price.index,
        y=price.values,
        name="Cocoa price",
        line={"color": ui.THEME["accent"], "width": 1.8},
        fill="tozeroy" if not log_scale else None,
        fillgradient={
            "type": "vertical",
            "colorscale": [[0, "rgba(232,163,61,0)"], [1, "rgba(232,163,61,0.32)"]],
        },
        hovertemplate="%{x|%b %Y}<br>US$%{y:,.0f}/t<extra></extra>",
    )
)
for x, label in (
    (ev.ts("validation_start"), "validation starts"),
    (ev.ts("test_start"), "test starts (models frozen)"),
):
    fig.add_vline(x=x, line_width=1, line_dash="dot", line_color=ui.THEME["muted"])
    fig.add_annotation(
        x=x,
        y=1,
        yref="paper",
        text=label,
        showarrow=False,
        xanchor="left",
        font={"color": ui.THEME["muted"], "size": 11},
    )
fig.update_layout(
    height=430,
    yaxis_type="log" if log_scale else "linear",
    yaxis_title="US$ per tonne",
    hovermode="x unified",
    showlegend=False,
    xaxis={
        "range": [price.index.max() - pd.DateOffset(years=10), price.index.max()],
        "rangeselector": {
            "buttons": [
                {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                {"count": 3, "label": "3Y", "step": "year", "stepmode": "backward"},
                {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
                {"count": 10, "label": "10Y", "step": "year", "stepmode": "backward"},
                {"step": "all", "label": "ALL"},
            ],
            "bgcolor": ui.THEME["panel2"],
            "activecolor": ui.THEME["accent"],
            "font": {"color": ui.THEME["text"]},
        },
    },
)
st.plotly_chart(fig, use_container_width=True, theme=None)

fxm = data.fx_monthly()
if not fxm.empty:
    with st.expander("US$ to Ghana cedi rate", expanded=False):
        recent = fxm.loc[fxm.index >= fxm.index.max() - pd.DateOffset(months=60)]
        daily = data.fx_daily()
        fxfig = go.Figure()
        fxfig.add_trace(
            go.Scatter(
                x=recent.index, y=recent.values, name="Monthly average", line={"color": "#a8650f"}
            )
        )
        if not daily.empty:
            fxfig.add_trace(
                go.Scatter(
                    x=daily["date"],
                    y=daily["ghs_per_usd"],
                    name="Daily (live API)",
                    mode="markers",
                    marker={"size": 4, "color": "#e6e9ef"},
                )
            )
        fxfig.update_layout(height=300, margin={"t": 10, "b": 10}, yaxis_title="GH₵ per US$")
        st.plotly_chart(fxfig, use_container_width=True, theme=None)
        st.caption(
            "IMF monthly averages up to January 2026, then the average of daily rates from the "
            "free currency-api service (public domain), refreshed whenever the app updates "
            "online. Used only to show cedi values; no model uses the exchange rate."
        )

r1, r2 = st.columns([3, 1])
with r2:
    if st.button("Prepare PDF report", use_container_width=True, key="pdf_prepare"):
        from cocoasense.report_pdf import build_report

        with st.spinner("Building the report…"):
            try:
                st.session_state["pdf_report"] = build_report(data.cfg())
            except FileNotFoundError:
                st.warning("Run `uv run cocoasense insights` first to build the forecasts.")
    if "pdf_report" in st.session_state:
        st.download_button(
            "Download PDF report",
            st.session_state["pdf_report"],
            "cocoasense_report.pdf",
            "application/pdf",
            use_container_width=True,
            type="primary",
        )
with r1:
    st.caption(
        "A three-page PDF for your supervisor: current forecasts with confidence, the track "
        "record of past forecasts, and the key findings."
    )

st.subheader("Held-out test results (January 2020 onward)")
table = data.test_metrics()
if table is None:
    data.missing("Test-period metrics", "evaluate --final")
else:
    full = table[table["period"] == "full"]
    for h in sorted(full["horizon"].unique()):
        sub = full[full["horizon"] == h].sort_values("mae")
        show = sub[["model", "mape", "rel_mae", "within_10pct", "coverage"]].copy()
        show["model"] = show["model"].map(data.label)
        show.columns = ["Model", "MAPE %", "MAE vs naive", "Within ±10 %", "Interval coverage"]
        st.markdown(f"**{h}-month horizon** — {int(sub['n'].iloc[0])} forecasts")
        st.dataframe(
            show.style.format(
                {
                    "MAPE %": "{:.2f}",
                    "MAE vs naive": "{:.3f}",
                    "Within ±10 %": "{:.0%}",
                    "Interval coverage": "{:.0%}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    st.caption(
        "MAE vs naive below 1 means the model beat the last-price baseline. "
        "Source: artifacts/metrics/test_metrics.csv (LSTM: test_metrics_with_lstm.csv)"
    )

scored = data.scored_rows()
if scored is not None and not scored.empty:
    st.subheader("Success rate at a glance (3-month forecasts, 2020 onward)")
    h = 3 if 3 in set(scored["horizon"]) else int(scored["horizon"].max())
    sub = scored[scored["horizon"] == h]
    rate = (
        sub.groupby(["track", "model"])["within_10pct"].mean().mul(100).rename("rate").reset_index()
    )
    rate["Model"] = rate["model"].map(data.label)
    rate["Track"] = rate["track"].map({"frozen": "Frozen", "learning": "Learning"})
    fig = go.Figure()
    for track, colour in (("Frozen", "#9aa4b2"), ("Learning", "#e8a33d")):
        r = rate[rate["Track"] == track]
        fig.add_bar(
            x=r["Model"], y=r["rate"], name=track, marker_color=colour, text=r["rate"].round(0)
        )
    fig.update_layout(
        barmode="group",
        height=340,
        yaxis_title="% of forecasts within ±10 % of the actual price",
        yaxis_range=[0, 100],
        margin={"t": 20},
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.caption("Details, success vs fail counts and confidence checks: see the Scorecard page.")

sim = data.simulation_table("strategy_metrics.csv")
if sim is not None:
    st.subheader("Simulated forward-sales strategies (test period)")
    s = data.strategy_display(sim[sim["period"] == "full"])[
        [
            "strategy",
            "mean_revenue_per_t",
            "worst10_mean_per_t",
            "regret_vs_best_fixed",
            "win_rate_vs_5050",
        ]
    ]
    s.columns = [
        "Strategy",
        "Mean US$/t",
        "Worst 10 % mean US$/t",
        "Regret vs best fixed",
        "Win rate vs 50/50",
    ]
    st.dataframe(
        s.style.format(
            {
                "Mean US$/t": "{:,.0f}",
                "Worst 10 % mean US$/t": "{:,.0f}",
                "Regret vs best fixed": "{:,.0f}",
            },
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Source: artifacts/simulation/strategy_metrics.csv")
