"""Predicted vs Actual: every forecast next to the real price, with the margin of error."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cocoasense.dashboard import data

data.page_header("Predicted vs actual prices")

scored = data.scored_rows()
if scored is None or scored.empty:
    data.missing("Scored forecasts", "insights")
    st.stop()

data.explainer(
    "Prices are in US$ per tonne; the table also shows cedis, converted at each month's average "
    "exchange rate. Each point is one forecast, placed at the month it was predicting. "
    "The **margin of error** "
    "is forecast minus actual: positive means the model guessed too high, negative too low. "
    "Green markers were within ±5 % of the actual price, amber within ±10 %, red further off."
)

c1, c2, c3 = st.columns(3)
tracks = [t for t in ("frozen", "learning") if t in set(scored["track"])]
track = c1.radio("Track", tracks, format_func=lambda t: data.TRACK_LABELS[t])
order = list(data.MODEL_LABELS)
models = sorted(scored.loc[scored["track"] == track, "model"].unique(), key=order.index)
model = c2.selectbox(
    "Model",
    models,
    index=models.index("xgboost") if "xgboost" in models else 0,
    format_func=data.label,
)
horizon = c3.radio(
    "Horizon", sorted(scored["horizon"].unique()), format_func=lambda h: f"{h} month(s) ahead"
)
sel = scored[
    (scored["track"] == track) & (scored["model"] == model) & (scored["horizon"] == horizon)
].sort_values("target_date")
sel = sel.assign(
    Result=pd.cut(
        sel["abs_pct_error"], [-1, 5, 10, 1e9], labels=["Within 5 %", "Within 10 %", "Missed"]
    ).astype(str),
    pct_error=(sel["y_pred"] - sel["y_true"]) / sel["y_true"] * 100,
)
RESULT_COLOURS = {"Within 5 %": "#16c784", "Within 10 %": "#fbbf24", "Missed": "#ea3943"}

k1, k2, k3, k4 = st.columns(4)
k1.metric("Forecasts", len(sel))
k2.metric(
    "Average error",
    f"{sel['abs_pct_error'].mean():.1f} %",
    f"US${sel['abs_error'].mean():,.0f}/t",
    delta_color="off",
)
k3.metric("Within ±10 %", f"{sel['within_10pct'].mean():.0%}")
k4.metric(
    "Tendency",
    "Too high" if sel["error"].mean() > 0 else "Too low",
    f"average bias US${sel['error'].mean():+,.0f}/t",
    delta_color="off",
)

# ----------------------------------------------------------- time series
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=pd.concat([sel["target_date"], sel["target_date"][::-1]]),
        y=pd.concat([sel["upper"], sel["lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(76,141,255,0.14)",
        line={"width": 0},
        name="80 % range",
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=sel["target_date"],
        y=sel["y_true"],
        name="Actual price",
        line={"color": "#e6e9ef", "width": 2},
    )
)
fig.add_trace(
    go.Scatter(
        x=sel["target_date"],
        y=sel["y_pred"],
        name="Forecast",
        mode="lines+markers",
        line={"color": "#4c8dff", "width": 1},
        marker={"color": sel["Result"].map(RESULT_COLOURS), "size": 7},
        customdata=sel[["abs_pct_error", "Result"]],
        hovertemplate="%{x|%b %Y}<br>Forecast US$%{y:,.0f}/t<br>Error %{customdata[0]:.1f} %"
        "<br>%{customdata[1]}<extra></extra>",
    )
)
fig.update_layout(
    height=440,
    yaxis_title="US$ per tonne",
    xaxis_title="Month being forecast",
    title=f"{data.label(model)}: forecast vs actual ({horizon}-month horizon)",
    legend={"orientation": "h", "y": -0.2},
)
st.plotly_chart(fig, use_container_width=True, theme=None)

# ----------------------------------------------------------- margin of error
st.subheader("Margin of error, month by month")
bars = px.bar(
    sel,
    x="target_date",
    y="pct_error",
    color="Result",
    color_discrete_map=RESULT_COLOURS,
    labels={"target_date": "Month being forecast", "pct_error": "Forecast − actual (% of actual)"},
)
for y in (-10, -5, 5, 10):
    bars.add_hline(y=y, line_dash="dot", line_color="#3a4352")
bars.update_layout(height=360)
st.plotly_chart(bars, use_container_width=True, theme=None)

c1, c2 = st.columns(2)
with c1:
    lo = min(sel["y_true"].min(), sel["y_pred"].min()) * 0.95
    hi = max(sel["y_true"].max(), sel["y_pred"].max()) * 1.05
    sc = px.scatter(
        sel,
        x="y_true",
        y="y_pred",
        color="Result",
        color_discrete_map=RESULT_COLOURS,
        labels={"y_true": "Actual US$/t", "y_pred": "Forecast US$/t"},
    )
    guides = ((1.0, "Perfect", "dot"), (1.1, "+10 %", None), (0.9, "−10 %", None))
    for factor, name, style in guides:
        sc.add_scatter(
            x=[lo, hi],
            y=[lo * factor, hi * factor],
            mode="lines",
            name=name,
            line={"color": "#e6e9ef" if style else "#3a4352", "dash": style},
        )
    sc.update_layout(height=400, title="Forecast against actual")
    st.plotly_chart(sc, use_container_width=True, theme=None)
with c2:
    hist = px.histogram(
        sel,
        x="pct_error",
        nbins=30,
        labels={"pct_error": "Forecast − actual (%)"},
        color_discrete_sequence=["#4c8dff"],
    )
    hist.add_vline(x=0, line_color="#e6e9ef")
    hist.update_layout(
        height=400, title="How big were the errors?", yaxis_title="Number of forecasts"
    )
    st.plotly_chart(hist, use_container_width=True, theme=None)

# ----------------------------------------------------------- table
st.subheader("Every forecast")
table = sel.sort_values("target_date", ascending=False)[
    [
        "decision_date",
        "target_date",
        "last_obs",
        "y_pred",
        "lower",
        "upper",
        "y_true",
        "error",
        "abs_pct_error",
        "Result",
        "confidence",
    ]
].rename(
    columns={
        "decision_date": "Made in",
        "target_date": "For month",
        "last_obs": "Price when made",
        "y_pred": "Forecast",
        "lower": "80 % low",
        "upper": "80 % high",
        "y_true": "Actual",
        "error": "Error US$/t",
        "abs_pct_error": "Error %",
        "confidence": "Confidence at the time",
    }
)
month_rate = sel.sort_values("target_date", ascending=False)["target_date"].map(data.fx_monthly())
table.insert(4, "Forecast GH₵", (table["Forecast"] * month_rate.to_numpy()).round(0))
table.insert(8, "Actual GH₵", (table["Actual"] * month_rate.to_numpy()).round(0))
for c in ("Made in", "For month"):
    table[c] = table[c].dt.strftime("%b %Y")
money_cols = {
    c: "{:,.0f}"
    for c in table.columns
    if c not in ("Made in", "For month", "Error %", "Result", "Confidence at the time")
}
st.dataframe(
    table.style.format({**money_cols, "Error %": "{:.1f}"}),
    hide_index=True,
    use_container_width=True,
)
st.download_button(
    "Download this table (CSV)",
    table.to_csv(index=False).encode("utf-8"),
    f"predicted_vs_actual_{track}_{model}_h{horizon}.csv",
    "text/csv",
)
