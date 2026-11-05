"""Forecast Explorer: what the system knew on a chosen past date, then what happened."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cocoasense.dashboard import data

data.page_header("Forecast Explorer")

monthly = data.monthly()
preds = data.test_predictions()
if monthly is None or preds is None:
    data.missing("Test-period forecasts", "evaluate --final")
    st.stop()

dates = sorted(preds["decision_date"].unique())
labels = [pd.Timestamp(d).strftime("%b %Y") for d in dates]
choice = st.select_slider(
    "Decision month (forecast made at the end of this month)",
    options=labels,
    value=labels[len(labels) // 2],
)
t = pd.Timestamp(dates[labels.index(choice)])
models = st.multiselect(
    "Models",
    sorted(preds["model"].unique()),
    default=sorted(preds["model"].unique()),
    format_func=data.label,
)
lookback = st.slider("Months of history to show", 12, 120, 36, step=12)
reveal = st.toggle("Reveal what actually happened next", value=False)

price = monthly["price_usd_t"].dropna()
known = price.loc[:t].iloc[-lookback:]  # only data available at the decision date
at_t = preds[(preds["decision_date"] == t) & (preds["model"].isin(models))]

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=known.index, y=known.values, name="Known price history", line={"color": "#3b2a1a"})
)
colors = {m: data.MODEL_COLOURS.get(data.label(m)) for m in data.MODEL_LABELS}
for model, g in at_t.groupby("model"):
    g = g.sort_values("horizon")
    fig.add_trace(
        go.Scatter(
            x=[t, *g["target_date"]],
            y=[known.iloc[-1], *g["y_pred"]],
            mode="lines+markers",
            name=f"{data.label(str(model))} forecast",
            line={"dash": "dot", "color": colors.get(str(model))},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [0, *(g["upper"] - g["y_pred"])],
                "arrayminus": [0, *(g["y_pred"] - g["lower"])],
            },
        )
    )
if reveal:
    horizon_end = t + pd.DateOffset(months=int(preds["horizon"].max()) + 3)
    after = price.loc[t:horizon_end]
    fig.add_trace(
        go.Scatter(
            x=after.index,
            y=after.values,
            name="What actually happened",
            line={"color": "black", "width": 3},
        )
    )
fig.add_vline(x=t, line_dash="dash", line_color="grey")
fig.update_layout(height=450, xaxis_title="Month", yaxis_title="US$ per tonne", margin={"t": 20})
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"Forecasts made at the end of {t:%B %Y}")
cols = ["model", "horizon", "target_date", "y_pred", "lower", "upper"]
if reveal:
    cols += ["y_true"]
table = at_t[cols].sort_values(["horizon", "model"]).copy()
table["model"] = table["model"].map(data.label)
table["target_date"] = pd.to_datetime(table["target_date"]).dt.strftime("%b %Y")
if reveal:
    table["Error %"] = (table["y_pred"] - table["y_true"]) / table["y_true"] * 100
rename = {
    "model": "Model",
    "horizon": "Months ahead",
    "target_date": "For month",
    "y_pred": "Forecast US$/t",
    "lower": "Lower (80 %)",
    "upper": "Upper (80 %)",
    "y_true": "Actual US$/t",
}
shown = table.rename(columns=rename)
money = [
    c for c in ("Forecast US$/t", "Lower (80 %)", "Upper (80 %)", "Actual US$/t") if c in shown
]
fmt = {c: "{:,.0f}" for c in money}
if "Error %" in shown:
    fmt["Error %"] = "{:+.1f}"
st.dataframe(shown.style.format(fmt), hide_index=True, use_container_width=True)
if not reveal:
    st.caption("Actual prices after the decision month are hidden until you reveal them.")
st.caption(
    "Source: artifacts/forecasts/test_predictions.parquet and lstm_test_predictions.parquet "
    "(models frozen at Dec 2019)."
)
