"""Live Forecast: what the learning models expect next, how sure to be, and recent hits/misses."""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cocoasense.dashboard import data
from cocoasense.evaluation.insights import brier, confidence_check, explain

data.page_header("Live Forecast")

monthly = data.monthly()
rows = data.insights_rows()
if monthly is None or rows is None or not (rows["track"] == "learning").any():
    data.missing("Live forecasts", "learn` and then `uv run cocoasense insights")
    st.stop()

price = monthly["price_usd_t"].dropna()
last = price.index.max()
log = data.update_log()

# ------------------------------------------------------------- freshness + update
c1, c2, c3 = st.columns([1, 1, 1.3])
c1.metric("Latest market price", f"US${price.iloc[-1]:,.0f}/t", f"{last:%B %Y} average")
age = (pd.Timestamp.today().to_period("M") - last.to_period("M")).n
c2.metric(
    "Data freshness",
    f"{age} month{'s' if age != 1 else ''} old",
    "World Bank publishes early each month",
    delta_color="off",
)
with c3:
    st.write("**Get the latest market prices**")
    if st.button("Update prices and forecasts", type="primary", use_container_width=True):
        with st.spinner("Downloading prices, scoring earlier forecasts and refitting models…"):
            res = subprocess.run(
                [sys.executable, "-m", "cocoasense.cli", "update", "--root", str(data.root())],
                capture_output=True,
                text=True,
            )
        if res.returncode == 0:
            st.success(res.stdout.strip().splitlines()[-1])
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("Update failed. If the download is blocked, put the file in data/raw/manual/.")
            st.code(res.stderr[-2000:])
    if log:
        st.caption(f"Last update: {log[-1]['timestamp'][:16].replace('T', ' ')} UTC")
    autos = [r for r in log if r.get("mode") == "auto"]
    if autos:
        st.caption(
            f"Automatic check {autos[-1]['timestamp'][:16].replace('T', ' ')} UTC: "
            f"{autos[-1]['status']}. New monthly prices are downloaded and the models retrained "
            "automatically (daily and at log-on, when online)."
        )
    else:
        st.caption("Automatic updates: turn on with `uv run cocoasense schedule-updates`.")

data.explainer(
    "These forecasts come from the **learning track**: every model is refitted each month on all "
    "prices known so far, so it keeps learning from the market. The **confidence** of each "
    "forecast is estimated from how close the same model's earlier forecasts were, adjusted for "
    "how volatile the market is now. It describes past performance, not a guarantee, and this is "
    "not trading advice."
)

live = rows[(rows["track"] == "learning") & rows["y_true"].isna()].copy()
if live.empty:
    st.info("No forecasts beyond the latest data month yet. Run an update.")
    st.stop()

horizon = st.radio(
    "Forecast horizon",
    sorted(live["horizon"].unique()),
    format_func=lambda h: f"{h} month{'s' if h > 1 else ''} ahead",
    horizontal=True,
)
lh = live[live["horizon"] == horizon].copy()
lh["label"] = lh["model"].map(data.label)
target = lh["target_date"].iloc[0]
consensus = lh.loc[lh["model"] != "naive", "y_pred"].mean()

st.subheader(f"Forecast for {target:%B %Y}")
cols = st.columns(len(lh) + 1)
cols[0].metric(
    "Model consensus",
    f"US${consensus:,.0f}/t",
    f"{(consensus / price.iloc[-1] - 1) * 100:+.1f} % vs {last:%b %Y}",
)
for col, (_, r) in zip(cols[1:], lh.iterrows(), strict=True):
    col.metric(
        r["label"], f"US${r['y_pred']:,.0f}/t", f"{(r['y_pred'] / r['last_obs'] - 1) * 100:+.1f} %"
    )
    conf = r["confidence"]
    col.markdown(
        f"<span style='background:{data.CONFIDENCE_COLOURS[conf]};color:white;padding:2px 8px;"
        f"border-radius:8px;font-size:0.8em'>{conf} confidence</span>",
        unsafe_allow_html=True,
    )

track_scored = data.scored_rows()
if track_scored is not None:
    past = track_scored[
        (track_scored["track"] == "learning") & (track_scored["horizon"] == horizon)
    ]
    held = confidence_check(past)
    held = held[held["confidence"] != "Not enough history"]
    if not held.empty:
        st.caption(
            f"How these labels have held up for {horizon}-month forecasts since 2020: "
            + "; ".join(
                f"**{r.confidence}**: {r.actual_within_10:.0f} % landed within ±10 % (n={r.n})"
                for r in held.itertuples()
            )
            + ". The labels are a guide based on past results, not a guarantee."
        )

# ------------------------------------------------------------------- fan chart
hist = price.loc[last - pd.DateOffset(months=18) :]
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Actual price", line={"color": "black"}))
for _, r in lh.iterrows():
    colour = data.MODEL_COLOURS.get(r["label"], "#555")
    fig.add_trace(
        go.Scatter(
            x=[last, r["target_date"]],
            y=[r["last_obs"], r["y_pred"]],
            name=r["label"],
            mode="lines+markers",
            line={"color": colour, "dash": "dot"},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [0, r["upper"] - r["y_pred"]],
                "arrayminus": [0, r["y_pred"] - r["lower"]],
                "color": colour,
            },
        )
    )
fig.update_layout(
    yaxis_title="US$ per tonne (monthly average)",
    xaxis_title="Month",
    height=420,
    legend={"orientation": "h", "y": -0.2},
    title="Recent prices and the forecasts, with 80 % ranges",
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------- explanation per model
st.subheader("What each model says, in plain words")
labels = list(lh["label"])
pick = st.selectbox("Model", labels, index=labels.index("XGBoost") if "XGBoost" in labels else 0)
row = lh[lh["label"] == pick].iloc[0]
g1, g2 = st.columns([1, 2])
with g1:
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=(row["p_within_10"] * 100) if pd.notna(row["p_within_10"]) else 0,
            number={"suffix": " %"},
            title={"text": "Chance of landing within ±10 %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": data.CONFIDENCE_COLOURS[row["confidence"]]},
                "steps": [
                    {"range": [0, 50], "color": "#fde0dc"},
                    {"range": [50, 70], "color": "#fff1d6"},
                    {"range": [70, 100], "color": "#e3f4e1"},
                ],
            },
        )
    )
    gauge.update_layout(height=260, margin={"t": 60, "b": 10, "l": 20, "r": 20})
    st.plotly_chart(gauge, use_container_width=True)
with g2:
    scored_all = data.scored_rows()
    reliable = scored_all is None or brier(scored_all[scored_all["track"] == "learning"]) < 0.25
    st.markdown(explain(row, pick, reliable).replace("$", r"\$"))
    if pd.notna(row["p_up"]):
        up = row["p_up"] * 100
        bar = go.Figure(
            go.Bar(
                x=[up, 100 - up],
                y=["", ""],
                orientation="h",
                marker_color=["#2ca02c", "#d62728"],
                text=[f"Higher {up:.0f} %", f"Lower {100 - up:.0f} %"],
                textposition="inside",
            )
        )
        bar.update_layout(
            barmode="stack",
            height=90,
            showlegend=False,
            margin={"t": 5, "b": 5},
            xaxis={"visible": False, "range": [0, 100]},
            yaxis={"visible": False},
        )
        st.caption(f"Chance the {target:%B %Y} price is higher / lower than {last:%B %Y}")
        if not reliable:
            st.warning(
                "Measured on 2020–2026, these up/down chances were no better than a coin flip "
                "(see the Scorecard page). Treat the direction as unknown.",
                icon="⚠️",
            )
        st.plotly_chart(bar, use_container_width=True)

# ------------------------------------------------------- recent forecast record
st.subheader("Recent forecasts vs what actually happened")
data.explainer(
    "The forecast record keeps every learning-track forecast exactly as it was made. When the "
    "real price for that month is published, it is scored here: ✅ within ±5 %, 🟠 within ±10 %, "
    "❌ further off."
)
scored = data.scored_rows()
if scored is not None:
    rec = scored[(scored["track"] == "learning") & (scored["horizon"] == horizon)]
    rec = rec[rec["target_date"] >= last - pd.DateOffset(months=11)].copy()
    rec["Model"] = rec["model"].map(data.label)
    rec["Result"] = pd.cut(
        rec["abs_pct_error"],
        [-1, 5, 10, 1e9],
        labels=["✅ within 5 %", "🟠 within 10 %", "❌ missed"],
    ).astype(str)
    show = rec.sort_values(["target_date", "Model"], ascending=[False, True])[
        [
            "target_date",
            "Model",
            "y_pred",
            "y_true",
            "error",
            "abs_pct_error",
            "Result",
            "confidence",
        ]
    ].rename(
        columns={
            "target_date": "Month",
            "y_pred": "Forecast US$/t",
            "y_true": "Actual US$/t",
            "error": "Error US$/t",
            "abs_pct_error": "Error %",
            "confidence": "Confidence at the time",
        }
    )
    show["Month"] = show["Month"].dt.strftime("%b %Y")
    st.dataframe(show.style.format(precision=1), hide_index=True, use_container_width=True)
    st.download_button(
        "Download the full forecast record (CSV)",
        scored[scored["track"] == "learning"].to_csv(index=False).encode("utf-8"),
        "cocoasense_forecast_record.csv",
        "text/csv",
    )
