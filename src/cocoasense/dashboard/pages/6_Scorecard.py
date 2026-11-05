"""Scorecard: successes vs failures, whether confidence labels held up, learning vs frozen."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cocoasense.dashboard import data
from cocoasense.evaluation.insights import (
    brier,
    confidence_check,
    probability_calibration,
    scorecard,
)

data.page_header("Scorecard: successes and failures")

scored = data.scored_rows()
if scored is None or scored.empty:
    data.missing("Scored forecasts", "insights")
    st.stop()
ins = data.cfg().insights
assert ins is not None
thresholds = ins.success_thresholds_pct
bounds = data.cfg().evaluation.period_bounds()


def in_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == "full":
        return df
    start, end = bounds[period]
    mask = df["target_date"] >= start
    if end is not None:
        mask &= df["target_date"] <= end
    return df[mask]


data.explainer(
    "Every forecast made for 2020 onwards is checked against the price that actually happened. "
    "A forecast counts as a **success** at ±5 % or ±10 % if the actual price was that close; "
    "the **direction** is right if the forecast and the market moved the same way; the **80 % "
    "range** succeeds if the actual price fell inside it. Choose a track: **Frozen** models "
    "were trained only on data up to 2019 (the strict test); **Learning** models were refitted "
    "every month on all prices known at the time."
)

c1, c2, c3 = st.columns(3)
tracks = [t for t in ("frozen", "learning") if t in set(scored["track"])]
track = c1.radio("Track", tracks, format_func=lambda t: data.TRACK_LABELS[t])
horizon = c2.radio(
    "Horizon", sorted(scored["horizon"].unique()), format_func=lambda h: f"{h} month(s) ahead"
)
period = c3.selectbox(
    "Period", ["full", *bounds], format_func=lambda p: data.PERIOD_LABELS.get(p, p)
)

sel = in_period(scored[(scored["track"] == track) & (scored["horizon"] == horizon)], period)
sel = sel.assign(Model=sel["model"].map(data.label))
card = scorecard(sel, ["Model"], thresholds)

# ------------------------------------------------------------------ headline
best = card.sort_values("within_10pct_pct", ascending=False).iloc[0]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Forecasts checked", f"{int(card['n'].sum()):,}")
k2.metric("Best model (within ±10 %)", best["Model"], f"{best['within_10pct_pct']:.0f} % success")
naive = card[card["Model"].str.startswith("Naive")]
if not naive.empty:
    beat = card[card["mape"] < naive["mape"].iloc[0]]["Model"].tolist()
    k3.metric(
        "Models beating 'no change'",
        f"{len(beat)} of {len(card) - 1}",
        ", ".join(beat) or "none",
        delta_color="off",
    )
k4.metric("Average error, best model", f"{best['mape']:.1f} %")

# ------------------------------------------------------------- success rates
st.subheader("How often were the forecasts right?")
long = card.melt(
    id_vars="Model",
    value_vars=[f"within_{t:g}pct_pct" for t in thresholds]
    + ["direction_correct_pct", "in_interval_pct"],
    var_name="Criterion",
    value_name="Success %",
)
names = {f"within_{t:g}pct_pct": f"Within ±{t:g} %" for t in thresholds}
names |= {"direction_correct_pct": "Direction right", "in_interval_pct": "Inside 80 % range"}
long["Criterion"] = long["Criterion"].map(names)
fig = px.bar(
    long,
    x="Criterion",
    y="Success %",
    color="Model",
    barmode="group",
    color_discrete_map=data.MODEL_COLOURS,
    text_auto=".0f",
)
fig.update_layout(yaxis_range=[0, 100], height=420)
fig.add_hline(y=80, line_dash="dot", annotation_text="an 80 % range should hold 80 % of the time")
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "The naive model predicts 'no change', so it has no direction; it is the benchmark every "
    "other model should beat."
)

t = st.select_slider("Success means the actual price was within ± (%)", thresholds, value=10)
col = f"within_{t:g}pct"
counts = sel.groupby("Model")[col].agg(success="sum", total="size").reset_index()
counts["fail"] = counts["total"] - counts["success"]
s1, s2 = st.columns([2, 1])
with s1:
    stacked = go.Figure()
    for kind, colour in (("success", "#2ca02c"), ("fail", "#d62728")):
        stacked.add_bar(
            y=counts["Model"],
            x=counts[kind],
            name=kind.title(),
            orientation="h",
            marker_color=colour,
            text=[
                f"{v:.0f} ({v / n:.0%})" for v, n in zip(counts[kind], counts["total"], strict=True)
            ],
        )
    stacked.update_layout(
        barmode="stack",
        height=320,
        xaxis_title="Number of forecasts",
        title=f"Success vs fail at ±{t:g} %",
    )
    st.plotly_chart(stacked, use_container_width=True)
with s2:
    one = st.selectbox("Model for the pie", list(counts["Model"]), index=len(counts) - 1)
    r = counts[counts["Model"] == one].iloc[0]
    pie = go.Figure(
        go.Pie(
            labels=["Success", "Fail"],
            values=[r["success"], r["fail"]],
            hole=0.55,
            marker_colors=["#2ca02c", "#d62728"],
        )
    )
    pie.update_layout(height=320, title=one)
    st.plotly_chart(pie, use_container_width=True)

# ------------------------------------------------------------- over time
st.subheader("Is it getting better or worse over time?")
parts = []
for _, g in sel.sort_values("target_date").groupby("Model"):
    parts.append(g.assign(rate=g[col].rolling(12, min_periods=6).mean() * 100))
roll = pd.concat(parts)
fig = px.line(
    roll,
    x="target_date",
    y="rate",
    color="Model",
    color_discrete_map=data.MODEL_COLOURS,
    labels={"target_date": "Month forecast", "rate": f"% within ±{t:g} % (last 12 months)"},
)
fig.update_layout(yaxis_range=[0, 100], height=380)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------- direction
st.subheader("Did the models call the direction?")
d1, d2 = st.columns([1, 1])
dmodel = d1.selectbox("Model", [m for m in counts["Model"] if not m.startswith("Naive")])
dm = sel[(sel["Model"] == dmodel) & sel["direction_correct"].notna()]
pred_up = np.where(dm["y_pred"] > dm["last_obs"], "Forecast up", "Forecast down")
act_up = np.where(dm["y_true"] > dm["last_obs"], "Actually up", "Actually down")
matrix = pd.crosstab(pd.Series(pred_up, name="Forecast"), pd.Series(act_up, name="Actual"))
heat = px.imshow(matrix, text_auto=True, color_continuous_scale="Greens", aspect="auto")
heat.update_layout(height=300, coloraxis_showscale=False)
d1.plotly_chart(heat, use_container_width=True)
d2.metric("Direction right", f"{dm['direction_correct'].mean():.0%}", f"{len(dm)} forecasts")
d2.caption("Correct calls are on the diagonal. A coin flip would get about 50 %.")

# --------------------------------------------------------- confidence check
st.subheader("Can you trust the confidence labels?")
data.explainer(
    "Each forecast was given a confidence label at the time it was made, using only earlier "
    "results. If the labels are useful, **High** forecasts should have succeeded more often than "
    "**Low** ones. This chart checks that against what actually happened."
)
tr = in_period(scored[(scored["track"] == track) & (scored["horizon"] == horizon)], period)
check = confidence_check(tr)
if not check.empty:
    fig = go.Figure()
    fig.add_bar(
        x=check["confidence"],
        y=check["actual_within_10"],
        name="Actually within ±10 %",
        marker_color=[data.CONFIDENCE_COLOURS[c] for c in check["confidence"]],
        text=[
            f"{v:.0f} % (n={n})" for v, n in zip(check["actual_within_10"], check["n"], strict=True)
        ],
    )
    fig.add_scatter(
        x=check["confidence"],
        y=check["predicted_p_within_10"],
        name="Estimated chance at the time",
        mode="markers",
        marker={"size": 14, "symbol": "diamond", "color": "black"},
    )
    fig.update_layout(yaxis_range=[0, 100], yaxis_title="%", height=360)
    st.plotly_chart(fig, use_container_width=True)

cal = probability_calibration(tr)
if not cal.empty:
    c1, c2 = st.columns([2, 1])
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line={"dash": "dot"})
    fig.add_scatter(
        x=cal["predicted"],
        y=cal["actual"],
        mode="lines+markers+text",
        text=[f"n={n}" for n in cal["n"]],
        textposition="top center",
        name="CocoaSense",
    )
    fig.update_layout(
        xaxis_title="Estimated chance the price would rise",
        yaxis_title="How often it actually rose",
        height=360,
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )
    c1.plotly_chart(fig, use_container_width=True)
    c2.metric("Brier score (lower is better)", f"{brier(tr):.3f}", "0.250 = coin flip", "off")
    c2.caption(
        "If the points follow the dotted line, a '70 % chance of a rise' really meant the price "
        "rose about 7 times in 10."
    )

# --------------------------------------------------------- learning vs frozen
if {"frozen", "learning"} <= set(scored["track"]):
    st.subheader("Does learning from new prices help?")
    data.explainer(
        "Same months, same models: the frozen track never updates after 2019, the learning track "
        "refits every month. Lower average error means better forecasts."
    )
    learned = set(scored.loc[scored["track"] == "learning", "model"])
    both = scored[(scored["horizon"] == horizon) & scored["model"].isin(learned)]
    both = in_period(both, period)
    both = both.assign(Model=both["model"].map(data.label))
    comp = scorecard(both, ["Model", "track"], thresholds)
    comp["Track"] = comp["track"].map({"frozen": "Frozen", "learning": "Learning"})
    fig = px.bar(
        comp,
        x="Model",
        y="mape",
        color="Track",
        barmode="group",
        text_auto=".1f",
        labels={"mape": "Average error (MAPE, %)"},
        color_discrete_map={"Frozen": "#7f7f7f", "Learning": "#1f77b4"},
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Full scorecard table"):
    st.dataframe(card.style.format(precision=1), hide_index=True, use_container_width=True)
    st.download_button(
        "Download (CSV)", card.to_csv(index=False).encode("utf-8"), "scorecard.csv", "text/csv"
    )
