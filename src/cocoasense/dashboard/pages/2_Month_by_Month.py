"""Month by month: every month's actual price beside each model's forecast, with filters and the
overall success and failure rate of whatever is selected."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cocoasense.dashboard import data, ui

data.page_header("Month by month")

scored = data.scored_rows()
ins = data.cfg().insights
if scored is None or scored.empty or ins is None:
    data.missing("Scored forecasts", "insights")
    st.stop()

data.explainer(
    "Every month from January 2020, with the price that actually happened and what each model had "
    "forecast for it. A forecast is a **success** if it landed within the chosen distance of the "
    "actual price, and a **failure** otherwise. The success and failure rates at the top are "
    "recalculated for whatever you select below."
)

# ------------------------------------------------------------------ filters
f1, f2, f3 = st.columns([1.3, 1, 1.4])
tracks = [t for t in ("frozen", "learning") if t in set(scored["track"])]
track = f1.radio("Models", tracks, format_func=lambda t: data.TRACK_LABELS[t], key="mbm_track")
horizon = f2.radio(
    "Forecast made",
    sorted(scored["horizon"].unique()),
    format_func=lambda h: f"{h} month{'s' if h > 1 else ''} ahead",
    key="mbm_h",
)
thresholds = [float(t) for t in ins.success_thresholds_pct]
thr = f3.radio(
    "Success means within",
    thresholds,
    index=len(thresholds) - 1,
    format_func=lambda t: f"±{t:g} % of the actual price",
    horizontal=True,
    key="mbm_thr",
)
hit_col = f"within_{thr:g}pct"

base = scored[(scored["track"] == track) & (scored["horizon"] == horizon)]
months = sorted(pd.to_datetime(base["target_date"].unique()))
labels = [f"{m:%b %Y}" for m in months]
g1, g2, g3 = st.columns([2, 2, 1])
start_l, end_l = g1.select_slider(
    "Months", options=labels, value=(labels[0], labels[-1]), key="mbm_range"
)
order = list(data.MODEL_LABELS)
model_opts = sorted(base["model"].unique(), key=order.index)
chosen = g2.multiselect(
    "Models to include", model_opts, default=model_opts, format_func=data.label, key="mbm_models"
)
show = g3.selectbox("Show", ["All", "Successes only", "Failures only"], key="mbm_show")
if not chosen:
    st.info("Pick at least one model.")
    st.stop()

start, end = months[labels.index(start_l)], months[labels.index(end_l)]
sel = base[
    (base["target_date"] >= start) & (base["target_date"] <= end) & base["model"].isin(chosen)
].copy()
sel["hit"] = sel[hit_col].astype(bool)
sel["Model"] = sel["model"].map(data.label)

# --------------------------------------------------------------- headline rates
n = len(sel)
hits = int(sel["hit"].sum())
rate = hits / n * 100 if n else 0.0
by_model = sel.groupby("Model")["hit"].mean().mul(100).sort_values(ascending=False)
ui.cards(
    [
        ui.Card(
            "Success rate",
            f"{rate:.1f}%",
            sub=f"{hits:,} of {n:,} forecasts within ±{thr:g} %",
            pill=("success", ui.THEME["up"]),
        ),
        ui.Card(
            "Failure rate",
            f"{100 - rate:.1f}%",
            sub=f"{n - hits:,} of {n:,} forecasts missed by more",
            pill=("failure", ui.THEME["down"]),
        ),
        ui.Card(
            "Best in selection",
            str(by_model.index[0]) if len(by_model) else "–",
            sub=f"{by_model.iloc[0]:.0f}% success" if len(by_model) else "",
        ),
        ui.Card(
            "Months covered",
            f"{sel['target_date'].nunique()}",
            sub=f"{start:%b %Y} – {end:%b %Y}",
        ),
    ]
)

c1, c2 = st.columns([1, 1.6])
with c1:
    donut = go.Figure(
        go.Pie(
            labels=["Success", "Failure"],
            values=[hits, n - hits],
            hole=0.62,
            marker_colors=[ui.THEME["up"], ui.THEME["down"]],
            sort=False,
            textinfo="percent",
        )
    )
    donut.update_layout(
        height=300,
        title="All selected forecasts",
        annotations=[
            {"text": f"{rate:.0f}%", "showarrow": False, "font": {"size": 26, "color": ui.ACTUAL}}
        ],
    )
    st.plotly_chart(donut, use_container_width=True, theme=None)
with c2:
    bars = go.Figure(
        go.Bar(
            x=by_model.values,
            y=by_model.index,
            orientation="h",
            marker_color=[data.MODEL_COLOURS.get(m, ui.THEME["info"]) for m in by_model.index],
            text=[f"{v:.0f}%" for v in by_model.values],
            textposition="outside",
        )
    )
    bars.update_layout(
        height=300,
        title=f"Success rate by model (within ±{thr:g} %)",
        xaxis={"range": [0, 105], "title": "% of forecasts"},
        yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(bars, use_container_width=True, theme=None)

# ------------------------------------------------------------ month-by-month strip
per_month = sel.groupby("target_date")["hit"].mean().mul(100)
strip = go.Figure()
strip.add_bar(
    x=per_month.index,
    y=per_month.values,
    name="Succeeded",
    marker_color=ui.THEME["up"],
    hovertemplate="%{x|%b %Y}: %{y:.0f}% of models succeeded<extra></extra>",
)
strip.add_bar(
    x=per_month.index,
    y=100 - per_month.values,
    name="Failed",
    marker_color=ui.THEME["down"],
    hovertemplate="%{x|%b %Y}: %{y:.0f}% of models failed<extra></extra>",
)
strip.update_layout(
    barmode="stack",
    height=260,
    title="Each month: share of the selected models that succeeded (green) or failed (red)",
    yaxis={"range": [0, 100], "title": "%"},
    bargap=0.12,
    legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
)
st.plotly_chart(strip, use_container_width=True, theme=None)

# ------------------------------------------------------------------- the table
st.subheader("Every month")
view = st.radio(
    "Layout", ["One row per month", "One row per forecast"], horizontal=True, key="mbm_view"
)
fx = data.fx_monthly()
sel["error_pct"] = (sel["y_pred"] - sel["y_true"]) / sel["y_true"] * 100


def colour_error(v: object) -> str:
    """Green inside the success range, red outside."""
    if not isinstance(v, (int, float)) or pd.isna(v):
        return ""
    good = abs(float(v)) <= thr
    return f"color: {ui.THEME['up'] if good else ui.THEME['down']}; font-weight: 600"


if view == "One row per month":
    actual = sel.groupby("target_date")["y_true"].first()
    table = pd.DataFrame({"Month": actual.index, "Actual US$/t": actual.to_numpy()})
    table["Actual GH₵/t"] = (actual * actual.index.map(fx)).to_numpy()
    for m in [x for x in model_opts if x in chosen]:
        rows = sel[sel["model"] == m].set_index("target_date")
        table[f"{data.label(m)} forecast"] = table["Month"].map(rows["y_pred"])
        table[f"{data.label(m)} error %"] = table["Month"].map(rows["error_pct"])
    counts = sel.groupby("target_date")["hit"].agg(["sum", "size"]).reindex(table["Month"])
    table["Models within range"] = [f"{int(s)} of {int(z)}" for s, z in counts.to_numpy()]
    share = per_month.reindex(table["Month"]).to_numpy()
    if show == "Successes only":
        table = table[share > 50]
    elif show == "Failures only":
        table = table[share <= 50]
    caption = (
        "A month counts as a success when more than half of the selected models were within "
        f"±{thr:g} % of the actual price. Error % = (forecast − actual) ÷ actual; green is within "
        "range, red is outside."
    )
else:
    table = sel[["target_date", "Model", "y_pred", "y_true", "error_pct", "hit", "confidence"]]
    table = table.rename(
        columns={
            "target_date": "Month",
            "y_pred": "Forecast US$/t",
            "y_true": "Actual US$/t",
            "error_pct": "Error %",
            "confidence": "Confidence at the time",
        }
    )
    table["Result"] = table.pop("hit").map({True: "✅ Success", False: "❌ Failure"})
    if show == "Successes only":
        table = table[table["Result"].str.contains("Success")]
    elif show == "Failures only":
        table = table[table["Result"].str.contains("Failure")]
    caption = f"Success = forecast within ±{thr:g} % of the actual price for that month."

table = table.sort_values("Month", ascending=False)
table["Month"] = pd.to_datetime(table["Month"]).dt.strftime("%b %Y")
err_cols = [c for c in table.columns if c.endswith("error %") or c == "Error %"]
money = [c for c in table.columns if "US$" in c or "GH₵" in c or c.endswith("forecast")]
styler = table.style.format(
    {**{c: "{:,.0f}" for c in money}, **{c: "{:+.1f}" for c in err_cols}}
).map(colour_error, subset=err_cols)
st.dataframe(styler, hide_index=True, use_container_width=True, height=520)
st.caption(f"{len(table):,} rows shown. {caption}")
st.download_button(
    "Download this table (CSV)",
    table.to_csv(index=False).encode("utf-8"),
    f"cocoasense_month_by_month_{track}_h{horizon}.csv",
    "text/csv",
)
