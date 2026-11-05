"""Model Performance: test-period accuracy by model, horizon and period."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from cocoasense.dashboard import data

data.page_header("Model Performance")

table = data.test_metrics()
preds = data.test_predictions()
if table is None or preds is None:
    data.missing("Test-period metrics", "evaluate --final")
    st.stop()

periods: list[str] = ["full", "pre_surge", "surge", "post_peak"]
names: dict[str, str] = data.PERIOD_LABELS
c1, c2 = st.columns(2)
period = c1.selectbox(
    "Period (by target month)",
    [p for p in periods if p in set(table["period"])],
    format_func=lambda key: names.get(key, key),
)
horizon = c2.radio("Horizon (months)", sorted(table["horizon"].unique()), horizontal=True)

sel = table[(table["period"] == period) & (table["horizon"] == horizon)].sort_values("mae")
show = sel[
    [
        "model",
        "n",
        "mae",
        "rmse",
        "mape",
        "rel_mae",
        "directional_acc",
        "within_10pct",
        "coverage",
        "dm_pvalue",
    ]
].copy()
show["model"] = show["model"].map(data.label)
show["n"] = show["n"].astype(int)
show = data.dashes(show, ["directional_acc"], ".0%")
show = data.dashes(show, ["dm_pvalue"], ".3f")
st.dataframe(
    show.rename(
        columns={
            "model": "Model",
            "n": "Forecasts",
            "mae": "MAE US$/t",
            "rmse": "RMSE US$/t",
            "mape": "MAPE %",
            "rel_mae": "MAE vs naive",
            "directional_acc": "Direction correct",
            "within_10pct": "Within ±10 %",
            "coverage": "80 % interval coverage",
            "dm_pvalue": "DM p-value vs naive",
        }
    ).style.format(
        {
            "MAE US$/t": "{:,.0f}",
            "RMSE US$/t": "{:,.0f}",
            "MAPE %": "{:.2f}",
            "MAE vs naive": "{:.3f}",
            "Within ±10 %": "{:.0%}",
            "80 % interval coverage": "{:.0%}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "MAE vs naive < 1 means better than the last-price baseline. The Diebold–Mariano test "
    "has low power on samples this small; treat p-values as indicative."
)

bars = table[(table["horizon"] == horizon) & (table["period"] != "full")].copy()
bars["model"] = bars["model"].map(data.label)
bars["period"] = bars["period"].map(data.PERIOD_LABELS)
fig = px.bar(
    bars,
    x="period",
    y="mape",
    color="model",
    barmode="group",
    labels={"period": "Test sub-period", "mape": "MAPE, %", "model": "Model"},
    category_orders={"period": [data.PERIOD_LABELS[p] for p in periods[1:]]},
    color_discrete_map=data.MODEL_COLOURS,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Error over time")
p = preds[preds["horizon"] == horizon].copy()
p["model"] = p["model"].map(data.label)
p["ape"] = (p["y_pred"] - p["y_true"]).abs() / p["y_true"] * 100
window = st.slider("Rolling window (months)", 1, 12, 6)
p = p.sort_values("target_date")
p["rolling_mape"] = p.groupby("model")["ape"].transform(lambda s: s.rolling(window).mean())
fig2 = px.line(
    p,
    x="target_date",
    y="rolling_mape",
    color="model",
    labels={
        "target_date": "Target month",
        "rolling_mape": f"{window}-month MAPE, %",
        "model": "Model",
    },
    color_discrete_map=data.MODEL_COLOURS,
)
st.plotly_chart(fig2, use_container_width=True)

with st.expander("Validation leaderboard (2016–2019, used for model selection)"):
    for h in (1, 3):
        v = data.metrics(f"validation_leaderboard_h{h}.csv")
        if v is not None:
            st.markdown(f"**{h}-month horizon**")
            st.dataframe(v, hide_index=True, use_container_width=True)

with st.expander("Does the exchange rate help? (sub-question 2)"):
    fx = data.metrics("test_fx_experiment.csv")
    if fx is None:
        st.write("Not available.")
    else:
        st.dataframe(
            fx[["model", "horizon", "n", "mae", "mape", "rel_mae", "dm_pvalue"]],
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "rel_mae compares each model with its own price-only twin on identical "
            "months (those with exchange-rate data)."
        )
