"""Model Performance: test-period accuracy by model, horizon and period."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from cocoasense.dashboard import data

data.page_header("Model Performance")

table = data.metrics("test_metrics.csv")
preds = data.test_predictions()
if table is None or preds is None:
    data.missing("Test-period metrics", "evaluate --final")
    st.stop()

periods: list[str] = ["full", "pre_surge", "surge", "post_peak"]
names: dict[str, str] = {
    "full": "Full test period",
    "pre_surge": "Pre-surge (2020–2022)",
    "surge": "Surge (2023 – Jan 2025)",
    "post_peak": "After the peak (Feb 2025 on)",
}
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
    ).style.format(precision=3),
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "MAE vs naive < 1 means better than the last-price baseline. The Diebold–Mariano test "
    "has low power on samples this small; treat p-values as indicative."
)

fig = px.bar(
    table[(table["horizon"] == horizon) & (table["period"] != "full")],
    x="period",
    y="mape",
    color="model",
    barmode="group",
    labels={"period": "Test sub-period", "mape": "MAPE, %", "model": "Model"},
    category_orders={"period": periods[1:]},
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Error over time")
p = preds[preds["horizon"] == horizon].copy()
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
