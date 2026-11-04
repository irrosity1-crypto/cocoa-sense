"""Strategy Simulator: recompute the cheap strategy arithmetic from stored forecasts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from cocoasense.dashboard import data
from cocoasense.simulation.engine import scenario_delivery, simulate
from cocoasense.simulation.futures_provider import SpotProxyProvider
from cocoasense.simulation.metrics import strategy_metrics
from cocoasense.simulation.strategies import FIXED_STRATEGIES, ForecastDriven

data.page_header("Strategy Simulator")

monthly = data.monthly()
preds = data.test_predictions()
frozen = data.frozen()
if monthly is None or preds is None or frozen is None:
    data.missing("Test-period forecasts and the frozen configuration", "evaluate --final")
    st.stop()

fz = frozen["simulation"]
sim_cfg = data.cfg().simulation
st.markdown(
    f"Frozen rule (chosen on 2016–2019 validation data): **{data.label(fz['driver_model'])}** "
    f"forecasts, **a = {fz['a']}**, horizon **{fz['horizon']} months**. Forward price = spot "
    "price at the decision month (*spot proxy*: no futures data)."
)

c1, c2, c3 = st.columns(3)
horizon = c1.radio(
    "Horizon (months)",
    sorted(preds["horizon"].unique()),
    index=sorted(preds["horizon"].unique()).index(fz["horizon"]),
    horizontal=True,
)
quantity = c2.number_input(
    "Quantity per decision (tonnes)", 1.0, 1_000_000.0, float(sim_cfg.quantity_t), step=100.0
)
rate = c3.slider("Financing rate (annual)", 0.0, 0.30, float(sim_cfg.financing_rate_annual), 0.01)
c4, c5, c6 = st.columns(3)
scenario = c4.selectbox(
    "Delivery price",
    ["actual", *sim_cfg.scenarios],
    format_func=lambda s: (
        "Actual (history)" if s == "actual" else f"HYPOTHETICAL: {s} (× {sim_cfg.scenarios[s]:.2f})"
    ),
)
models = sorted(preds["model"].unique())
driver = c5.selectbox(
    "Forecast model", models, index=models.index(fz["driver_model"]), format_func=data.label
)
a = c6.slider("Rule strength a", 0.0, 3.0, float(fz["a"]), 0.05)
if a != float(fz["a"]) or driver != fz["driver_model"]:
    st.info(
        "You changed the frozen rule. This is exploration only: settings chosen by looking "
        "at test-period results are not a fair test."
    )

base = preds[(preds["model"] == driver) & (preds["horizon"] == horizon)]
provider = SpotProxyProvider(monthly["price_usd_t"])
override = None
if scenario != "actual":
    override = scenario_delivery(base, provider, sim_cfg.scenarios[scenario])
results = simulate(
    base,
    [*FIXED_STRATEGIES, ForecastDriven(a)],
    provider,
    quantity,
    rate,
    delivery_override=override,
)

m = strategy_metrics(results)
st.dataframe(
    m[
        [
            "strategy",
            "mean_revenue_per_t",
            "std_revenue_per_t",
            "worst10_mean_per_t",
            "regret_vs_best_fixed",
            "win_rate_vs_5050",
            "mean_w",
        ]
    ]
    .rename(
        columns={
            "strategy": "Strategy",
            "mean_revenue_per_t": "Mean US$/t",
            "std_revenue_per_t": "Std US$/t",
            "worst10_mean_per_t": "Worst 10 % mean US$/t",
            "regret_vs_best_fixed": "Regret vs best fixed",
            "win_rate_vs_5050": "Win rate vs 50/50",
            "mean_w": "Mean forward share",
        }
    )
    .style.format(precision=2),
    hide_index=True,
    use_container_width=True,
)
total = results.groupby("strategy")["revenue"].sum() / 1e6
st.caption(
    f"Total simulated revenue over {results['decision_date'].nunique()} decisions "
    f"(US$ million): " + ", ".join(f"{k} {v:,.1f}" for k, v in total.items())
)

fig = px.line(
    results,
    x="decision_date",
    y="revenue_per_t",
    color="strategy",
    labels={
        "decision_date": "Decision month",
        "revenue_per_t": "Revenue, US$/t",
        "strategy": "Strategy",
    },
)
st.plotly_chart(fig, use_container_width=True)
fd = results[results["strategy"] == "forecast_driven"]
fig_w = px.area(
    fd,
    x="decision_date",
    y="w",
    labels={"decision_date": "Decision month", "w": "Share sold forward"},
)
fig_w.update_yaxes(range=[0, 1])
st.plotly_chart(fig_w, use_container_width=True)
st.caption(
    "Computed live from artifacts/forecasts/test_predictions.parquet with the formulas "
    "in docs/simulation_assumptions.md. Period: decision months "
    f"{pd.Timestamp(results['decision_date'].min()):%b %Y} – "
    f"{pd.Timestamp(results['decision_date'].max()):%b %Y}."
)
