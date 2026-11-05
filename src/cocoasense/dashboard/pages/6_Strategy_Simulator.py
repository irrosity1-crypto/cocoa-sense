"""Strategy Simulator: recompute the cheap strategy arithmetic from stored forecasts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from cocoasense.dashboard import data
from cocoasense.simulation.currency import to_cedis
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
    f"forecasts, **a = {fz['a']}**, horizon **{fz['horizon']} months**. Forward price = the "
    "decision month's World Bank/ICCO price, an average of the nearest three futures positions "
    "(not the exact delivery contract; no openly licensed contract data exists)."
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

currency = st.radio("Currency", ["US$", "Ghana cedis (GHS)"], horizontal=True, key="currency")
unit, value, total_col = "US$", "revenue_per_t", "revenue"
if currency != "US$":
    n_all = results["decision_date"].nunique()
    cedi = to_cedis(results, data.fx_monthly())
    if cedi.empty:
        st.info("No delivery month in this run has an observed exchange rate; showing US$.")
    else:
        results, unit, value, total_col = cedi, "GHS", "revenue_ghs_per_t", "revenue_ghs"
        st.caption(
            "Each delivery is paid in US$ and converted at the delivery month's average rate: "
            "IMF monthly averages to January 2026, then averages of daily rates from the live "
            f"currency-api service. {results['decision_date'].nunique()} of {n_all} deliveries "
            "have a rate (months without one are left out, not extrapolated). All strategies on "
            "a date share one rate, so which strategy wins on each date is the same as in US$. "
            "The written results (docs/results.md, section 9) use IMF rates only."
        )

m = data.strategy_display(strategy_metrics(results, value))
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
            "mean_revenue_per_t": f"Mean {unit}/t",
            "std_revenue_per_t": f"Std {unit}/t",
            "worst10_mean_per_t": f"Worst 10 % mean {unit}/t",
            "regret_vs_best_fixed": "Regret vs best fixed",
            "win_rate_vs_5050": "Win rate vs 50/50",
            "mean_w": "Mean forward share",
        }
    )
    .style.format(precision=2),
    hide_index=True,
    use_container_width=True,
)
results["strategy_name"] = results["strategy"].map(data.strategy_label)
total = results.groupby("strategy_name")[total_col].sum() / 1e6
st.caption(
    f"Total simulated revenue over {results['decision_date'].nunique()} decisions "
    f"({unit} million): " + ", ".join(f"{k} {v:,.1f}" for k, v in total.items())
)

fig = px.line(
    results,
    x="decision_date",
    y=value,
    color="strategy_name",
    labels={
        "decision_date": "Decision month",
        value: f"Revenue, {unit}/t",
        "strategy_name": "Strategy",
    },
    color_discrete_map=data.STRATEGY_COLOURS,
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
