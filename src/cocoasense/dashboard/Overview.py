"""CocoaSense AI dashboard: Overview page. Run with ``uv run cocoasense dashboard``."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from cocoasense.dashboard import data

data.page_header("CocoaSense AI — Overview")

st.markdown(
    "CocoaSense AI forecasts the monthly world cocoa price 1 and 3 months ahead and tests "
    "whether those forecasts would have improved simulated forward-sales decisions for a cocoa "
    "exporter such as Ghana. Use the pages on the left to explore forecasts at any past date, "
    "compare models, and run the strategy simulator."
)

monthly = data.monthly()
if monthly is None:
    data.missing("Validated price data", "validate")
    st.stop()

ev = data.cfg().evaluation
price = monthly["price_usd_t"].dropna()
c1, c2, c3 = st.columns(3)
c1.metric(
    "Months of price data",
    f"{len(price)}",
    f"{price.index.min():%b %Y} – {price.index.max():%b %Y}",
    delta_color="off",
    delta_arrow="off",
)
c2.metric(
    "Training cut-off", "Dec 2019", "no model sees later data", delta_color="off", delta_arrow="off"
)
c3.metric(
    "Latest monthly price",
    f"US$ {price.iloc[-1]:,.0f}/t",
    f"{price.index[-1]:%b %Y}",
    delta_color="off",
    delta_arrow="off",
)

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=price.index, y=price.values, name="Cocoa price", line={"color": "#3b2a1a"})
)
fig.add_vrect(
    x0=ev.ts("validation_start"),
    x1=ev.ts("validation_end"),
    fillcolor="#f2c14e",
    opacity=0.25,
    line_width=0,
    annotation_text="validation",
    annotation_position="top",
)
fig.add_vrect(
    x0=ev.ts("test_start"),
    x1=price.index.max(),
    fillcolor="#5fa8d3",
    opacity=0.25,
    line_width=0,
    annotation_text="test (held out)",
    annotation_position="top",
)
fig.update_layout(
    height=380,
    margin={"t": 30, "b": 10},
    yaxis_type="log",
    xaxis_title="Month",
    yaxis_title="US$ per tonne (log scale)",
)
st.plotly_chart(fig, use_container_width=True)

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
