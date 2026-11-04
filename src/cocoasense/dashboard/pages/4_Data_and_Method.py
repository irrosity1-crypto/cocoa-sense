"""Data and Method: sources, data quality, protocol and the frozen configuration."""

from __future__ import annotations

import streamlit as st
import yaml

from cocoasense.dashboard import data

data.page_header("Data and Method")

st.subheader("Sources")
man = data.manifest()
if man is None:
    data.missing("The raw-data manifest", "ingest")
else:
    st.dataframe(
        man[["source", "path", "rows", "date_min", "date_max", "origin", "retrieved_at", "sha256"]],
        hide_index=True,
        use_container_width=True,
    )
    for key, src in data.cfg().data.sources.items():
        st.markdown(
            f"- **{key}** — {src.description}. [{src.landing_page}]({src.landing_page}) "
            f"· {src.licence}"
        )

st.subheader("Evaluation protocol")
ev = data.cfg().evaluation
st.markdown(
    f"""
- **Development:** models trained on targets {ev.train_start} – {ev.train_end}; settings chosen
  on validation targets {ev.validation_start} – {ev.validation_end}.
- **Freeze:** the chosen settings are written to `configs/frozen.yaml` and committed.
- **Final:** models refitted once on targets up to **{ev.final_train_end}**, then frozen. Every
  month from **{ev.test_start}** is forecast with real history as input but **no retraining**.
- Splits are by time only. Five automated leakage tests guard features, forecasts, training
  windows, tuning and strategy decisions.
"""
)

st.subheader("Data quality")
dq = data.text(data.cfg().paths.artifacts / "data_quality.md")
if dq is None:
    data.missing("The data-quality report", "validate")
else:
    st.markdown(dq)

st.subheader("Frozen configuration")
fz = data.frozen()
if fz is None:
    st.write("Not frozen yet.")
else:
    st.code(yaml.safe_dump(fz, sort_keys=False), language="yaml")
