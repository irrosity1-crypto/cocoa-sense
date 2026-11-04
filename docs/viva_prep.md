# Viva preparation

Likely questions with evidence-based answers. Numbers refer to `docs/results.md`.

**Why not just predict the last price?**
That is the naive baseline, the first row of every results table. The point of the study is
whether anything beats it. Out of sample, nothing did by a meaningful or significant margin
(best: ARIMA, 3.6 % lower MAE at 1 month, DM p = 0.77).

**Did you use information from the future?**
No. Splits are by time only; features at month t use data up to t; models were fitted once on
data to December 2019 and never refitted. Five automated leakage tests (`tests/leakage/`) check
truncation, perturbation (×10 future prices change nothing), training windows, fit-on-train-only
and strategy decisions. The test period is locked in code behind `--final`, and the frozen
configuration was committed (`4c076fa`) before the logged final run.

**Why freeze the models instead of retraining each month?**
The requirement was that no model is trained on post-2019 data, so every forecast can be
compared honestly with what happened. It is a stricter test than an expanding window. The cost
(models cannot adapt to the 2023 regime) is part of the finding.

**Your models failed. Is the project a failure?**
No. The research question is whether forecasts improve decisions; a rigorous "no", with an
explanation of when and why, answers it. The explanation is specific: the models learned
mean reversion from 1991–2019 and the surge broke it; intervals were too narrow once volatility
doubled.

**Why these models?**
A ladder of complexity: naive → statistical (ARIMA) → machine learning (Random Forest, XGBoost).
LSTM was left as a stretch goal: with ~400–800 monthly points it would mostly fit noise.

**Is this Ghana's real gain or loss?**
No. It is a simulation on public monthly prices with a spot-proxy forward price. Contract terms,
basis, financing arrangements and delivery obligations are not public (`docs/limitations.md`).

**How did you choose `a` and the training start?**
On the 2016–2019 validation period only, with models trained on data to 2015, then frozen in
`configs/frozen.yaml` (decisions D7, D9, D14). `a` landed on the edge of its pre-declared grid;
the grid was not widened to avoid overfitting 48 validation months.

**Does the exchange rate matter?**
Not for forecasting the world price: paired tests changed MAE by under 1 % in either direction.

**Why the World Bank series and not FRED?**
Twice the history (1960 vs 1992), same underlying ICCO price; they agree to within 0.43 % on
average over 417 common months (`artifacts/data_quality.md`).

**What would you do next?**
Contract-level futures data to replace the spot proxy; a volatility-regime guard that falls back
to a fixed split; supply-side features (production, weather) obtained legally; a comparison with
periodically refitted models, reported separately from the frozen result.

**How can I reproduce your numbers?**
`uv sync --all-groups && uv run cocoasense all`. Raw files and their SHA-256 hashes are
committed; seeds are fixed and Random Forest runs single-threaded so reruns are bit-for-bit
identical (decision D15).
