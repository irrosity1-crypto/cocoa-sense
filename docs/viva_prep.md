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
An LSTM was added afterwards as a stretch model with its own validation → freeze → final cycle
(D16). As expected with ~700 monthly training sequences, it was the least accurate model overall
(1-month MAPE 8.13 %), but the best in the 2023–25 surge, because it learned momentum where the
trees learned mean reversion. Neither belief held across regimes.

**What does it look like in cedis?**
Converted at the delivery-month rate (D17), the ranking of strategies is the same, because every
strategy on a date gets the same rate. Simulated cedi revenue per tonne rose ×7.7 from 2020 to
2025; about 60 % of that (in log terms) was cocoa prices and 40 % cedi depreciation.

**Why not use real futures prices?**
No free, openly licensed contract-by-contract history exists for ICE cocoa (D18). But the World
Bank price is already a futures price: the ICCO average of the nearest three positions in New York
and London. So the forward price is a near-futures average, not a spot price. What it misses is
the exact contract for each delivery month, which matters most when the curve is steep, as in 2024.

**Is this Ghana's real gain or loss?**
No. It is a simulation on public monthly prices with an approximate forward price. Contract terms,
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
Contract-level futures data (licensed) for the exact delivery contract; a volatility-regime guard that falls back
to a fixed split; supply-side features (production, weather) obtained legally; a comparison with
periodically refitted models, reported separately from the frozen result.

**How can I reproduce your numbers?**
`uv sync --all-groups && uv run cocoasense all`. Raw files and their SHA-256 hashes are
committed; seeds are fixed and Random Forest runs single-threaded so reruns are bit-for-bit
identical (decision D15).
