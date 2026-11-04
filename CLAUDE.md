# CocoaSense AI — standing rules for Claude Code

Final-year Computer Science & Engineering project. It forecasts the monthly world cocoa price,
simulates forward-sales strategies for a cocoa exporter such as Ghana, and presents results in a
Streamlit dashboard. This is assessed academic work: correctness, reproducibility and honesty
matter more than the number of features.

## Non-negotiable rules

1. **Training cut-off.** No model is ever trained on a target month later than **December 2019**
   (`final_train_end`). Test-period forecasts (January 2020 onward) use frozen parameters; models
   may read observed history as input but are never refitted on it. This is the author's explicit
   requirement so that forecasts can be compared with the actual prices.
2. **Never fabricate data or results.** Every number in the dashboard, README or docs must come
   from code run on real data. Synthetic fixtures are allowed only in tests and are labelled.
3. **Free, legal data only.** No paywalled data (ICCO subscriptions, ICE, Databento), no
   credentials, no working around blocks. If a download fails, ask the user to place the file in
   `data/raw/manual/`.
4. **Prevent look-ahead.** Split by time only. Features at month t use data up to t. The five
   leakage tests in `tests/leakage/` must always pass.
5. **Test period is touched once.** Tune on validation (2016–2019), write and commit
   `configs/frozen.yaml`, then run `cocoasense evaluate --final`. Never tune on test results.
6. **State limitations.** Results are simulations on public prices, never Ghana's actual
   gains or losses, and never a promise of accurate forecasts.
7. **Keep it simple and explainable.** Baseline first; readable, tested code.
8. **Ask when it matters**; record cheap-to-reverse choices in `docs/decisions.md`.

## Architecture (each layer reads only from the one before)

ingestion → validation → features → models → evaluation → simulation → dashboard.
The dashboard reads `artifacts/` and `data/` only and never trains. Everything tunable is in
`configs/`. Seeds come from config.

## Quality gate (every change)

`ruff check` + `ruff format --check` clean, `mypy` clean, all tests passing, coverage ≥ 80 % on
`src/` (dashboard excluded), docs updated. On Windows without make:
`uv run ruff check src tests; uv run mypy; uv run pytest --cov`.

## Working agreement

- Plan before coding; run tests before claiming they pass; never skip a failing test or lower a
  threshold.
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`), one branch per phase.
- After each phase report: what was built, commands and real results, key numbers with source
  files, deviations, open questions, next phase.
- Never: invent numbers, look at the test period before the freeze, describe simulation output as
  real profit or loss, add unplanned features without asking, delete raw data.
