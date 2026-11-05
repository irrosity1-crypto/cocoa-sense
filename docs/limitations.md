# Limitations

1. **Simulation, not reality.** Strategy results are computed on public monthly prices under the
   assumptions in `simulation_assumptions.md`. They are not Ghana's or COCOBOD's gains or losses,
   which depend on contract terms, delivery obligations, counterparty risk and costs that are not
   public.
2. **Spot proxy for forward prices.** Without contract-level futures data, the forward price is
   approximated by the decision-month spot price, ignoring the futures basis. Cocoa futures were
   widely reported to be in steep backwardation during the 2024 surge (later delivery priced below
   spot); if so, this proxy overstates what forward selling would have earned in that period.
   Verify with contract-level data before relying on surge-period strategy results.
3. **Monthly averages.** Both forecasts and simulated sales use monthly average prices, which are
   smoother than the daily prices at which contracts are actually struck.
4. **Small sample.** Roughly 800 months of history, of which the test period has about 80 months
   per horizon and the surge period about 25. A few months can dominate the scores; the
   Diebold–Mariano test has low power at these sizes.
5. **Frozen models.** By design (no training after 2019) the models cannot adapt to the 2023
   regime change. This is a deliberate test of out-of-sample robustness, and also a reason the
   models may underperform a periodically refitted alternative.
6. **Regime break.** The 2023–25 price surge was driven by supply shocks (weather, disease,
   smuggling, under-investment) that are not in the price history the models see.
7. **Exchange rate coverage.** The monthly FX series ends January 2026, so the FX experiment
   covers a slightly shorter test window than the main models.
8. **No fundamentals.** Production, grindings and stocks data (paid ICCO bulletins) are not used.
9. **Model scope.** Futures-curve simulation is a stretch item and is not part of the results.
   The LSTM stretch model was added after the main test run (decision D16). Its configuration and
   frozen settings were committed before its own test run, but it was designed by someone who had
   already seen the other models' test results, so its test is not fully blind. It is reported
   separately (`docs/results.md` section 8) and does not change the main results.
