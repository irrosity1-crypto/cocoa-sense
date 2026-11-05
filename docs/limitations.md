# Limitations

1. **Simulation, not reality.** Strategy results are computed on public monthly prices under the
   assumptions in `simulation_assumptions.md`. They are not Ghana's or COCOBOD's gains or losses,
   which depend on contract terms, delivery obligations, counterparty risk and costs that are not
   public.
2. **Approximate forward price.** No free, openly licensed source of contract-by-contract ICE
   cocoa settlements was found (ICE sells end-of-day history; Barchart needs a login and forbids
   redistribution; free sites offer only a continuous front-month series), so the exact contract
   for each delivery month is not used (decision D18). The forward price is the decision month's
   World Bank/ICCO price, which is itself a futures price: the average of the nearest three
   futures positions in New York and London (Pink Sheet definition). It therefore already
   reflects prices for delivery a few months ahead, but it averages several maturities and two
   exchanges, and is a monthly average. During the 2024 surge the curve was widely reported to be
   in steep backwardation (later delivery priced well below nearby); averaging over three
   positions softens but does not remove that effect, so surge-period results for forward selling
   should be read with care.
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
   covers a slightly shorter test window than the main models, and the cedi revenue view covers
   deliveries to January 2026 only (73 of 82). The cedi view uses a monthly average rate; the
   rates COCOBOD actually received are not public.
8. **No fundamentals.** Production, grindings and stocks data (paid ICCO bulletins) are not used.
9. **Model scope.** Exact-contract futures prices are not used (item 2, decision D18).
   The LSTM stretch model was added after the main test run (decision D16). Its configuration and
   frozen settings were committed before its own test run, but it was designed by someone who had
   already seen the other models' test results, so its test is not fully blind. It is reported
   separately (`docs/results.md` section 8) and does not change the main results.
10. **Confidence labels and the learning track (D19).** The labels summarise how similar past
    forecasts turned out; they are not guarantees. Checked on 2020–2026 they were informative only
    in part (a High label at 1 month and a Low label at 3 months held up; the others did not rank
    outcomes), and the up/down probability had no skill. Monthly refitting did not improve
    accuracy. "Live" prices are the latest published monthly averages, not intraday quotes.
