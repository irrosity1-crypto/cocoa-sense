# Simulation assumptions

The simulation asks: *had a seller followed each rule at every decision month, what revenue per
tonne would it have earned, given public prices?* It is a stylised experiment. It is **not** a
reconstruction of Ghana's or COCOBOD's actual forward contracts, whose terms are not public.

## Model of one decision

At the end of decision month *t* the seller has *Q* tonnes to deliver *h* months later. It sells a
share *w* forward at price *F_t* and the remainder at the delivery-month price *P_{t+h}*:

    R = Q · [ w · F_t + (1 − w) · P_{t+h} ] − C_fin
    C_fin = r · (h / 12) · (1 − w) · Q · F_t

## Assumptions

1. **Forward price (spot proxy).** With free data only, *F_t* is the monthly average spot price of
   month *t*. Real forward/futures prices differ from spot by a basis (carry, convenience yield),
   which is ignored. A `CsvFuturesProvider` accepts user-supplied settlement prices
   (`date, contract_month, settle`) to replace this; none were used for the reported results.
2. **Delivery price.** *P_{t+h}* is the monthly average price of the delivery month (the same
   series as the forecasts). Intra-month timing is ignored.
3. **Financing cost.** The unsold share is financed until delivery at an annual rate *r* (default
   10 %; sensitivity 0–20 %), valued at *F_t* so the cost is known at decision time. Forward sales
   carry no financing cost in this model; in practice they also support pre-export financing.
4. **Quantity.** A constant hypothetical *Q* (default 1,000 t) at every decision month. Revenue is
   linear in *Q*, so results are reported per tonne.
5. **No transaction costs, margin calls, counterparty risk, quality premiums or currency
   conversion.** Revenue is in US dollars.
6. **Independent decisions.** Each decision month is evaluated separately (overlapping 3-month
   positions are not netted); this is a comparison of rules, not a cash-flow plan.
7. **Information.** The forecast-driven share uses only the frozen model's forecast and interval
   made with data up to *t*, and *F_t* (leakage test 5).

## Strategies

| Strategy | Rule |
|---|---|
| Sell forward | w = 1 |
| Wait | w = 0 |
| Fixed 50/50 | w = 0.5 |
| Forecast-driven | w = clip(0.5 + a · z, 0, 1), z = (F_t − forecast) / (½ · interval width) |

`a` and the forecasting model that drives the rule are chosen on the validation period only
(decision months 2015-10 to 2019-09, horizon 3), by highest mean revenue per tonne, and frozen in
`configs/frozen.yaml` before the test run. `a = 0` reproduces Fixed 50/50.

## Metrics

Mean revenue per tonne; standard deviation; mean of the worst 10 % of outcomes; regret (average
shortfall against the best of the three fixed strategies at each date, which is only knowable in
hindsight); win rate against Fixed 50/50. Reported for the full test period and by delivery-month
sub-period (pre-surge, surge, post-peak).

## Stress scenarios (hypothetical)

Delivery price set to 1.20×, 0.85× or 1.00× the decision-date price, holding the forecast-driven
shares fixed. These show how each rule behaves under a given move; they are not predictions.
