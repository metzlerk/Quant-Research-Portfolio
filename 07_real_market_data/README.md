# Real Market Data Validation

## Overview

Modules 1-6 of this portfolio are built and validated primarily against simulated data
(GBM price paths, Ornstein-Uhlenbeck processes, synthetic order books) so that estimators
can be checked against a known ground truth. This module does the complementary exercise:
it pulls **live, real market data** and uses it to stress-test four of the portfolio's
models against what actually happened.

1. **Walk-forward GARCH volatility forecasting** (validates Module 1) — refit on a
   trailing real-return window, forecast forward, compare to what volatility actually
   realized.
2. **Rolling VaR backtesting with the Kupiec (1995) test** (validates Module 4) — daily
   95% historical VaR re-estimated on trailing real returns, checked against actual
   breaches, and formally tested for correct coverage.
3. **Implied volatility from a live option chain** (validates Module 5) — Black-Scholes
   implied vol recovered from real, currently-listed SPY option quotes using this
   portfolio's own solver, cross-checked against an independent source (Yahoo Finance).
4. **Out-of-sample strategy backtests** (validates Module 2) — the Mean Reversion and
   Momentum strategies, unmodified and untuned, run on a real multi-asset ETF universe.

## Why This Module Exists

A model that is only ever checked against data drawn from its own assumptions can look
arbitrarily good. This module asks the harder question: does it still behave sensibly once
the ground truth is unknown, the data is real and noisy, and the market doesn't oblige your
distributional assumptions? Results are reported as-is, including the places where a model
visibly strains (a volatility spike no backward-looking model could have forecast, an
untuned strategy with an unremarkable Sharpe ratio) rather than tuned to look better.

## Implementation Details

### Key Functions (`real_market_data.py`)

1. **`fetch_close_prices`**: Wide-format close-price history for a ticker list, via the
   shared `DataManager` cache (`utils/data_utils.py`).
2. **`fetch_option_chain`**: Live listed option chain for a ticker via `yfinance`, with an
   optional `min_days_out` filter to skip noisy 0-1 day expiries.
3. **`implied_vol_smile`**: Recovers Black-Scholes implied volatility per strike from real
   bid/ask quotes using Module 5's `BlackScholesPricer.implied_volatility`, filtering out
   illiquid and wide-spread quotes that would otherwise produce unstable solves.
4. **`rolling_garch_vol_backtest`**: Walk-forward GARCH(1,1) forecast evaluation using
   Module 1's `GarchModel`.
5. **`rolling_var_backtest`** / **`kupiec_pof_test`**: Walk-forward historical VaR backtest
   and the Kupiec proportion-of-failures likelihood-ratio test, using Module 4's
   `calculate_var_es`.

### A Note on Data Caching

While building this module we found and fixed two real bugs in the shared `DataManager`
cache (`utils/data_utils.py`) that had gone unnoticed because no notebook previously
exercised the caching path (all of Modules 1, 3, and 4 call `yfinance` directly instead):

- `_cache_data` crashed outright on current `yfinance` versions, which omit the `'Adj
  Close'` column when `auto_adjust=True` (now the default).
- `_cache_data` wrote with `to_sql(..., if_exists='replace')`, which drops and rebuilds the
  *entire* cache table on every call — caching a second symbol silently wiped out the first.
- `fetch_equity_data` treated any cached rows inside `[start_date, end_date]` as a full
  cache hit, even if the cached range didn't actually reach back to `start_date` — so
  requesting a wider date range after a narrower one was cached would silently return
  incomplete history instead of fetching the missing older data.

All three are covered by regression tests in `tests/test_portfolio.py`.

## Usage Examples

### Real Price Data

```python
from real_market_data import fetch_close_prices

prices = fetch_close_prices(['SPY', 'QQQ', 'TLT', 'GLD'], '2018-01-01', '2026-08-23',
                             data_dir='../data', use_cache=True)
returns = prices.pct_change().dropna()
```

### Walk-Forward GARCH Backtest

```python
from real_market_data import rolling_garch_vol_backtest

backtest = rolling_garch_vol_backtest(returns['SPY'], window=500, refit_every=63,
                                       forecast_horizon=21)
```

### VaR Backtest with the Kupiec Test

```python
from real_market_data import rolling_var_backtest, kupiec_pof_test

backtest = rolling_var_backtest(returns['SPY'], window=500, confidence=0.95)
kupiec = kupiec_pof_test(len(backtest), int(backtest['violation'].sum()), confidence=0.95)
```

### Live Option Chain Implied Volatility

```python
from real_market_data import fetch_option_chain, implied_vol_smile

expiry, spot, calls, puts = fetch_option_chain('SPY', min_days_out=25)
smile = implied_vol_smile(calls, spot, expiry, pd.Timestamp.now().normalize(),
                           rate=0.04, option_type='call')
```

## Notebook

For the full walkthrough with real, live-pulled results and figures, see:

- `real_market_data_analysis.ipynb`

## Caveats

- Results reflect whatever the market was doing on the day the notebook was run; re-running
  it later will pull fresh data and produce different (but equally real) numbers.
- The strategy universe (4 broad ETFs) is intentionally small and untuned — it demonstrates
  honest out-of-sample behavior, not a production-ready trading system.
- Yahoo Finance's option chain data can have stale or zero implied-vol fields for illiquid
  contracts; `implied_vol_smile` filters on volume and bid/ask spread width to mitigate this.

## References

1. Kupiec, P. (1995). "Techniques for verifying the accuracy of risk measurement models."
   *Journal of Derivatives*, 3(2), 73-84.
2. Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity."
   *Journal of Econometrics*, 31(3), 307-327.
3. Hull, J. (2018). *Options, Futures, and Other Derivatives*. Pearson.

## Author

Kevin J.D. Metzler
PhD Candidate, Mathematical Sciences, Worcester Polytechnic Institute

## License

This module is part of the Quantitative Research Portfolio.
