"""
Real Market Data Validation

This module pulls real, live market data (equity prices and listed option
chains via Yahoo Finance) and uses it to *validate* the theoretical models
built elsewhere in this portfolio against actual market outcomes:

- Walk-forward GARCH(1,1) volatility forecasting (Module 1) evaluated
  against subsequently realized volatility.
- Rolling historical Value-at-Risk (Module 4) backtested with the Kupiec
  (1995) proportion-of-failures likelihood-ratio test.
- Black-Scholes implied volatility (Module 5) recovered from live listed
  option quotes to trace out the empirical volatility smile/skew.

The rest of the portfolio is largely built and validated on simulated data
(GBM, OU processes, synthetic order books) because that lets the underlying
mathematics be checked against known ground truth. This module closes the
loop: it asks whether the models still behave sensibly once the ground
truth is unknown and the data is real, noisy, and non-stationary.

Author: Kevin J.D. Metzler
"""

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# Reuse the shared utilities and per-module pricing/risk/volatility code
# rather than re-implementing it here.
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _module_dir in ("utils", "01_volatility_modeling", "04_risk_management", "05_derivatives_pricing"):
    sys.path.append(os.path.join(_ROOT_DIR, _module_dir))


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

def fetch_close_prices(
    tickers: List[str],
    start_date: str,
    end_date: str,
    data_dir: str = "../data",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of tickers as a wide DataFrame.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to fetch.
    start_date, end_date : str
        'YYYY-MM-DD' bounds for the price history.
    data_dir : str
        Directory used by the shared ``DataManager`` cache.
    use_cache : bool
        Whether to read/write the on-disk cache.

    Returns
    -------
    pd.DataFrame
        Columns are tickers, index is trading date, values are (adjusted)
        close prices. Tickers that fail to download are silently dropped.
    """
    from data_utils import DataManager

    dm = DataManager(data_dir=data_dir)
    closes = {}
    for ticker in tickers:
        data = dm.fetch_equity_data(ticker, start_date, end_date, use_cache=use_cache)
        if isinstance(data.columns, pd.MultiIndex):
            data = data[ticker]
        if len(data) > 0 and "Close" in data.columns:
            close = data["Close"]
            # Fresh yfinance pulls are tz-aware; cached (SQLite round-trip)
            # data is tz-naive. Normalize both to tz-naive before combining,
            # otherwise pandas refuses to build a shared index.
            close.index = pd.to_datetime(close.index).tz_localize(None)
            closes[ticker] = close

    prices = pd.DataFrame(closes).sort_index()
    return prices.dropna(how="all")


@dataclass
class OptionQuote:
    """A single real, live listed option quote pulled from the chain."""
    strike: float
    bid: float
    ask: float
    last_price: float
    market_implied_vol: float
    volume: float
    open_interest: float

    @property
    def mid_price(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return 0.5 * (self.bid + self.ask)
        return self.last_price


def fetch_option_chain(
    ticker: str,
    expiry: Optional[str] = None,
    min_days_out: int = 0,
) -> Tuple[str, float, pd.DataFrame, pd.DataFrame]:
    """
    Fetch a real, live listed option chain for ``ticker``.

    Parameters
    ----------
    ticker : str
        Underlying ticker symbol.
    expiry : str, optional
        'YYYY-MM-DD' expiry to fetch. If not given, the nearest listed expiry
        that is at least ``min_days_out`` calendar days away is used.
    min_days_out : int
        Skip very-near-dated (e.g. 0-1 day) expiries by default, since their
        quotes are dominated by pin-risk/gamma noise rather than a smooth
        volatility smile. Falls back to the nearest available expiry if
        nothing qualifies.

    Returns
    -------
    Tuple of (expiry used, current spot price, calls DataFrame, puts DataFrame)
    """
    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        raise ValueError(f"No listed options found for {ticker}.")

    if expiry is None:
        today = pd.Timestamp.now().normalize()
        candidates = [e for e in expiries if (pd.Timestamp(e) - today).days >= min_days_out]
        expiry = candidates[0] if candidates else expiries[0]
    elif expiry not in expiries:
        raise ValueError(f"{expiry} is not a listed expiry for {ticker}. Available: {expiries}")

    spot = tk.history(period="1d")["Close"].iloc[-1]
    chain = tk.option_chain(expiry)
    return expiry, float(spot), chain.calls, chain.puts


# ---------------------------------------------------------------------------
# Module 5 validation: implied volatility smile from real option quotes
# ---------------------------------------------------------------------------

def implied_vol_smile(
    options_df: pd.DataFrame,
    spot: float,
    expiry: str,
    as_of: pd.Timestamp,
    rate: float,
    option_type: str,
    dividend_yield: float = 0.0,
    min_volume: int = 1,
    max_relative_spread: float = 0.5,
) -> pd.DataFrame:
    """
    Recover Black-Scholes implied volatility from real market option quotes
    using this portfolio's own ``BlackScholesPricer.implied_volatility``
    solver (Module 5), and compare it to Yahoo's reported implied vol.

    Parameters
    ----------
    options_df : pd.DataFrame
        The `calls` or `puts` frame returned by :func:`fetch_option_chain`.
    spot : float
        Current underlying spot price.
    expiry : str
        'YYYY-MM-DD' expiry date matching ``options_df``.
    as_of : pd.Timestamp
        Valuation date (usually "now").
    rate : float
        Continuously-compounded risk-free rate to use in the model.
    option_type : str
        'call' or 'put'.
    dividend_yield : float
        Continuous dividend yield assumption for the underlying.
    min_volume : int
        Drop illiquid strikes with less than this much traded volume, since
        their quotes are too stale/wide to be informative.
    max_relative_spread : float
        Drop quotes whose (ask - bid) / mid exceeds this fraction. Wide
        two-sided spreads on thin strikes translate into large, unstable
        swings in implied volatility that reflect market microstructure
        noise rather than a genuine view on volatility.

    Returns
    -------
    pd.DataFrame
        One row per strike with market mid-price, our recovered model
        implied vol, Yahoo's reported implied vol, and moneyness.
    """
    from derivatives_pricing import OptionParams, BlackScholesPricer

    maturity = max((pd.Timestamp(expiry) - as_of).days, 1) / 365.0

    rows = []
    for _, row in options_df.iterrows():
        volume = row.get("volume", 0) or 0
        if volume < min_volume:
            continue

        # Require a genuine two-sided market rather than falling back to a
        # potentially stale last-trade price, which can be days old for a
        # thin strike and wildly inconsistent with the current spot.
        bid, ask = row.get("bid", 0.0), row.get("ask", 0.0)
        if bid <= 0 or ask <= 0 or ask < bid:
            continue

        mid = 0.5 * (bid + ask)
        if mid <= 0.01 or (ask - bid) / mid > max_relative_spread:
            continue

        params = OptionParams(
            spot=spot,
            strike=float(row["strike"]),
            maturity=maturity,
            rate=rate,
            volatility=0.3,  # placeholder, only used for OptionParams validation
            dividend_yield=dividend_yield,
            option_type=option_type,
        )
        try:
            model_iv = BlackScholesPricer.implied_volatility(mid, params)
        except ValueError:
            continue

        rows.append({
            "strike": float(row["strike"]),
            "moneyness": float(row["strike"]) / spot,
            "mid_price": mid,
            "volume": volume,
            "open_interest": row.get("openInterest", np.nan),
            "model_implied_vol": model_iv,
            "yahoo_implied_vol": row.get("impliedVolatility", np.nan),
        })

    return pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Module 1 validation: walk-forward GARCH volatility forecasting
# ---------------------------------------------------------------------------

def rolling_garch_vol_backtest(
    returns: pd.Series,
    window: int = 500,
    refit_every: int = 21,
    forecast_horizon: int = 21,
) -> pd.DataFrame:
    """
    Walk-forward evaluation of GARCH(1,1) volatility forecasts against
    subsequently realized volatility.

    On a rolling basis, refit a GARCH(1,1) model on the trailing ``window``
    of returns every ``refit_every`` trading days, forecast cumulative
    variance over the next ``forecast_horizon`` days, and compare it to the
    variance that actually realized over that same forward window. This is
    an honest out-of-sample test: at each refit point the model only ever
    sees data strictly before it.

    Parameters
    ----------
    returns : pd.Series
        Daily simple or log returns, in decimal form (e.g. 0.01 = 1%).
    window : int
        Trailing estimation window, in trading days.
    refit_every : int
        How often (in trading days) to refit the model.
    forecast_horizon : int
        Forward window (in trading days) used to both forecast and to
        measure realized variance.

    Returns
    -------
    pd.DataFrame
        One row per refit date with forecast and realized annualized
        volatility, plus the forecast error.
    """
    from volatility_models import GarchModel, VolatilityModelConfig

    returns = returns.dropna()
    results = []

    start = window
    end = len(returns) - forecast_horizon
    for t in range(start, end, refit_every):
        train = returns.iloc[t - window: t]
        forward = returns.iloc[t: t + forecast_horizon]
        if len(forward) < forecast_horizon:
            break

        config = VolatilityModelConfig(model_type="GARCH", distribution="normal", p=1, q=1)
        model = GarchModel(config)
        try:
            model.fit(train)
            forecast = model.forecast(horizon=forecast_horizon)
        except Exception:
            continue

        # GarchModel rescales returns by 100 internally when magnitudes are
        # small, so forecast variance is in "percent^2" units; convert back
        # to decimal-return variance units for a fair comparison.
        rescaled = model.config.rescale and train.abs().mean() < 0.1
        scale = 1e-4 if rescaled else 1.0

        forecast_var_cum = float(np.sum(forecast["variance"])) * scale
        realized_var_cum = float(np.sum(forward.values ** 2))

        forecast_vol_ann = np.sqrt(forecast_var_cum / forecast_horizon * 252)
        realized_vol_ann = np.sqrt(realized_var_cum / forecast_horizon * 252)

        results.append({
            "refit_date": returns.index[t],
            "forecast_vol_annualized": forecast_vol_ann,
            "realized_vol_annualized": realized_vol_ann,
            "forecast_error": forecast_vol_ann - realized_vol_ann,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Module 4 validation: rolling VaR backtest with the Kupiec (1995) test
# ---------------------------------------------------------------------------

def rolling_var_backtest(
    returns: pd.Series,
    window: int = 500,
    confidence: float = 0.95,
    method: str = "historical",
) -> pd.DataFrame:
    """
    Walk-forward historical/parametric VaR backtest.

    At each day t (t > window), estimate one-day VaR from the trailing
    ``window`` of returns using this portfolio's own ``calculate_var_es``
    (Module 4), then check whether the actual return on day t breached it.
    Only past data is used at every step.

    Returns
    -------
    pd.DataFrame
        Columns: date, var_estimate, actual_return, violation (bool).
    """
    from risk_utils import calculate_var_es

    returns = returns.dropna()
    records = []
    for t in range(window, len(returns)):
        train = returns.iloc[t - window: t]
        actual = returns.iloc[t]
        var_result = calculate_var_es(train, confidence_levels=[confidence], method=method)
        var_estimate = var_result[f"CL_{confidence:.0%}"]["VaR"]

        records.append({
            "date": returns.index[t],
            "var_estimate": var_estimate,
            "actual_return": actual,
            "violation": bool(actual < var_estimate),
        })

    return pd.DataFrame(records)


def kupiec_pof_test(n_obs: int, n_violations: int, confidence: float = 0.95) -> Dict[str, float]:
    """
    Kupiec (1995) proportion-of-failures likelihood-ratio test for
    unconditional VaR coverage.

    Under the null hypothesis that the VaR model has the correct violation
    rate p = 1 - confidence, the likelihood-ratio statistic

        LR_POF = -2 ln[ (1-p)^(n-x) p^x / (1-x/n)^(n-x) (x/n)^x ]

    is asymptotically chi-squared distributed with 1 degree of freedom,
    where n is the number of observations and x is the observed number of
    VaR violations.

    Returns
    -------
    dict with the observed/expected violation rates, the LR statistic, and
    the p-value. A small p-value (e.g. < 0.05) rejects the null hypothesis
    that the model's stated confidence level matches its actual coverage.
    """
    p = 1 - confidence
    x = n_violations
    n = n_obs
    p_hat = x / n if n > 0 else 0.0

    def _log_lik(prob: float, successes: int, trials: int) -> float:
        prob = min(max(prob, 1e-10), 1 - 1e-10)
        return (trials - successes) * np.log(1 - prob) + successes * np.log(prob)

    ll_null = _log_lik(p, x, n)
    ll_alt = _log_lik(p_hat, x, n)
    lr_stat = -2 * (ll_null - ll_alt)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    return {
        "n_obs": n,
        "n_violations": x,
        "expected_violation_rate": p,
        "observed_violation_rate": p_hat,
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "reject_null_at_5pct": bool(p_value < 0.05),
    }
