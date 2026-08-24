# Market Microstructure

## Overview

This module provides comprehensive tools for analyzing market microstructure, the study of how financial markets operate at fine temporal and spatial scales. Key topics include:

1. **Order Book Dynamics**: Limit order book (LOB) modeling and analysis
2. **Bid-Ask Spread Decomposition**: Separating adverse selection from inventory costs
3. **Market Impact Models**: Linear and power-law impact estimation
4. **Optimal Execution**: Almgren-Chriss framework for minimizing execution costs
5. **High-Frequency Metrics**: Realized volatility, clustering, duration analysis
6. **Illiquidity Measures**: Amihud and Roll spread estimators

## Mathematical Foundations

### Limit Order Book

The LOB aggregates supply and demand at discrete price levels:

- **Bid side**: Buy orders from best (highest) to worst prices
- **Ask side**: Sell orders from best (lowest) to worst prices
- **Spread**: $S = P_{\text{ask}} - P_{\text{bid}}$
- **Mid-price**: $M = \frac{P_{\text{bid}} + P_{\text{ask}}}{2}$
- **Imbalance**: $I = \frac{\text{BidDepth} - \text{AskDepth}}{\text{BidDepth} + \text{AskDepth}}$

### Market Impact Models

**Linear Impact:**
$$I(v) = \alpha + \beta \cdot \frac{v}{V}$$

**Power-Law Impact:**
$$I(v) = c \cdot \left(\frac{v}{V}\right)^\lambda$$

where $v$ = order volume, $V$ = market volume, and $\lambda \in [0.5, 1.5]$ empirically.

Decomposition into temporary and permanent components:
- **Temporary**: Market rebounds after order execution
- **Permanent**: Persistent price change reflecting new information

### Almgren-Chriss Optimal Execution

Minimize expected cost:

$$\min_{\mathbf{x}} \left[ \sum_{i=1}^N C(x_i) + \lambda \sum_{i=1}^N g_i(x_i) \right]$$

where:
- $x_i$ = execution at period $i$
- $C(x_i)$ = direct execution cost (spread + temporary impact)
- $g_i(x_i)$ = permanent impact, discounted by remaining periods
- $\lambda$ = risk aversion parameter

### High-Frequency Metrics

**Realized Volatility:**
$$RV = \sqrt{\sum_i r_i^2}$$

**Amihud Illiquidity:**
$$\text{ILLIQ} = \text{E}\left[\frac{|r_i|}{P_i \cdot V_i}\right]$$

**Roll Spread Estimator:**
$$S_{\text{Roll}} = 2\sqrt{-\text{Cov}(\Delta P_t, \Delta P_{t+1})}$$

**Order Clustering - Fano Factor:**
$$F = \frac{\text{Var}(N)}{\text{Mean}(N)}$$

where $F > 1$ indicates clustering (orders don't arrive as Poisson).

## Implementation Details

### Key Classes

1. **`OrderBookSnapshot`**: Represents LOB state with prices and volumes
2. **`LimitOrderBook`**: Aggregates snapshots and computes metrics
3. **`LinearImpactModel` & `PowerLawImpactModel`**: Market impact estimation
4. **`OptimalExecutionAC`**: Solves Almgren-Chriss execution problem
5. **`HighFrequencyMetrics`**: Computes HFT-specific measures

### Microstructure Metrics

The module provides:
- Spread statistics (mean, std, distribution)
- Order imbalance trends
- Depth profile analysis
- Bid-ask bounce detection
- Adverse selection vs. inventory cost decomposition

## Usage Examples

### Order Book Analysis

```python
from market_microstructure import OrderBookSnapshot, LimitOrderBook

# Create snapshots with bid/ask prices and volumes
snapshot = OrderBookSnapshot(
    timestamp=0.0,
    bid_prices=np.array([100.00, 99.99, 99.98]),
    bid_volumes=np.array([1000, 500, 300]),
    ask_prices=np.array([100.01, 100.02, 100.03]),
    ask_volumes=np.array([800, 400, 200])
)

# Analyze book
lob = LimitOrderBook()
lob.add_snapshot(snapshot)

# Compute metrics
metrics = lob.compute_microstructure_metrics(window=20)
spread_components = lob.estimate_spread_components()
```

### Market Impact Comparison

```python
from market_microstructure import LinearImpactModel, PowerLawImpactModel

# Define models
linear = LinearImpactModel(alpha=0.001, beta_temp=0.0005)
power = PowerLawImpactModel(lambda_temp=0.5)

# Compute impact for 100k shares with 1M daily volume
volume = 100000
daily_vol = 1e6

temp_lin, perm_lin = linear.compute_impact(volume, daily_vol)
temp_pow, perm_pow = power.compute_impact(volume, daily_vol)

print(f"Linear: temp={temp_lin:.4f}, perm={perm_lin:.4f}")
print(f"Power:  temp={temp_pow:.4f}, perm={perm_pow:.4f}")
```

### Optimal Execution

```python
from market_microstructure import OptimalExecutionAC, LinearImpactModel

# Define execution problem
impact_model = LinearImpactModel()
executor = OptimalExecutionAC(
    initial_quantity=100000,
    time_horizon=1.0,      # 1 day
    num_periods=20,         # 20 periods
    impact_model=impact_model,
    daily_volume=1e6
)

# Solve with different risk aversion
schedule_aggressive = executor.optimal_execution(lambda_risk=1e-7)
schedule_patient = executor.optimal_execution(lambda_risk=1e-4)

# Higher λ = more patient (spread execution)
```

### High-Frequency Metrics

```python
from market_microstructure import HighFrequencyMetrics

# Create metrics from trade data
trade_data = pd.DataFrame({
    'timestamp': [...],
    'price': [...],
    'quantity': [...]
})

metrics = HighFrequencyMetrics(trade_data)

# Compute various measures
realized_vol = metrics.compute_realized_volatility(window=50)
amihud = metrics.compute_amihud_illiquidity()
roll_spread = metrics.compute_roll_spread()
clustering = metrics.compute_order_clustering()
duration = metrics.compute_order_duration()
```

## Notebooks

For detailed walkthroughs and visualizations, see:

- `market_microstructure_analysis.ipynb` - Comprehensive analysis with examples

## Key Empirical Findings

### Bid-Ask Spread Components

Empirical studies show:
- **Inventory cost**: 40-60% of spread (short-term variation)
- **Adverse selection**: 30-50% of spread (informed trading)
- **Order processing**: 10-20% of spread (market maker cost)

### Market Impact

- Temporary impact ≈ 0.01-0.05% per $1M traded
- Permanent impact ≈ 0.002-0.01% per $1M traded
- Power-law exponent $\lambda$ typically 0.4-0.6 (concave)

### Order Clustering

- Fano factor typically 1.5-3.0 (significant clustering)
- Hurst exponent 0.50-0.65 (persistent clustering pattern)
- Inter-arrival times exhibit long-range dependence

## References

1. **Almgren, R., & Chriss, N. (2001)**. "Optimal execution of portfolio transactions." *Journal of Risk*, 3(2), 5-39.

2. **Roll, R. (1984)**. "A simple implicit measure of the effective bid-ask spread in an efficient market." *Journal of Finance*, 39(4), 1127-1139.

3. **Hasbrouck, J., & Saar, G. (2013)**. "Low-latency trading." *Review of Financial Studies*, 26(9), 2888-2925.

4. **Fong, K. Y., & Holbrook, A. K. (2016)**. "Information and volatility linkages in the listing day returns of newly listed Chinese stocks." *Pacific-Basin Finance Journal*, 40, 381-392.

5. **Biais, B., Hillion, P., & Spatt, C. (1995)**. "An empirical analysis of the limit order book and the order flow in the Paris Bourse." *Journal of Finance*, 50(5), 1655-1689.

## Testing

Run unit tests:

```bash
python test_market_microstructure.py
```

Tests cover:
- Order book snapshots and metrics
- Spread decomposition
- Market impact models
- Optimal execution
- HFT metrics computation

## Author

Kevin J.D. Metzler
PhD Candidate, Mathematical Sciences, Worcester Polytechnic Institute

## License

This module is part of the Quantitative Research Portfolio.
