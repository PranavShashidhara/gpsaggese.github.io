# Causal Analysis of Financial Tradability

Analysis of cryptocurrency price predictability using causal inference, Monte Carlo simulation, and ML-based classification — applied to Bitcoin (BTC/USD) hourly data.

## Overview

This project investigates whether price movements in crypto markets are predictable enough to trade profitably, accounting for transaction costs, slippage, and market regime. It combines classical technical analysis with causal inference methods and simulation-based PnL attribution.

**Data source:** [Kaggle — Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data) (1-minute OHLCV, 2013–2021, by mczielinski). Falls back to synthetic BTC data if Kaggle credentials are unavailable.

## Repository Structure

```
.
├── master.ipynb                  # Main analysis notebook (Exercise #1 entry point)
└── financial_trading_utils.py    # All helper functions (data, features, simulation, analysis)
```

## Setup

**Dependencies:**

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
pip install kagglehub[pandas-datasets]   # optional, for live Kaggle data
```

## Usage

Open `master.ipynb` and run cells in order. The notebook imports `financial_trading_utils` as `ftu` and uses `%autoreload` so edits to the utils file are picked up without restarting the kernel.

```python
import financial_trading_utils as ftu

config = ftu.SimulationConfig(asset='BTC', frequency='1h',
                               start_date='2023-01-01', end_date='2023-12-31')
df = ftu.load_exercise_data(config)
returns, timestamps = ftu.compute_returns(df)
pnl_sims, stats = ftu.simulate_trading_with_hit_rate(returns, hit_rate=0.52)
```

## Exercise #1 — Hit Rate → PnL Framework

The main notebook exercise sweeps hit rates from 45% to 80% and runs 10,000 Monte Carlo simulations per rate using bootstrapped BTC hourly returns. Key outputs:

- **Breakeven hit rate** given commission (0.1%) + slippage (0.05%)
- **PnL distribution** (mean, std, 5th/95th percentiles) per hit rate
- **Probability of positive PnL** across the sweep
- **Sensitivity analysis**: Sharpe ratio and max drawdown vs. hit rate

### Key parameters (defaults)

| Parameter | Default | Notes |
|---|---|---|
| `transaction_cost` | 0.15% per trade | commission + slippage combined |
| `num_simulations` | 10,000 | Monte Carlo runs |
| `hit_rates` | 0.45 → 0.55 | swept in 1% steps |
| `frequency` | `1h` | resampled from 1-min Kaggle data |

## Notes

- All resampling uses lowercase pandas frequency strings (`1h`, `1d`) for pandas 2.0+ compatibility.
- `simulate_trading_with_hit_rate` bootstraps from actual return distributions rather than assuming normality — preserves fat tails and skew.
- The module logs to stdout at `INFO` level. To suppress: `logging.getLogger('financial_trading_utils').setLevel(logging.WARNING)`.