# Causal Analysis of Financial Tradability

Analysis of cryptocurrency price predictability using causal inference, Monte Carlo simulation, and ML-based classification — applied to Bitcoin (BTC/USD) hourly data.

## Overview

This project investigates whether price movements in crypto markets are predictable enough to trade profitably, accounting for transaction costs, slippage, and market regime. It combines classical technical analysis with causal inference methods and simulation-based PnL attribution.

**Data source:** [Kaggle — Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data) (1-minute OHLCV, 2013–2021). Falls back to synthetic BTC data if Kaggle credentials are unavailable.

## Repository Structure

```
.
├── master.ipynb                      # Main analysis notebook (Exercise #1 entry point)
├── financial_trading_utils.py        # Helper functions (data, features, simulation, analysis)
├── requirements.txt                  # Pinned dependencies (generated from requirements.in)
├── requirements.in                   # High-level dependency specifications
├── Dockerfile                        # Container image definition
├── docker_build.sh                   # Build Docker image
├── docker_jupyter.sh                 # Launch Jupyter in container
├── docker_bash.sh                    # Interactive bash shell in container
├── docker_clean.sh                   # Clean up containers
├── run_jupyter.sh                    # Run Jupyter locally
├── monte_carlo_analysis.png          # Output: Monte Carlo simulation results
├── comprehensive_analysis.png        # Output: Full analysis visualization
└── walk_forward_validation.csv       # Output: Backtesting results
```

## Quick Start

### Setup Dependencies with `uv`

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Compile requirements.in → requirements.txt
uv pip compile requirements.in -o requirements.txt
```

This command:
- Reads high-level specs from `requirements.in`
- Resolves all dependencies
- Pins exact versions to `requirements.txt`
- Docker uses the locked `requirements.txt` for reproducible builds

### Run with Docker

```bash
# Build image
./docker_build.sh

# Launch Jupyter
./docker_jupyter.sh
```

Then open `master.ipynb` and run cells in order.

## Exercise #1 — Hit Rate → PnL Framework

The main notebook exercise sweeps hit rates from 45% to 80% and runs 10,000 Monte Carlo simulations per rate using bootstrapped BTC hourly returns.

### Key Outputs

- **Breakeven hit rate** given transaction costs
- **PnL distribution** (mean, std, 5th/95th percentiles) per hit rate
- **Probability of positive PnL** across the sweep
- **Sensitivity analysis**: Sharpe ratio and max drawdown vs. hit rate

### Parameters

| Parameter | Default | Notes |
|---|---|---|
| `transaction_cost` | 0.15% per trade | commission + slippage combined |
| `num_simulations` | 10,000 | Monte Carlo runs |
| `hit_rates` | 0.45–0.55 | swept in 1% steps |
| `frequency` | `1h` | hourly resampled from 1-min data |

## How to Run

1. Open `master.ipynb`
2. The notebook imports `financial_trading_utils` as `ftu` and uses `%autoreload` for live edits
3. Run cells in order; outputs are saved as PNG and CSV files
4. Review `monte_carlo_analysis.png` and `comprehensive_analysis.png` for results

## Notes

- Docker setup reference: [Project template README](/class_project/project_template/README.md)
- All utility functions are self-contained in `financial_trading_utils.py`
- Notebook is the single entry point for analysis