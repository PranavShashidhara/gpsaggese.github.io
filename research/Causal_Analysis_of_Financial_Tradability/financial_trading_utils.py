"""
Import as:

import research.Causal_Analysis_of_Financial_Tradability.financial_trading_utils as ftu
"""

import dataclasses as dc
import datetime as dt
import logging as lg
import typing as tp
import warnings as w

import numpy as np
import pandas as pd

w.filterwarnings("ignore")

lg.basicConfig(
    level=lg.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = lg.getLogger(__name__)

# =============================================================================
# EXERCISE #1: SIMULATION CONFIGURATION
# =============================================================================

# #############################################################################
# SimulationConfig
# #############################################################################


@dc.dataclass
class SimulationConfig:
    """Configuration for Exercise #1 simulation."""

    asset: str = "BTC"  # Asset symbol
    frequency: str = "1h"  # Time frequency
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    hit_rates: tp.Optional[tp.List[float]] = None  # Hit rates to test
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% per trade
    num_simulations: int = 10000  # Monte Carlo runs
    seed: int = 42

    def __post_init__(self) -> None:
        """Initialize default hit_rates if not provided."""
        if self.hit_rates is None:
            self.hit_rates = list(np.arange(0.45, 0.56, 0.01))  # 45% to 55% in 1% steps


# =============================================================================
# EXERCISE #1: DATA PIPELINE FUNCTIONS
# =============================================================================


def load_exercise_data(config: SimulationConfig) -> pd.DataFrame:
    """
    Load cryptocurrency data at specified frequency.

    Easy to swap: asset, frequency, date range.

    Parameters
    ----------
    config : SimulationConfig
        Configuration with asset, frequency, dates

    Returns
    -------
    pd.DataFrame
        OHLCV data with timestamp index
    """
    logger.info("Loading %s data at %s frequency", config.asset, config.frequency)
    try:
        # Load Bitcoin data from Kaggle
        df = load_kaggle_bitcoin_data(
            start_date=config.start_date, end_date=config.end_date
        )
        # Resample to target frequency
        if config.frequency != "1min":
            df = resample_to_interval(df, config.frequency)
        logger.info("Loaded %d records for %s", len(df), config.asset)
        return df
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Could not load from Kaggle: %s. Using synthetic data.", e)
        df = generate_synthetic_bitcoin_data(
            start_date=config.start_date,
            end_date=config.end_date,
            interval=config.frequency,
        )
        return df


def compute_returns(df: pd.DataFrame) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Compute percentage returns from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with close prices

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (returns array, timestamps array)
    """
    returns = df["close"].pct_change().dropna()
    timestamps = df["timestamp"].iloc[1:].values  # Align with returns
    logger.info("Computed %d returns", len(returns))
    return returns.values, timestamps


# =============================================================================
# EXERCISE #1: CORE SIMULATION ENGINE
# =============================================================================


def simulate_trading_with_hit_rate(
    returns: np.ndarray,
    hit_rate: float,
    num_simulations: int = 10000,
    transaction_cost: float = 0.0015,
    trade_probability: float = 1.0,
    seed: int = 42,
) -> tp.Tuple[np.ndarray, tp.Dict]:
    """
    Simulate trading strategy with given hit rate.

    Parameters
    ----------
    returns : np.ndarray
        Array of market returns
    hit_rate : float
        Probability of correct prediction (0-1)
    num_simulations : int, default=10000
        Number of Monte Carlo simulations
    transaction_cost : float, default=0.0015
        Transaction cost per trade (0.15%)
    trade_probability : float, default=1.0
        Probability of taking a trade
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    Tuple[np.ndarray, Dict]
        PnL simulations and statistics dictionary
    """
    np.random.seed(seed)

    pnl_simulations = np.zeros(num_simulations)
    winning_trades = np.zeros(num_simulations)
    losing_trades = np.zeros(num_simulations)
    max_drawdown_list = np.zeros(num_simulations)

    n_returns = len(returns)

    for sim_idx in range(num_simulations):

        # Sample returns
        sampled_returns = np.random.choice(returns, size=n_returns, replace=True)

        # True direction of returns (+1 or -1)
        true_direction = np.sign(sampled_returns)
        true_direction[true_direction == 0] = 1  # handle zero returns

        # Generate correctness (1 = correct prediction, 0 = wrong)
        is_correct = np.random.binomial(1, hit_rate, n_returns)

        # Predicted direction
        predicted_direction = np.where(
            is_correct == 1, true_direction, -true_direction
        )

        # Trade decision (optional filtering)
        trade_mask = np.random.rand(n_returns) < trade_probability

        # PnL calculation
        pnl_per_bar = trade_mask * (predicted_direction * sampled_returns)

        # Apply transaction cost ONLY when trading
        pnl_per_bar -= trade_mask * transaction_cost

        # Cumulative PnL
        cumulative_pnl = np.cumsum(pnl_per_bar)
        total_pnl = cumulative_pnl[-1]
        pnl_simulations[sim_idx] = total_pnl

        # Stats
        winning_trades[sim_idx] = np.sum(pnl_per_bar > 0)
        losing_trades[sim_idx] = np.sum(pnl_per_bar < 0)

        # Max drawdown
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown_list[sim_idx] = np.min(drawdown)

    # Aggregate statistics
    stats_dict = {
        "hit_rate": hit_rate,
        "mean_pnl": np.mean(pnl_simulations),
        "std_pnl": np.std(pnl_simulations),
        "min_pnl": np.min(pnl_simulations),
        "max_pnl": np.max(pnl_simulations),
        "percentile_5": np.percentile(pnl_simulations, 5),
        "percentile_50": np.percentile(pnl_simulations, 50),
        "percentile_95": np.percentile(pnl_simulations, 95),
        "prob_profit": np.mean(pnl_simulations > 0),
        "sharpe_ratio": (
            np.mean(pnl_simulations) / np.std(pnl_simulations)
            if np.std(pnl_simulations) > 0
            else 0
        ),
        "avg_winning_trades": np.mean(winning_trades),
        "avg_losing_trades": np.mean(losing_trades),
        "avg_max_drawdown": np.mean(max_drawdown_list),
    }

    return pnl_simulations, stats_dict


# =============================================================================
# DATA COLLECTION FUNCTIONS - KAGGLE DATASET
# =============================================================================


def load_kaggle_bitcoin_data(
    start_date: tp.Optional[str] = None, end_date: tp.Optional[str] = None
) -> pd.DataFrame:
    """
    Load Bitcoin historical data from Kaggle dataset.

    Uses exact code from:
    https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data

    Dataset: mczielinski/bitcoin-historical-data
    Data: OHLCV at 1-minute granularity from 2013-2021

    Parameters
    ----------
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format

    Returns
    -------
    pd.DataFrame
        Bitcoin OHLCV data
    """
    try:
        # Install dependencies as needed:
        # pip install kagglehub[pandas-datasets]
        import kagglehub as kh

        logger.info("Loading Kaggle Bitcoin historical data...")
        # Set the path to the file you'd like to load
        file_path = "btcusd_1-min_data.csv"
        # Load the latest version
        df = kh.dataset_download(
            "mczielinski/bitcoin-historical-data",
            file_path,
        )
        logger.info("Raw data shape: %s", df.shape)
        logger.info("Columns: %s", df.columns.tolist())
        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()
        # Handle timestamp column - Kaggle dataset has 'time' column with Unix timestamp
        if "time" in df.columns:
            # Original column in milliseconds
            df["timestamp"] = pd.to_datetime(df["time"], unit="s")
            df = df.drop("time", axis=1)
        elif "timestamp" in df.columns:
            # Ensure numeric timestamps are also treated as ms
            if pd.api.types.is_numeric_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            df["timestamp"] = pd.to_datetime(df.index)
        # Filter by start/end dates
        if start_date:
            df = df[df["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.to_datetime(end_date)]
        # Ensure required columns exist
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error("Dataset missing required columns: %s", missing_cols)
            logger.error("Available columns: %s", df.columns.tolist())
            raise ValueError(f"Dataset must contain: {', '.join(required_cols)}")
        # Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info(
            "Loaded %d records from %s to %s",
            len(df),
            df["timestamp"].min(),
            df["timestamp"].max(),
        )
        return df
    except ImportError as e:
        logger.error(
            "kagglehub not installed. Install with: pip install kagglehub[pandas-datasets]"
        )
        raise ImportError(
            "kagglehub not installed. Install with: pip install kagglehub[pandas-datasets]"
        ) from e
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error loading Kaggle data: %s", e)
        raise


def resample_to_interval(df: pd.DataFrame, interval: str = "1h") -> pd.DataFrame:
    """
    Resample 1-minute Bitcoin data to desired interval.

    Parameters
    ----------
    df : pd.DataFrame
        1-minute OHLCV data with timestamp index
    interval : str, default='1h'
        Target interval ('5min', '15min', '1h', '4h', '1d')

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV data
    """
    df = df.copy()
    df = df.set_index("timestamp")
    # Map intervals to pandas frequency strings (using lowercase)
    freq_map = {
        "5min": "5min",
        "15min": "15min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    freq = freq_map.get(interval, "1h")
    resampled = pd.DataFrame()
    resampled["open"] = df["open"].resample(freq).first()
    resampled["high"] = df["high"].resample(freq).max()
    resampled["low"] = df["low"].resample(freq).min()
    resampled["close"] = df["close"].resample(freq).last()
    resampled["volume"] = df["volume"].resample(freq).sum()
    resampled = resampled.reset_index()
    resampled = resampled.dropna()
    logger.info("Resampled to %s: %d records", interval, len(resampled))
    return resampled


def generate_synthetic_bitcoin_data(
    start_date: tp.Optional[str] = None,
    end_date: tp.Optional[str] = None,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Generate synthetic Bitcoin OHLCV data for testing when Kaggle data unavailable.

    Parameters
    ----------
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    interval : str, default='1h'
        Time interval

    Returns
    -------
    pd.DataFrame
        Synthetic OHLCV data
    """
    start = dt.datetime.strptime(start_date or "2024-02-01", "%Y-%m-%d")
    end = dt.datetime.strptime(end_date or "2024-02-28", "%Y-%m-%d")
    # Map intervals to pandas frequency strings (lowercase for pandas 2.0+)
    freq_map = {
        "1min": "1min",
        "5min": "5min",
        "15min": "15min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    freq = freq_map.get(interval, "1h")
    dates = pd.date_range(start, end, freq=freq)
    np.random.seed(42)
    base_price = 45000  # Bitcoin reference price
    prices = base_price + np.cumsum(np.random.randn(len(dates)) * 100)
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": prices + np.abs(np.random.randn(len(dates)) * 50),
            "low": prices - np.abs(np.random.randn(len(dates)) * 50),
            "close": prices + np.random.randn(len(dates)) * 50,
            "volume": np.random.uniform(1000, 5000, len(dates)),
        }
    )
    logger.info(
        "Generated synthetic Bitcoin data: %d records for %s interval",
        len(df),
        interval,
    )
    return df