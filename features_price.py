"""
features_price.py
-----------------
Pulls historical price data and computes technical features for each ticker.

Features produced:
  - return_30d, return_60d, return_90d       — momentum
  - rsi_14                                    — overbought/oversold
  - volume_spike_ratio                        — recent volume vs 90d avg
  - dist_from_52w_high, dist_from_52w_low    — range position
  - realized_vol_30d                          — recent volatility
  - beta_1y                                   — market sensitivity
  - ma_cross_signal                           — golden/death cross flag
  - price                                     — current price

Output:
  - data/processed/features_price.csv
"""

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

SP500_TICKER = "^GSPC"
HISTORY_PERIOD = "2y"


def compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """OLS beta of stock vs market."""
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return cov[0, 1] / cov[1, 1]


def compute_ticker_features(ticker: str, market_returns: pd.Series) -> dict:
    """Download price history for one ticker and return feature dict."""
    try:
        df = yf.download(
            ticker,
            period=HISTORY_PERIOD,
            auto_adjust=True,
            progress=False,
        )
        if df.empty or len(df) < 90:
            return {"ticker": ticker, "error": "insufficient_data"}

        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        # --- Momentum ---
        ret = lambda n: (close.iloc[-1] - close.iloc[-n]) / close.iloc[-n]
        return_30d = ret(30)
        return_60d = ret(60)
        return_90d = ret(90)

        # --- RSI ---
        rsi = RSIIndicator(close=close, window=14).rsi().iloc[-1]

        # --- Volume spike ---
        vol_recent = volume.iloc[-5:].mean()
        vol_baseline = volume.iloc[-90:].mean()
        volume_spike_ratio = vol_recent / vol_baseline if vol_baseline > 0 else np.nan

        # --- 52-week range ---
        high_52w = close.iloc[-252:].max()
        low_52w = close.iloc[-252:].min()
        price = close.iloc[-1]
        dist_from_52w_high = (price - high_52w) / high_52w  # negative = below high
        dist_from_52w_low = (price - low_52w) / low_52w

        # --- Realized volatility ---
        daily_returns = close.pct_change().dropna()
        realized_vol_30d = daily_returns.iloc[-30:].std() * np.sqrt(252)

        # --- Beta ---
        beta_1y = compute_beta(daily_returns.iloc[-252:], market_returns)

        # --- MA cross signal: +1 golden cross, -1 death cross, 0 neutral ---
        sma_50 = SMAIndicator(close=close, window=50).sma_indicator()
        sma_200 = SMAIndicator(close=close, window=200).sma_indicator()
        if sma_50.iloc[-1] > sma_200.iloc[-1] and sma_50.iloc[-2] <= sma_200.iloc[-2]:
            ma_cross_signal = 1
        elif sma_50.iloc[-1] < sma_200.iloc[-1] and sma_50.iloc[-2] >= sma_200.iloc[-2]:
            ma_cross_signal = -1
        else:
            ma_cross_signal = 0

        return {
            "ticker": ticker,
            "price": round(float(price), 2),
            "return_30d": round(float(return_30d), 4),
            "return_60d": round(float(return_60d), 4),
            "return_90d": round(float(return_90d), 4),
            "rsi_14": round(float(rsi), 2),
            "volume_spike_ratio": round(float(volume_spike_ratio), 4),
            "dist_from_52w_high": round(float(dist_from_52w_high), 4),
            "dist_from_52w_low": round(float(dist_from_52w_low), 4),
            "realized_vol_30d": round(float(realized_vol_30d), 4),
            "beta_1y": round(float(beta_1y), 4),
            "ma_cross_signal": int(ma_cross_signal),
            "error": None,
        }

    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def build_price_features(universe_path: str = "data/raw/universe.csv") -> pd.DataFrame:
    """
    Compute price features for all tickers in universe.

    Args:
        universe_path: Path to universe.csv from universe.py
    """
    universe = pd.read_csv(universe_path)
    tickers = universe["ticker"].tolist()

    print(f"Downloading market returns ({SP500_TICKER})...")
    sp500 = yf.download(SP500_TICKER, period=HISTORY_PERIOD, auto_adjust=True, progress=False)
    market_returns = sp500["Close"].squeeze().pct_change().dropna()

    print(f"Computing price features for {len(tickers)} tickers...")
    results = []
    for ticker in tqdm(tickers):
        result = compute_ticker_features(ticker, market_returns)
        results.append(result)
        time.sleep(0.3)  # rate limiting

    df = pd.DataFrame(results)

    errors = df[df["error"].notna()]
    if not errors.empty:
        print(f"  {len(errors)} tickers had errors: {errors['ticker'].tolist()}")

    df_clean = df[df["error"].isna()].drop(columns=["error"])
    out_path = os.path.join(PROCESSED_DIR, "features_price.csv")
    df_clean.to_csv(out_path, index=False)
    print(f"Price features saved: {out_path} ({len(df_clean)} stocks)")
    return df_clean


if __name__ == "__main__":
    df = build_price_features()
    print(df.head(10).to_string(index=False))
