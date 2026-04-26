"""
build_dataset.py
----------------
Merges all feature tables and generates the target label for model training.

Target label:
  - did the stock return > GAIN_THRESHOLD in the next FORWARD_DAYS trading days?
  - Binary: 1 = yes (gained), 0 = no

This requires historical data to generate labels. For current scoring (inference),
labels are not needed — features alone drive the propensity score.

Outputs:
  - data/processed/dataset_train.csv   — labeled, for model training
  - data/processed/dataset_score.csv  — current features, for live scoring
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
RAW_DIR = os.getenv("DATA_DIR", "data/raw")

FORWARD_DAYS = 30       # predict gain over this horizon
GAIN_THRESHOLD = 0.05   # 5% gain threshold for positive label


def load_features() -> pd.DataFrame:
    """Merge all feature CSVs on ticker."""
    price = pd.read_csv(os.path.join(PROCESSED_DIR, "features_price.csv"))
    valuation = pd.read_csv(os.path.join(PROCESSED_DIR, "features_valuation.csv"))
    edgar = pd.read_csv(os.path.join(PROCESSED_DIR, "features_edgar.csv"))

    df = price.merge(valuation, on="ticker", how="left")
    df = df.merge(edgar, on="ticker", how="left")

    # Attach sector from universe
    universe = pd.read_csv(os.path.join(RAW_DIR, "universe.csv"))
    df = df.merge(universe[["ticker", "name", "sector", "market_cap"]], on="ticker", how="left")

    return df


def generate_forward_labels(tickers: list[str]) -> pd.DataFrame:
    """
    For each ticker, compute whether price is >GAIN_THRESHOLD higher
    FORWARD_DAYS trading days from now.

    In a full backtesting setup, this loops over historical windows.
    Here we generate a single forward label from today's price (for inference validation).

    For true training labels, you'd shift this logic to be historical:
      - For date T in history, pull price at T and T+30, compute label
      - Stack these across many historical T values to build training set
    """
    print(f"Generating {FORWARD_DAYS}-day forward return labels...")
    rows = []
    for ticker in tqdm(tickers):
        try:
            hist = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)
            if hist.empty or len(hist) < FORWARD_DAYS + 5:
                continue

            close = hist["Close"].squeeze()

            # Label: did price go up > threshold from 30 days ago to today?
            # (simulates: if we had bought 30 days ago, did we win?)
            price_entry = float(close.iloc[-(FORWARD_DAYS + 1)])
            price_exit = float(close.iloc[-1])
            fwd_return = (price_exit - price_entry) / price_entry
            label = int(fwd_return > GAIN_THRESHOLD)

            rows.append({
                "ticker": ticker,
                "fwd_return_30d": round(fwd_return, 4),
                "label": label,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def build_dataset(generate_labels: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry: merge features, optionally attach labels.

    Returns:
        (train_df, score_df) tuple
        - train_df: labeled dataset for model training
        - score_df: current features for scoring all stocks today
    """
    df = load_features()
    print(f"Merged features: {df.shape[0]} stocks, {df.shape[1]} columns")

    # Score dataset = current snapshot, no labels needed
    score_df = df.copy()
    out_score = os.path.join(PROCESSED_DIR, "dataset_score.csv")
    score_df.to_csv(out_score, index=False)
    print(f"Score dataset saved: {out_score}")

    train_df = pd.DataFrame()
    if generate_labels:
        labels = generate_forward_labels(df["ticker"].tolist())
        train_df = df.merge(labels, on="ticker", how="inner")
        out_train = os.path.join(PROCESSED_DIR, "dataset_train.csv")
        train_df.to_csv(out_train, index=False)
        print(f"Train dataset saved: {out_train} ({len(train_df)} labeled stocks)")
        print(f"  Label distribution: {train_df['label'].value_counts().to_dict()}")

    return train_df, score_df


if __name__ == "__main__":
    train_df, score_df = build_dataset(generate_labels=True)
    print("\nSample (score):")
    print(score_df[["ticker", "price", "return_30d", "rsi_14", "pe_ratio"]].head(10).to_string(index=False))
