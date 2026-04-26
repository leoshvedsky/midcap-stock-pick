"""
features_valuation.py
---------------------
Pulls valuation ratios and analyst signal proxies via yfinance.

Features produced:
  - pe_ratio              — trailing P/E
  - forward_pe            — forward P/E (analyst estimate)
  - ps_ratio              — price-to-sales
  - pb_ratio              — price-to-book
  - ev_to_ebitda          — enterprise value / EBITDA
  - short_ratio           — short interest / avg daily volume (days to cover)
  - institutional_pct     — % shares held by institutions
  - analyst_target_upside — (target price - current) / current

Output:
  - data/processed/features_valuation.csv
"""

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def get_valuation_features(ticker: str) -> dict:
    """Fetch valuation ratios for a single ticker."""
    row = {"ticker": ticker}
    try:
        t = yf.Ticker(ticker)
        info = t.info

        row["pe_ratio"] = info.get("trailingPE", np.nan)
        row["forward_pe"] = info.get("forwardPE", np.nan)
        row["ps_ratio"] = info.get("priceToSalesTrailing12Months", np.nan)
        row["pb_ratio"] = info.get("priceToBook", np.nan)
        row["ev_to_ebitda"] = info.get("enterpriseToEbitda", np.nan)
        row["short_ratio"] = info.get("shortRatio", np.nan)
        row["institutional_pct"] = info.get("heldPercentInstitutions", np.nan)

        # Analyst price target upside
        current = info.get("currentPrice", np.nan)
        target = info.get("targetMeanPrice", np.nan)
        if current and target and current > 0:
            row["analyst_target_upside"] = round((target - current) / current, 4)
        else:
            row["analyst_target_upside"] = np.nan

        row["error"] = None

    except Exception as e:
        row["error"] = str(e)

    return row


def build_valuation_features(universe_path: str = "data/raw/universe.csv") -> pd.DataFrame:
    universe = pd.read_csv(universe_path)
    tickers = universe["ticker"].tolist()

    print(f"Fetching valuation features for {len(tickers)} tickers...")
    results = []
    for ticker in tqdm(tickers):
        results.append(get_valuation_features(ticker))
        time.sleep(0.5)

    df = pd.DataFrame(results)
    errors = df[df["error"].notna()]
    if not errors.empty:
        print(f"  {len(errors)} tickers had errors.")

    df_clean = df[df["error"].isna()].drop(columns=["error"])
    out_path = os.path.join(PROCESSED_DIR, "features_valuation.csv")
    df_clean.to_csv(out_path, index=False)
    print(f"Valuation features saved: {out_path} ({len(df_clean)} stocks)")
    return df_clean


if __name__ == "__main__":
    df = build_valuation_features()
    print(df.head(10).to_string(index=False))
