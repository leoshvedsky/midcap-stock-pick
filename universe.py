"""
universe.py
-----------
Builds the stock universe: mid-cap equities (~$2B-$10B market cap).

Sources:
  - iShares IWR ETF holdings (Russell Mid-Cap) via direct CSV download
  - Fallback: manually curated mid-cap ticker list

Output:
  - data/raw/universe.csv  — ticker, name, sector, market_cap
"""

import os
import time
import requests
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# iShares Russell Mid-Cap ETF holdings (free, no auth required)
IWR_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239718/ishares-russell-mid-cap-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWR_holdings&dataType=fund"
)

MARKET_CAP_MIN = 2_000_000_000   # $2B
MARKET_CAP_MAX = 10_000_000_000  # $10B
MAX_TICKERS = 150                 # cap for runtime; tune as needed

RAW_DIR = os.getenv("DATA_DIR", "data/raw")
os.makedirs(RAW_DIR, exist_ok=True)


def fetch_iwr_holdings() -> pd.DataFrame:
    """Download IWR ETF holdings CSV and return clean ticker list."""
    print("Fetching IWR mid-cap ETF holdings...")
    try:
        resp = requests.get(IWR_HOLDINGS_URL, timeout=30)
        resp.raise_for_status()
        # iShares CSVs have metadata rows at top — skip until header row
        lines = resp.text.splitlines()
        header_idx = next(i for i, l in enumerate(lines) if l.startswith("Ticker"))
        df = pd.read_csv(
            pd.io.common.StringIO("\n".join(lines[header_idx:])),
            thousands=",",
        )
        df = df.rename(columns={"Ticker": "ticker", "Name": "name", "Sector": "sector"})
        df = df[df["ticker"].str.match(r"^[A-Z]{1,5}$", na=False)]
        return df[["ticker", "name", "sector"]].drop_duplicates("ticker")
    except Exception as e:
        print(f"  IWR fetch failed: {e}. Using fallback ticker list.")
        return _fallback_tickers()


def _fallback_tickers() -> pd.DataFrame:
    """Hardcoded mid-cap sample if ETF fetch fails."""
    tickers = [
        ("MTDR", "Matador Resources", "Energy"),
        ("IBOC", "International Bancshares", "Financials"),
        ("AXTA", "Axalta Coating Systems", "Materials"),
        ("CWST", "Casella Waste Systems", "Industrials"),
        ("SFM",  "Sprouts Farmers Market", "Consumer Staples"),
        ("TMHC", "Taylor Morrison Home", "Consumer Discretionary"),
        ("PRGO", "Perrigo Company", "Health Care"),
        ("UFPI", "UFP Industries", "Industrials"),
        ("CATY", "Cathay General Bancorp", "Financials"),
        ("MGEE", "MGE Energy", "Utilities"),
        ("BCPC", "Balchem Corporation", "Materials"),
        ("LBRT", "Liberty Energy", "Energy"),
        ("PLXS", "Plexus Corp", "Technology"),
        ("GRBK", "Green Brick Partners", "Consumer Discretionary"),
        ("ITRI", "Itron Inc", "Technology"),
        ("CNMD", "CONMED Corporation", "Health Care"),
        ("DXPE", "DXP Enterprises", "Industrials"),
        ("FCFS", "FirstCash Holdings", "Financials"),
        ("IOSP", "Innospec Inc", "Materials"),
        ("MATX", "Matson Inc", "Industrials"),
    ]
    return pd.DataFrame(tickers, columns=["ticker", "name", "sector"])


def filter_by_market_cap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hit yfinance to get current market cap and filter to mid-cap range.
    Batches tickers to avoid rate limits.
    """
    print(f"Filtering {len(df)} tickers by market cap (${MARKET_CAP_MIN/1e9:.0f}B–${MARKET_CAP_MAX/1e9:.0f}B)...")
    tickers = df["ticker"].tolist()
    market_caps = {}

    for i in tqdm(range(0, len(tickers), 20), desc="  yfinance batches"):
        batch = tickers[i : i + 20]
        try:
            data = yf.download(
                batch,
                period="1d",
                auto_adjust=True,
                progress=False,
            )
            for t in batch:
                try:
                    info = yf.Ticker(t).fast_info
                    market_caps[t] = getattr(info, "market_cap", None)
                except Exception:
                    market_caps[t] = None
        except Exception:
            for t in batch:
                market_caps[t] = None
        time.sleep(1)  # polite rate limiting

    df["market_cap"] = df["ticker"].map(market_caps)
    df = df.dropna(subset=["market_cap"])
    df = df[
        (df["market_cap"] >= MARKET_CAP_MIN) & (df["market_cap"] <= MARKET_CAP_MAX)
    ]
    return df.head(MAX_TICKERS)


def build_universe(skip_market_cap_filter: bool = False) -> pd.DataFrame:
    """
    Main entry point. Returns and saves universe DataFrame.

    Args:
        skip_market_cap_filter: If True, skips the yfinance cap filter (faster for dev).
    """
    df = fetch_iwr_holdings()

    if not skip_market_cap_filter:
        df = filter_by_market_cap(df)

    out_path = os.path.join(RAW_DIR, "universe.csv")
    df.to_csv(out_path, index=False)
    print(f"Universe saved: {out_path} ({len(df)} stocks)")
    return df


if __name__ == "__main__":
    universe = build_universe(skip_market_cap_filter=False)
    print(universe.head(10).to_string(index=False))
