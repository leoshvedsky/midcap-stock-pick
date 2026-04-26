"""
run_pipeline.py
---------------
End-to-end runner. Execute this to go from nothing → ranked stock list.

Usage:
  python run_pipeline.py                  # full pipeline
  python run_pipeline.py --skip-universe  # reuse existing universe.csv
  python run_pipeline.py --score-only     # skip training, just score
  python run_pipeline.py --dev            # fast dev mode (small universe, no market cap filter)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.universe import build_universe
from src.features_price import build_price_features
from src.features_valuation import build_valuation_features
from src.features_edgar import build_edgar_features
from src.build_dataset import build_dataset
from src.model import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Stock Alpha Ranker Pipeline")
    parser.add_argument("--skip-universe", action="store_true", help="Reuse existing universe.csv")
    parser.add_argument("--score-only", action="store_true", help="Skip model training, score only")
    parser.add_argument("--dev", action="store_true", help="Dev mode: fast, small universe")
    args = parser.parse_args()

    print("=" * 60)
    print("  STOCK ALPHA RANKER")
    print("=" * 60)

    # Step 1: Build universe
    if not args.skip_universe:
        print("\n[1/5] Building universe...")
        build_universe(skip_market_cap_filter=args.dev)
    else:
        print("\n[1/5] Skipping universe (using existing).")

    # Step 2: Pull price features
    print("\n[2/5] Building price features...")
    build_price_features()

    # Step 3: Pull valuation features
    print("\n[3/5] Building valuation features...")
    build_valuation_features()

    # Step 4: Pull EDGAR features
    print("\n[4/5] Building EDGAR features...")
    build_edgar_features()

    # Step 5: Build dataset and train
    print("\n[5/5] Building dataset and training model...")
    build_dataset(generate_labels=not args.score_only)

    if not args.score_only:
        run_pipeline()
    else:
        print("Score-only mode: skipping training. Load existing model to score.")

    print("\n✓ Pipeline complete. Results in data/processed/ranked_stocks.csv")
    print("  Run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
