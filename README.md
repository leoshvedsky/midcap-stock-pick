# Stock Alpha Ranker

A mid-cap stock screening and ranking tool that assigns a **propensity score** (0–1) to each stock based on its probability of a short-term gain (>5% in 30 days).

Built with: `yfinance` · `SEC EDGAR` · `XGBoost` · `Streamlit`

---

## What it does

1. **Universe selection** — pulls mid-cap stocks from the iShares IWR ETF (~$2B–$10B market cap)
2. **Feature engineering** — price momentum, technical indicators, valuation ratios, EDGAR earnings, insider sentiment
3. **Model training** — XGBoost binary classifier trained on 30-day forward returns
4. **Ranking** — scores and ranks all current stocks by upside probability
5. **Dashboard** — Streamlit UI with filters, charts, and risk flags

---

## Setup

```bash
git clone <your-repo>
cd stock-alpha-ranker

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — add your EDGAR_USER_AGENT (just your name + email, free)
```

---

## Run

### Full pipeline (first time)
```bash
python run_pipeline.py
```

### Dev mode (fast, skips market cap filter)
```bash
python run_pipeline.py --dev
```

### Reuse existing universe, re-pull features + retrain
```bash
python run_pipeline.py --skip-universe
```

### Dashboard
```bash
streamlit run dashboard/app.py
```

---

## Project structure

```
stock-alpha-ranker/
├── src/
│   ├── universe.py           # Stock universe selection (IWR mid-cap ETF)
│   ├── features_price.py     # Price + technical features (yfinance)
│   ├── features_valuation.py # Valuation ratios (yfinance)
│   ├── features_edgar.py     # Earnings + insider data (SEC EDGAR)
│   ├── build_dataset.py      # Merge features + generate labels
│   └── model.py              # XGBoost training + scoring + ranking
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── data/
│   ├── raw/                  # universe.csv
│   └── processed/            # feature CSVs, ranked_stocks.csv
├── models/                   # saved model artifacts
├── notebooks/                # EDA (add your own)
├── run_pipeline.py           # End-to-end runner
├── requirements.txt
└── .env.example
```

---

## Features used in model

| Group | Features |
|---|---|
| Price / momentum | 30/60/90d return, RSI, volume spike, 52w range position |
| Volatility | Realized vol (30d), beta vs S&P 500, MA cross signal |
| Valuation | P/E, Forward P/E, P/S, P/B, EV/EBITDA, short ratio |
| Analyst | Price target upside, institutional ownership % |
| Fundamentals | Revenue growth YoY, gross margin, EPS surprise |
| Insider | Buy count 90d, net insider sentiment score |
| Earnings timing | Days since last report, earnings within 30d flag |

---

## Data sources (all free)

| Data | Source |
|---|---|
| Price history, ratios | `yfinance` |
| Earnings, insider filings | SEC EDGAR (free API, requires User-Agent) |
| Stock universe | iShares IWR ETF holdings (free CSV) |
| Macro context (optional) | FRED API (free key) |

---

## Notes

- **Not investment advice.** Propensity scores are model outputs, not financial recommendations.
- EDGAR requests require a valid `User-Agent` header (your name + email). Set this in `.env`.
- The label generation uses a trailing 30-day window as a proxy. For production backtesting, you would generate labels across many historical time windows (rolling forward).
- CPU-only — no GPU required. Full pipeline runs in ~20–40 min depending on universe size.
