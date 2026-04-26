"""
model.py
--------
Trains an XGBoost binary classifier to predict short-term upside probability.
Produces propensity scores (0–1) for each stock in the scoring dataset.

Pipeline:
  1. Load labeled training data
  2. Impute missing values, encode categoricals
  3. Train logistic regression baseline + XGBoost
  4. Score current universe → ranked output

Output:
  - models/xgb_model.joblib
  - data/processed/ranked_stocks.csv   ← main output for dashboard
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Features used in model — order matters for reproducibility
FEATURE_COLS = [
    # Price / momentum
    "return_30d", "return_60d", "return_90d",
    "rsi_14", "volume_spike_ratio",
    "dist_from_52w_high", "dist_from_52w_low",
    "realized_vol_30d", "beta_1y", "ma_cross_signal",
    # Valuation
    "pe_ratio", "forward_pe", "ps_ratio", "pb_ratio",
    "ev_to_ebitda", "short_ratio", "institutional_pct",
    "analyst_target_upside",
    # Fundamentals (EDGAR)
    "revenue_growth_yoy", "gross_margin", "eps_surprise",
    "days_since_earnings", "earnings_within_30d",
    "insider_buy_count_90d", "insider_net_sentiment",
]

TARGET_COL = "label"


def load_data():
    train_path = os.path.join(PROCESSED_DIR, "dataset_train.csv")
    score_path = os.path.join(PROCESSED_DIR, "dataset_score.csv")

    train = pd.read_csv(train_path)
    score = pd.read_csv(score_path)

    return train, score


def build_preprocessor():
    """Median imputation — robust to financial NaNs."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])


def train_baseline(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Logistic regression baseline."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, C=0.1)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"  Baseline LR  — CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
    """XGBoost classifier, CPU-friendly settings."""
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_train)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="auc",
        tree_method="hist",   # CPU-optimized
        random_state=42,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_imputed, y_train, cv=cv, scoring="roc_auc")
    print(f"  XGBoost      — CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    model.fit(X_imputed, y_train)
    return model, imputer


def feature_importance_report(model: xgb.XGBClassifier, feature_names: list) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nTop 10 features:")
    print(imp.head(10).to_string(index=False))
    return imp


def score_and_rank(
    model: xgb.XGBClassifier,
    imputer: SimpleImputer,
    score_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply model to current universe and produce ranked output."""
    available = [c for c in FEATURE_COLS if c in score_df.columns]
    X_score = score_df[available].values
    X_score_imputed = imputer.transform(X_score)

    probs = model.predict_proba(X_score_imputed)[:, 1]
    score_df = score_df.copy()
    score_df["propensity_score"] = probs.round(4)
    score_df["rank"] = score_df["propensity_score"].rank(ascending=False).astype(int)

    # Risk flag: high vol + high beta + earnings imminent
    score_df["risk_flag"] = (
        (score_df.get("realized_vol_30d", 0) > 0.40) |
        (score_df.get("beta_1y", 1) > 1.8) |
        (score_df.get("earnings_within_30d", 0) == 1)
    ).astype(int)

    output_cols = [
        "rank", "ticker", "name", "sector", "price", "market_cap",
        "propensity_score",
        "return_30d", "rsi_14", "analyst_target_upside",
        "revenue_growth_yoy", "pe_ratio", "insider_net_sentiment",
        "earnings_within_30d", "risk_flag",
    ]
    output_cols = [c for c in output_cols if c in score_df.columns]
    ranked = score_df[output_cols].sort_values("rank")

    out_path = os.path.join(PROCESSED_DIR, "ranked_stocks.csv")
    ranked.to_csv(out_path, index=False)
    print(f"\nRanked stocks saved: {out_path}")
    return ranked


def run_pipeline():
    print("Loading data...")
    train_df, score_df = load_data()

    # Guard: need at least 2 classes to train
    if train_df[TARGET_COL].nunique() < 2:
        print("Warning: Only one class in training labels. Cannot train. Check label generation.")
        return

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    print(f"Training on {len(train_df)} stocks, {len(FEATURE_COLS)} features")
    print(f"  Positive rate: {y_train.mean():.1%}")

    print("\nTraining models...")
    _ = train_baseline(X_train, y_train)
    xgb_model, imputer = train_xgboost(X_train, y_train)

    feature_importance_report(xgb_model, FEATURE_COLS)

    # Save model
    joblib.dump({"model": xgb_model, "imputer": imputer, "features": FEATURE_COLS},
                os.path.join(MODEL_DIR, "xgb_model.joblib"))
    print(f"\nModel saved: {MODEL_DIR}/xgb_model.joblib")

    print("\nScoring current universe...")
    ranked = score_and_rank(xgb_model, imputer, score_df)
    print("\nTop 10 stocks by propensity score:")
    print(ranked.head(10)[["rank", "ticker", "propensity_score", "price", "risk_flag"]].to_string(index=False))

    return ranked


if __name__ == "__main__":
    run_pipeline()
