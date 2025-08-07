# station3_model.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.covariance import LedoitWolf

# -------------------- CONFIG --------------------
INPUT_FILE = "../CryptoSentry/data_outputs/token_features.csv"
OUTPUT_FILE = "../CryptoSentry/data_outputs/portfolio_weights.csv"
PLOT_DIR = "../CryptoSentry/figures/station3"
REBALANCE_FREQ = 7  # days
RISK_FREE_RATE = 0.0

# -------------------- LOAD FEATURES --------------------
def load_feature_matrix():
    df = pd.read_csv(INPUT_FILE, parse_dates=["time"])
    df = df.dropna(subset=["return", "tssi", "sentiment_momentum", "volatility"])
    return df

# -------------------- FORECAST RETURNS --------------------
def forecast_returns(df):
    df = df.copy()
    df["forecasted_return"] = df["return"] + 0.5 * df["sentiment_momentum"]  # Appendix A.1 formula
    return df

# -------------------- COVARIANCE ESTIMATION --------------------
def estimate_covariance(df):
    pivot = df.pivot(index="time", columns="symbol", values="return").dropna()
    cov_matrix = LedoitWolf().fit(pivot).covariance_
    return cov_matrix, pivot.columns.tolist()

# -------------------- OPTIMISATION --------------------
def mean_variance_optimisation(mu, cov, risk_free=RISK_FREE_RATE):
    inv_cov = np.linalg.pinv(cov)
    ones = np.ones(len(mu))
    w = inv_cov @ mu
    w /= ones @ inv_cov @ mu
    return w

# -------------------- BUILD PORTFOLIO --------------------
def build_portfolio(df):
    output = []
    dates = sorted(df["time"].unique())
    for i in range(0, len(dates), REBALANCE_FREQ):
        period = df[df["time"] <= dates[i]]
        recent = period[period["time"] > dates[i] - timedelta(days=30)]

        if recent["symbol"].nunique() < 2:
            continue

        mu_df = recent.groupby("symbol")["forecasted_return"].mean()
        cov, symbols = estimate_covariance(recent)
        mu = mu_df.reindex(symbols)

        if mu.isnull().any() or (mu.fillna(0) == 0).all():
            continue

        mu = mu.values
        weights = mean_variance_optimisation(mu, cov)

        allocation = pd.DataFrame({
            "time": [dates[i]] * len(symbols),
            "symbol": symbols,
            "weight": weights
        }).dropna(subset=["weight"])
        output.append(allocation)

    return pd.concat(output, ignore_index=True)

# -------------------- VISUALISATION --------------------
def plot_weights(weights_df):
    os.makedirs(PLOT_DIR, exist_ok=True)
    weights_df["time"] = pd.to_datetime(weights_df["time"])

    # Stacked bar chart (valid for negative + positive weights)
    pivot_df = weights_df.pivot(index="time", columns="symbol", values="weight").fillna(0)
    pivot_df.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab10")
    plt.title("Portfolio Weights Over Time")
    plt.ylabel("Weight")
    plt.xlabel("Rebalance Date")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.grid(True, axis="y")
    plt.savefig(f"{PLOT_DIR}/weights_over_time.png")
    plt.close()

    # Allocation snapshot bar chart per rebalance
    for t in weights_df["time"].unique():
        snapshot = weights_df[weights_df["time"] == t].sort_values("weight", ascending=False)
        plt.figure(figsize=(8, 4))
        sns.barplot(data=snapshot, x="symbol", y="weight", palette="Set2")
        plt.title(f"Token Allocation on {t.date()}")
        plt.ylabel("Weight")
        plt.xlabel("Token")
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/allocation_snapshot_{t.date()}.png")
        plt.close()

# -------------------- RUN PIPELINE --------------------
def run_station3():
    print("📈 Loading features...")
    df = load_feature_matrix()

    print("🔮 Forecasting returns...")
    df = forecast_returns(df)

    print("📊 Building portfolio...")
    weights_df = build_portfolio(df)

    print("💾 Saving weights...")
    weights_df.to_csv(OUTPUT_FILE, index=False)

    print("📉 Plotting portfolio allocations...")
    plot_weights(weights_df)

    print("✅ Station 3 complete. Outputs saved to:", OUTPUT_FILE, "and", PLOT_DIR)

if __name__ == "__main__":
    run_station3()
