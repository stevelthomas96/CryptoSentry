# station2_feature_engineering.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# -------------------- Config --------------------
PRICE_FILE = "../CryptoSentry/data_outputs/crypto_prices.csv"
NEWS_FILE = "../CryptoSentry/data_outputs/news_with_sentiment.csv"
OUTPUT_FILE = "../CryptoSentry/data_outputs/token_features.csv"
PLOT_DIR = "../CryptoSentry/figures/station2"
WINDOW_VOL = 7  # days
WINDOW_MOM = 3  # days
EMA_WINDOW = 5

# -------------------- Load Data --------------------
def load_data():
    prices = pd.read_csv(PRICE_FILE, parse_dates=["time"])
    news = pd.read_csv(NEWS_FILE, parse_dates=["published_on"])
    return prices, news

# -------------------- Price Features --------------------
def generate_price_features(df):
    df = df.sort_values("time")
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(WINDOW_VOL).std()
    df["momentum"] = df["close"].pct_change(WINDOW_MOM)
    return df

# -------------------- Sentiment Indexing --------------------
def tag_tokens_from_category(cat):
    if not isinstance(cat, str):
        return []
    return [token for token in ["BTC", "ETH", "ADA", "XRP", "SOL", "DOGE", "DOT", "AVAX", "LINK", "MATIC"] if token in cat.upper()]

def build_sentiment_index(news):
    news = news.copy()
    news["tokens"] = news["categories"].apply(tag_tokens_from_category)
    news = news.explode("tokens").dropna(subset=["tokens"])
    news["date"] = news["published_on"].dt.date

    if "title_sentiment" in news.columns and "text_sentiment" in news.columns:
        news["weighted_sentiment"] = 0.6 * news["title_sentiment"] + 0.4 * news["text_sentiment"]
    else:
        news["weighted_sentiment"] = news["sentiment"]

    daily_sentiment = news.groupby(["tokens", "date"], observed=True)["weighted_sentiment"].mean().reset_index()
    daily_sentiment.rename(columns={"tokens": "symbol", "date": "time", "weighted_sentiment": "tssi"}, inplace=True)
    daily_sentiment["time"] = pd.to_datetime(daily_sentiment["time"])

    std_dev = news.groupby(["tokens", "date"], observed=True)["weighted_sentiment"].std().reset_index()
    std_dev.rename(columns={"weighted_sentiment": "sentiment_disagreement"}, inplace=True)
    std_dev["time"] = pd.to_datetime(std_dev["date"])
    std_dev.drop(columns=["date"], inplace=True)
    std_dev.rename(columns={"tokens": "symbol"}, inplace=True)

    volume = news.groupby(["tokens", "date"], observed=True).size().reset_index(name="article_volume")
    volume.rename(columns={"tokens": "symbol", "date": "time"}, inplace=True)
    volume["time"] = pd.to_datetime(volume["time"])

    tssi_df = pd.merge(daily_sentiment, std_dev, on=["symbol", "time"], how="left")
    tssi_df = pd.merge(tssi_df, volume, on=["symbol", "time"], how="left")
    tssi_df = tssi_df.sort_values(["symbol", "time"])
    tssi_df["sentiment_momentum"] = tssi_df.groupby("symbol")["tssi"].transform(lambda x: x.ewm(span=EMA_WINDOW).mean())
    return tssi_df

def build_market_sentiment_index(news):
    news = news.copy()
    news["date"] = news["published_on"].dt.date
    daily_msi = news.groupby("date")["sentiment"].mean().reset_index()
    daily_msi["time"] = pd.to_datetime(daily_msi["date"])
    daily_msi["msi"] = daily_msi["sentiment"].ewm(span=EMA_WINDOW).mean()
    return daily_msi[["time", "msi"]]

# -------------------- Merge Features --------------------
def merge_features(price_df, senti_df, msi_df):
    full = pd.merge(price_df, senti_df, on=["symbol", "time"], how="left")
    full = pd.merge(full, msi_df, on="time", how="left")
    full["tssi"] = full["tssi"].ffill()
    full["sentiment_momentum"] = full["sentiment_momentum"].ffill()
    full["sentiment_disagreement"] = full["sentiment_disagreement"].ffill()
    full["article_volume"] = full["article_volume"].fillna(0)
    full["msi"] = full["msi"].ffill()
    return full

# -------------------- Visualisation --------------------
def generate_feature_plots(df):
    os.makedirs(PLOT_DIR, exist_ok=True)
    df["time"] = pd.to_datetime(df["time"])
    tokens = df["symbol"].unique()

    for token in tokens:
        subset = df[df["symbol"] == token]
        if subset["tssi"].notna().sum() == 0:
            continue
        plt.figure(figsize=(10, 4))
        plt.plot(subset["time"], subset["tssi"], label="TSSI", color="steelblue")
        plt.title(f"TSSI Over Time – {token}")
        plt.xlabel("Date")
        plt.ylabel("TSSI")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/tssi_{token}.png")
        plt.close()

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="sentiment_momentum", y="return", hue="symbol", alpha=0.6)
    plt.title("Sentiment Momentum vs. Return")
    plt.xlabel("Sentiment Momentum")
    plt.ylabel("Token Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/sentiment_vs_return.png")
    plt.close()

    volume_timeline = df.groupby("time")["article_volume"].sum().reset_index()
    plt.figure(figsize=(10, 4))
    plt.plot(volume_timeline["time"], volume_timeline["article_volume"], color="darkorange")
    plt.title("Total Article Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Articles")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/article_volume_over_time.png")
    plt.close()

    corr_features = ["return", "volatility", "momentum", "tssi", "sentiment_momentum", "msi"]
    corr = df[corr_features].dropna().corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between Token Features")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/feature_correlation_heatmap.png")
    plt.close()

# -------------------- Pipeline --------------------
def run_station2():
    print("📊 Loading data...")
    price_df, news_df = load_data()

    print("⚙️ Generating price-based features...")
    price_df = price_df.groupby("symbol").apply(generate_price_features).reset_index(drop=True)

    print("🧠 Building sentiment indices...")
    sentiment_df = build_sentiment_index(news_df)
    msi_df = build_market_sentiment_index(news_df)

    print("🔗 Merging features...")
    features = merge_features(price_df, sentiment_df, msi_df)

    print("💾 Saving token feature matrix...")
    features.to_csv(OUTPUT_FILE, index=False)

    print("📈 Generating appendix/visualisation plots...")
    generate_feature_plots(features)

    print("✅ Station 2 complete. Outputs saved to:", OUTPUT_FILE, "and", PLOT_DIR)

if __name__ == "__main__":
    run_station2()
