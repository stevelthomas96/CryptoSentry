# station1_data_pipeline.py

import os
import pandas as pd
import requests
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from vader_custom_lexicon import custom_lexicon

# -------------------- Configuration --------------------
API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", "REPLACE_ME")
TOKENS = ["BTC", "ETH", "ADA", "XRP", "SOL", "DOGE", "DOT", "AVAX", "LINK", "MATIC"]
FIAT = "USD"
DAYS = 30
NEWS_LIMIT = 100
OUTPUT_DIR = "./data_outputs"

analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(custom_lexicon)

# -------------------- Price Data --------------------
def fetch_price_data(token: str, limit: int = DAYS):
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": token,
        "tsym": FIAT,
        "limit": limit,
        "api_key": API_KEY
    }
    res = requests.get(url, params=params)
    res.raise_for_status()
    data = res.json()["Data"]["Data"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df["symbol"] = token
    return df

# -------------------- News Data --------------------
def fetch_news():
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={API_KEY}"
    res = requests.get(url)
    res.raise_for_status()
    articles = res.json()["Data"][:NEWS_LIMIT]
    return pd.DataFrame(articles)

def clean_news(df):
    df = df.drop_duplicates(subset="id")
    df["published_on"] = pd.to_datetime(df["published_on"], unit="s")
    df.rename(columns={"category": "categories"}, inplace=True)
    df["body"] = df["body"].fillna("")
    df["title"] = df["title"].fillna("")
    df["text"] = df["title"] + ". " + df["body"]
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df

def apply_sentiment(df):
    df = df.copy()
    df["title_sentiment"] = df["title"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["text_sentiment"] = df["body"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["sentiment"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    return df

# -------------------- Pipeline --------------------
def run_station1():
    print("📉 Fetching price data...")
    price_dfs = [fetch_price_data(token) for token in TOKENS]
    all_prices = pd.concat(price_dfs, ignore_index=True)

    print("📰 Fetching news articles...")
    news = fetch_news()
    news_clean = clean_news(news)
    news_scored = apply_sentiment(news_clean)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_prices.to_csv(f"{OUTPUT_DIR}/crypto_prices.csv", index=False)
    news_scored.to_csv(f"{OUTPUT_DIR}/news_with_sentiment.csv", index=False)

    print("✅ Station 1 complete. Data saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    run_station1()
