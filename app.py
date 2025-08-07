# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# ----------------- CONFIG -----------------
DATA_DIR = "data_outputs"
FIGURE_DIR_2 = "figures/station2"
FIGURE_DIR_3 = "figures/station3"

# ----------------- LOAD DATA -----------------
@st.cache_data
def load_data():
    prices = pd.read_csv(f"{DATA_DIR}/crypto_prices.csv", parse_dates=["time"])
    news = pd.read_csv(f"{DATA_DIR}/news_with_sentiment.csv", parse_dates=["published_on"])
    features = pd.read_csv(f"{DATA_DIR}/token_features.csv", parse_dates=["time"])
    weights = pd.read_csv(f"{DATA_DIR}/portfolio_weights.csv", parse_dates=["time"])
    return prices, news, features, weights

prices, news, features, weights = load_data()

# ----------------- SIDEBAR NAV -----------------
st.sidebar.title("CryptoSentry Navigation")
page = st.sidebar.radio("Go to", [
    "📌 About & Onboarding",
    "📊 Portfolio Dashboard",
    "🧠 Sentiment Analysis",
    "📈 Feature Correlation",
    "⚙️ Rebalancing Snapshots"
])

# ----------------- PAGE 1: ABOUT -----------------
if page == "📌 About & Onboarding":
    st.title("🚀 CryptoSentry AI Dashboard")
    st.markdown("""
    Welcome to CryptoSentry – your intelligent crypto portfolio assistant.

    **This prototype** shows how real-time price and news sentiment data can be used to:
    - Visualise market signals
    - Forecast token returns
    - Suggest optimised rebalancing

    ---
    🔎 Explore the sidebar to:
    - View sentiment data
    - Examine model features
    - Track historical allocations
    """)

    st.subheader("🧪 Example Article")
    st.dataframe(news[["published_on", "title", "symbol", "sentiment"]].sample(1))

# ----------------- PAGE 2: PORTFOLIO -----------------
elif page == "📊 Portfolio Dashboard":
    st.title("💼 Portfolio Allocation Over Time")

    st.markdown("#### Portfolio Weights (Stacked View)")
    st.image(f"{FIGURE_DIR_3}/weights_over_time.png", use_column_width=True)

    st.markdown("#### Token Allocations at Each Rebalance")
    for t in weights["time"].dt.date.unique():
        st.image(f"{FIGURE_DIR_3}/allocation_snapshot_{t}.png", caption=f"Token Weights on {t}")

# ----------------- PAGE 3: SENTIMENT -----------------
elif page == "🧠 Sentiment Analysis":
    st.title("📉 Sentiment Insights")

    st.markdown("#### Total Article Volume Over Time")
    st.image(f"{FIGURE_DIR_2}/article_volume_over_time.png", use_column_width=True)

    st.markdown("#### Sentiment vs Return")
    st.image(f"{FIGURE_DIR_2}/sentiment_vs_return.png", use_column_width=True)

    st.markdown("#### Sample Articles with High Sentiment")
    high_sentiment = news[news["sentiment"] > 0.9].sort_values("sentiment", ascending=False)
    st.dataframe(high_sentiment[["published_on", "title", "symbol", "sentiment"]].head(5))

# ----------------- PAGE 4: FEATURES -----------------
elif page == "📈 Feature Correlation":
    st.title("📊 Token Feature Analysis")

    st.markdown("#### Feature Correlation Heatmap")
    st.image(f"{FIGURE_DIR_2}/feature_correlation_heatmap.png", use_column_width=False)

    st.markdown("#### Explore Token Statistics")
    token = st.selectbox("Select token:", features["symbol"].unique())
    st.dataframe(features[features["symbol"] == token].head(10))

# ----------------- PAGE 5: REBALANCING -----------------
elif page == "⚙️ Rebalancing Snapshots":
    st.title("🔁 Portfolio Rebalancing Events")

    for t in weights["time"].dt.date.unique():
        st.markdown(f"### Rebalance Date: {t}")
        snapshot = weights[weights["time"].dt.date == t].sort_values("weight", ascending=False)
        st.dataframe(snapshot)
