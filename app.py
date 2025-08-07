# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="CryptoSentry AI", layout="wide")

# ---------------- SESSION STATE LOGIN ---------------- #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔐 CryptoSentry Login")
    with st.form("Login Form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            st.session_state.logged_in = True
            st.rerun()

if not st.session_state.logged_in:
    login_page()
    st.stop()

# ---------------- LOAD DATA ---------------- #
data_dir = "data_outputs"
fig_dir = "figures"

try:
    news = pd.read_csv(os.path.join(data_dir, "news_with_sentiment.csv"))
    weights = pd.read_csv(os.path.join(data_dir, "portfolio_weights.csv"))
    features = pd.read_csv(os.path.join(data_dir, "token_features.csv"))
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# Ensure datetime columns
weights['time'] = pd.to_datetime(weights['time'])
features['time'] = pd.to_datetime(features['time'])
news['published_on'] = pd.to_datetime(news['published_on'], errors='coerce')

# ---------------- SIDEBAR NAVIGATION ---------------- #
with st.sidebar:
    st.title("📊 Navigation")
    selection = st.radio("Go to:", ["Portfolio Dashboard", "Market Signals", "News Sentiment"])

# ---------------- PORTFOLIO DASHBOARD ---------------- #
if selection == "Portfolio Dashboard":
    st.title("📈 Portfolio Dashboard")

    latest_date = weights['time'].max()
    latest_snapshot = weights[weights['time'] == latest_date]

    st.subheader(f"Portfolio Allocation on {latest_date.date()}")
    fig_pie = px.pie(latest_snapshot, values='weight', names='symbol', hole=0.3)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Weight Distribution Over Time")
    pivot = weights.pivot(index='time', columns='symbol', values='weight').fillna(0)
    fig_area = px.area(pivot, x=pivot.index, y=pivot.columns, title="Token Weights Over Time")
    st.plotly_chart(fig_area, use_container_width=True)

# ---------------- MARKET SIGNALS ---------------- #
elif selection == "Market Signals":
    st.title("📊 Market Signals")

    st.subheader("Sentiment Momentum vs Return")
    scatter_fig = px.scatter(
        features,
        x="sentiment_momentum",
        y="return",
        color="symbol",
        title="Sentiment Momentum vs Return",
        opacity=0.7
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.subheader("Feature Correlation Heatmap")
    corr = features[["return", "volatility", "momentum", "tssi", "sentiment_momentum"]].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------- NEWS SENTIMENT ---------------- #
elif selection == "News Sentiment":
    st.title("📰 News Article Sentiment")

    st.subheader("Random Sample of Articles")
    st.dataframe(news[["published_on", "title", "text", "sentiment"]].sample(5))

    st.subheader("Sentiment Distribution by Token")
    # Infer tokens from categories field where possible
    token_labels = ["BTC", "ETH", "XRP", "ADA", "DOGE", "SOL"]
    def extract_token(cat):
        if isinstance(cat, str):
            for t in token_labels:
                if t in cat:
                    return t
        return "OTHER"
    news["symbol"] = news["categories"].apply(extract_token)

    sentiment_counts = news.groupby("symbol")["sentiment"].count().reset_index().rename(columns={"sentiment": "count"})
    fig_sent = px.bar(sentiment_counts, x="symbol", y="count", color="symbol", title="Sentiment Article Counts by Token")
    st.plotly_chart(fig_sent, use_container_width=True)

    st.subheader("Article Volume Over Time")
    daily_volume = news.groupby(news["published_on"].dt.date).size().reset_index(name="count")
    fig_vol = px.line(daily_volume, x="published_on", y="count", title="Article Volume Over Time")
    st.plotly_chart(fig_vol, use_container_width=True)
