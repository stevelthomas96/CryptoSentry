# app.py

import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# --------------- CONFIG ----------------
DATA_DIR = "data_outputs"
FIGURE_DIR = "figures"
TOKENS = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "DOT", "LINK", "XRP", "MATIC", "ADA"]
# ----------------- LOGIN -------------------
def login_page():
    st.set_page_config(page_title="CryptoSentry Login", layout="centered")
    st.title("🔐 CryptoSentry Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if username.strip() and password.strip():
                st.session_state["authenticated"] = True
                st.experimental_rerun()
            else:
                st.error("Username and password cannot be empty.")

# ------------------ CHECK LOGIN --------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_page()
    st.stop()
# --------------- PAGE SETUP ------------
st.set_page_config(page_title="CryptoSentry", layout="wide")
st.title("📈 CryptoSentry AI Dashboard")
st.markdown("""
Welcome to **CryptoSentry** – your intelligent crypto portfolio assistant.

This prototype shows how real-time price and news sentiment data can be used to:
- Visualise market signals  
- Forecast token returns  
- Suggest optimised rebalancing  
""")

# --------------- LOAD DATA ----------------
@st.cache_data
def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    return pd.read_csv(path, parse_dates=["time"], low_memory=False)

@st.cache_data
def load_news():
    df = pd.read_csv(os.path.join(DATA_DIR, "news_with_sentiment.csv"), parse_dates=["published_on"], low_memory=False)
    df["categories"] = df["categories"].astype(str)

    # Extract token symbol from 'categories'
    def extract_token(categories_str):
        for token in TOKENS:
            if token in categories_str.split("|"):
                return token
        return "OTHER"

    df["symbol"] = df["categories"].apply(extract_token)
    return df

# --------------- SIDEBAR NAVIGATION ------------
page = st.sidebar.radio("📂 Explore the data:", ["Sentiment Articles", "Model Features", "Portfolio Dashboard"])

# --------------- PAGE 1: Sentiment Articles ------------
if page == "Sentiment Articles":
    st.subheader("🗞️ News Article Sentiment")
    news = load_news()

    with st.expander("Random Sample of Articles"):
        st.dataframe(news[["published_on", "title", "symbol", "sentiment"]].sample(5).reset_index(drop=True))

    st.subheader("📊 Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=news, x="sentiment", hue="symbol", multiple="stack", bins=30, palette="tab10", edgecolor=None)
    st.pyplot(fig)

# --------------- PAGE 2: Model Features ------------
elif page == "Model Features":
    st.subheader("📈 Token Features and Correlations")
    df = load_csv("token_features.csv")

    with st.expander("Preview of Engineered Features"):
        st.dataframe(df.head())

    st.subheader("🔁 Feature Correlation Matrix")
    numeric_features = df[["return", "volatility", "momentum", "tssi", "sentiment_momentum"]]
    corr = numeric_features.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(fig)

# --------------- PAGE 3: Portfolio Dashboard ------------
elif page == "Portfolio Dashboard":
    st.subheader("📊 Portfolio Allocation Weights")
    weights = load_csv("portfolio_weights.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = weights.pivot(index="time", columns="symbol", values="weight").fillna(0)
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_ylabel("Weight")
    ax.set_title("Portfolio Weights Over Time")
    st.pyplot(fig)

    st.subheader("🔍 Snapshot Allocation")
    selected_date = st.selectbox("Choose Rebalance Date:", sorted(weights["time"].unique(), reverse=True))
    snapshot = weights[weights["time"] == selected_date].sort_values("weight", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=snapshot, x="symbol", y="weight", palette="Set2")
    ax.set_title(f"Token Allocation on {pd.to_datetime(selected_date).date()}")
    st.pyplot(fig)
