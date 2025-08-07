import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from PIL import Image
import plotly.figure_factory as ff

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="CryptoSentry Portfolio", layout="wide")
DATA_PATH = "data_outputs/portfolio_weights.csv"

# Mock token market caps
LARGE_CAP = {"BTC", "ETH"}
MID_CAP = {"DOT", "MATIC", "AVAX"}

# ---------------- LOAD DATA ---------------- #
@st.cache_data
def load_portfolio_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df = df[df["weight"].notna()]
    return df

# ---------------- RISK PROFILE ---------------- #
def compute_risk_profile(df):
    latest = df[df["time"] == df["time"].max()].copy()  # FIXED
    def cap_group(symbol):
        if symbol in LARGE_CAP:
            return "Large-cap"
        elif symbol in MID_CAP:
            return "Mid-cap"
        else:
            return "Small-cap"
    latest["cap"] = latest["symbol"].apply(cap_group)
    profile = latest.groupby("cap")["weight"].sum().reset_index()
    return profile

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.title("📊 Navigation")
    selection = st.radio("Go to:", ["Portfolio Overview", "Market Signals", "News Sentiment"])
    

# Remove default padding
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Load logo
logo = Image.open("cryptosentrylogo.png")

# Adjusted column layout to compensate for sidebar
col1, col2, col3 = st.columns([1.4, 1, 1])  # Custom offset for better centering
with col2:
    st.image(logo, width=300)

# ---------------- MAIN: PORTFOLIO ---------------- #
if selection == "Portfolio Overview":
    portfolio_df = load_portfolio_data()
    latest_date = portfolio_df["time"].max()
    latest_df = portfolio_df[portfolio_df["time"] == latest_date]
    risk_df = compute_risk_profile(portfolio_df)

    # ---------- ROW 1: Summary + Pie Chart ---------- #
    row1_col1, row1_col2 = st.columns([1, 2])

    with row1_col1:
        with st.container(border=True):
            st.markdown("### 💰 Portfolio Summary")
            st.metric("Net Worth", "$102,365")
            st.metric("Total Assets", "$102,365")
            st.metric("Total Liabilities", "$0")
            st.metric("Claimable Rewards", "$0")
            st.markdown("---")
            st.markdown("### ⚖️ Risk Profile")
            for _, row in risk_df.iterrows():
                label = row["cap"]
                weight = round(row["weight"] * 100)
                progress = max(0, min(weight, 100))
                st.progress(progress, text=f"{label}: {weight}%")

    with row1_col2:
        with st.container(border=True):
            st.markdown("### 🧩 Portfolio Allocation")

            # Hard clean to eliminate any undefined entries
            latest_df = latest_df.copy()
            latest_df = latest_df.dropna(subset=["symbol", "weight"])
            latest_df = latest_df[latest_df["symbol"].astype(str).str.strip() != ""]
            latest_df = latest_df[latest_df["symbol"].astype(str).str.lower() != "nan"]
            latest_df = latest_df[latest_df["symbol"].str.lower() != "undefined"]
            
            latest_df["weight"] = pd.to_numeric(latest_df["weight"], errors="coerce")
            latest_df = latest_df[latest_df["weight"] > 0]


            fig = px.pie(
                latest_df,
                names="symbol",
                values="weight",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                plot_bgcolor="#0E0E0E",
                paper_bgcolor="#0E0E0E",
                font=dict(color="#FFD700"),
                title_font=dict(color="#FFD700"),
                margin=dict(t=0, b=9, l=0, r=0)
            )
            
            fig.update_traces(
                textfont=dict(color='#FFD700', size=16), 
                textposition='outside'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div style='height:180px;'></div>", unsafe_allow_html=True)



    # ---------- ROW 2: Weight Dist + Portfolio Value ---------- #
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        with st.container(border=True):
            st.markdown("### 📊 Weight Distribution Over Time")
            pivot = portfolio_df.pivot(index="time", columns="symbol", values="weight").fillna(0)
            fig_area = px.area(
                pivot,
                x=pivot.index,
                y=pivot.columns,
                title="Token Weights Over Time",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_area, use_container_width=True)

    with row2_col2:
        with st.container(border=True):
            st.markdown("### 💹 Portfolio Value Over Time")
            prices_df = pd.read_csv("data_outputs/crypto_prices.csv", parse_dates=["time"])
            weights_df = load_portfolio_data()
            prices_df = prices_df[["time", "symbol", "close"]]
            merged = pd.merge(weights_df, prices_df, on=["time", "symbol"])
            merged["dollar_value"] = merged["weight"] * merged["close"]
            portfolio_value = merged.groupby("time")["dollar_value"].sum().reset_index()

            fig_value = px.line(
                portfolio_value,
                x="time",
                y="dollar_value",
                title="Net Portfolio Value ($)",
                markers=True,
                line_shape="spline",
            )
            st.plotly_chart(fig_value, use_container_width=True)


# ---------------- PLACEHOLDER TABS ---------------- #
elif selection == "Market Signals":
    st.title("📊 Market Signals")
    prices_df = pd.read_csv("data_outputs/crypto_prices.csv", parse_dates=["time"])
    prices_df = prices_df[["time", "symbol", "close"]].dropna()
    prices_df = prices_df[prices_df["symbol"].astype(str).str.lower() != "undefined"]

    # Pivot prices and calculate returns
    price_pivot = prices_df.pivot(index="time", columns="symbol", values="close").sort_index()
    momentum_7d = price_pivot.pct_change(periods=7)
    momentum_30d = price_pivot.pct_change(periods=30)

    momentum_latest = pd.DataFrame({
        "symbol": momentum_7d.columns,
        "7d_return": momentum_7d.iloc[-1].values,
        "30d_return": momentum_30d.iloc[-1].values
    }).dropna()

    # Plot
    st.subheader("📈 Token Momentum (7-Day vs 30-Day Returns)")
    fig_momentum = px.bar(
        momentum_latest.sort_values("7d_return", ascending=False),
        x="symbol",
        y=["7d_return", "30d_return"],
        barmode="group",
        title="Recent Momentum: Short-Term vs Long-Term",
        labels={"value": "Return", "variable": "Period"}
    )
    st.plotly_chart(fig_momentum, use_container_width=True)
    
    # Volatility: 14-day rolling standard deviation
    return_df = price_pivot.pct_change().dropna()
    rolling_volatility = return_df.rolling(window=14).std().dropna()
    volatility_df = rolling_volatility.reset_index().melt(id_vars="time", var_name="symbol", value_name="volatility")

    st.subheader("🌪️ Rolling Volatility (14-Day)")
    fig_volatility = px.line(
        volatility_df,
        x="time",
        y="volatility",
        color="symbol",
        title="Token Volatility Over Time"
    )
    st.plotly_chart(fig_volatility, use_container_width=True)
    
    # Correlation matrix of last 60 days
    correlation_matrix = return_df.iloc[-60:].corr()

    st.subheader("📊 Token Correlation Matrix")

    # Manually scale the figure to improve readability
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",  # Stretch it horizontally
    )

    fig_corr.update_layout(
        width=900,   # Wider plot
        height=900,  # Taller plot
        font=dict(size=12, color="white"),
        margin=dict(t=50, l=100, b=100, r=50),
        title="60-Day Correlation Between Tokens",
        paper_bgcolor="#0E0E0E",
        plot_bgcolor="#0E0E0E"
    )

    st.plotly_chart(fig_corr, use_container_width=False)

elif selection == "News Sentiment":
    import plotly.express as px

    st.title("📰 News Article Sentiment")

    # --- Load Data --- #
    # Token-level sentiment time series
    sentiment_df = pd.read_csv("data_outputs/sentiment_timeseries.csv", parse_dates=["date"])

    # Full article-level sentiment data (for article feed + MSI)
    article_df = pd.read_csv("data_outputs/news_with_sentiment.csv")
    article_df["time"] = pd.to_datetime(article_df["published_on"], errors="coerce")
    article_df["tokens"] = article_df["tags"].str.split("|")
    exploded = article_df.explode("tokens")
    exploded["symbol"] = exploded["tokens"].str.strip().str.upper()
    exploded = exploded[exploded["symbol"].notna() & (exploded["symbol"] != "")]

    # Filter to valid tokens
    VALID_TOKENS = ["BTC", "ETH", "ADA", "XRP", "DOT", "DOGE", "AVAX", "SOL", "MATIC", "LINK"]
    filtered_df = sentiment_df[sentiment_df["symbol"].isin(VALID_TOKENS)]

    # --- 1. Sentiment Trend Explorer --- #
    st.subheader("📈 Token Sentiment Scores Over Time")

    selected_tokens = st.multiselect(
        "Select tokens to view sentiment trends:",
        options=VALID_TOKENS,
        default=["BTC", "ETH", "ADA"]
    )

    filtered_tokens = filtered_df[filtered_df["symbol"].isin(selected_tokens)]

    fig_sentiment = px.line(
        filtered_tokens,
        x="date",
        y="sentiment",
        color="symbol",
        markers=True,
        title="Daily Average Sentiment (TSSI)"
    )
    fig_sentiment.update_layout(
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        yaxis_range=[-1, 1],
        plot_bgcolor="#0E0E0E",
        paper_bgcolor="#0E0E0E",
        font=dict(color="white")
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # --- 2. Sentiment Disagreement (Volatility Proxy) --- #
    st.subheader("🧠 Sentiment Disagreement (News-Based Volatility)")

    disagreement_df = (
        filtered_tokens.groupby("symbol")
        .rolling(window=3, on="date")["sentiment"]
        .std()
        .reset_index()
        .rename(columns={"sentiment": "sentiment_std"})
    )

    fig_disagree = px.line(
        disagreement_df,
        x="date",
        y="sentiment_std",
        color="symbol",
        title="Sentiment Standard Deviation Over Time"
    )
    fig_disagree.update_layout(
        yaxis_title="Std Dev of Sentiment",
        plot_bgcolor="#0E0E0E",
        paper_bgcolor="#0E0E0E",
        font=dict(color="white")
    )
    st.plotly_chart(fig_disagree, use_container_width=True)

    # --- 3. Market Sentiment Index (MSI) --- #
    st.subheader("📉 Market Sentiment Index (MSI)")

    msi_df = (
        sentiment_df.groupby("date")["sentiment"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment": "MSI"})
    )
    msi_df["date"] = pd.to_datetime(msi_df["date"])

    fig_msi = px.line(
        msi_df,
        x="date",
        y="MSI",
        title="Overall Market Sentiment Index (5-Day EMA Approx.)",
        markers=True
    )
    fig_msi.update_layout(
        yaxis_range=[-1, 1],
        plot_bgcolor="#0E0E0E",
        paper_bgcolor="#0E0E0E",
        font=dict(color="white")
    )
    st.plotly_chart(fig_msi, use_container_width=True)

    # --- 4. Sentiment-Tagged News Article Feed --- #
    st.subheader("📰 Recent News Articles")

    num_articles = st.slider("How many articles to display?", min_value=5, max_value=50, value=10)
    latest_articles = article_df.sort_values("published_on", ascending=False).head(num_articles)

    for _, row in latest_articles.iterrows():
        st.markdown(f"""
        **🗞️ [{row['title']}]({row['url']})**  
        *Published:* {row['published_on']} | *Source:* `{row['source']}`  
        *Tags:* `{row['tags']}`  
        *Sentiment Score:* `{round(row['sentiment'], 3)}`  
        ---
        """)

    # --- 5. Expandable Audit DataFrame --- #
    with st.expander("📄 Show full sentiment dataframe"):
        st.dataframe(filtered_df.sort_values(["symbol", "date"]), use_container_width=True)