import pandas as pd
import numpy as np
import cvxpy as cp

def load_sentiment_data(sentiment_path, valid_tokens):
    sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"])
    sentiment_df = sentiment_df[sentiment_df["symbol"].isin(valid_tokens)].copy()
    sentiment_df = sentiment_df.groupby(["date", "symbol"])["sentiment"].mean().unstack().fillna(0)
    return sentiment_df

def compute_sentiment_momentum(sentiment_df, window=3):
    return sentiment_df.ewm(span=window, adjust=False).mean()

def load_price_data(price_path, valid_tokens):
    price_df = pd.read_csv(price_path, parse_dates=["time"])
    price_df = price_df[price_df["symbol"].isin(valid_tokens)]
    pivot = price_df.pivot(index="time", columns="symbol", values="close")
    returns = pivot.pct_change().dropna()
    return returns

def mean_variance_optimisation(returns, sentiment_scores, risk_aversion=1.0):
    latest_sentiment = sentiment_scores.iloc[-1]
    expected_returns = returns.rolling(window=7).mean().iloc[-1]
    
    # Combine with sentiment (simple weighted sum)
    blended_returns = 0.6 * expected_returns + 0.4 * latest_sentiment
    blended_returns = blended_returns.fillna(0)

    cov_matrix = returns.cov()

    symbols = blended_returns.index.tolist()
    w = cp.Variable(len(symbols))
    risk = cp.quad_form(w, cov_matrix.values)
    ret = blended_returns.values @ w

    constraints = [
        cp.sum(w) == 1,
        w >= 0.05,
        w <= 0.30
    ]

    prob = cp.Problem(cp.Maximize(ret - risk_aversion * risk), constraints)
    prob.solve()

    weights = pd.Series(w.value, index=symbols)
    return weights.clip(lower=0, upper=1).div(weights.sum())

def sentiment_enhanced_optimizer(price_path="data_outputs/crypto_prices.csv",
                                 sentiment_path="data_outputs/sentiment_timeseries.csv",
                                 risk_aversion=0.1):
    # Load crypto prices
    prices = pd.read_csv(price_path, parse_dates=["time"])
    sentiment = pd.read_csv(sentiment_path, parse_dates=["date"])

    # Pivot prices to wide format
    price_df = prices.pivot(index="time", columns="symbol", values="close").sort_index()

    # Calculate daily returns
    returns = price_df.pct_change().dropna()

    # Limit to recent data (e.g. last 30 days)
    recent_returns = returns.tail(30)

    # Calculate expected return and covariance
    mean_returns = recent_returns.mean()
    cov_matrix = recent_returns.cov()

    # Latest sentiment per token
    latest_sentiment = sentiment.sort_values("date").drop_duplicates("symbol", keep="last").set_index("symbol")["sentiment"]

    # Align with available tokens
    common_tokens = list(set(mean_returns.index) & set(latest_sentiment.index))
    mean_returns = mean_returns[common_tokens]
    cov_matrix = cov_matrix.loc[common_tokens, common_tokens]
    sentiment_scores = latest_sentiment[common_tokens]

    # Adjust returns using sentiment (e.g., amplify returns for high sentiment tokens)
    sentiment_weighting = 1 + sentiment_scores  # linear boost
    adjusted_returns = mean_returns * sentiment_weighting

    # CVXPY optimization
    n = len(common_tokens)
    w = cp.Variable(n)

    objective = cp.Maximize(adjusted_returns.values @ w - risk_aversion * cp.quad_form(w, cov_matrix.values))
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Return weights as DataFrame
    weights_df = pd.DataFrame({
        "symbol": common_tokens,
        "suggested_weight": w.value
    })
    weights_df["suggested_weight"] = weights_df["suggested_weight"].round(4)
    return weights_df
