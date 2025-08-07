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
