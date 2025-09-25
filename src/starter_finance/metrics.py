import numpy as np
import pandas as pd


def calculate_cagr(prices: pd.Series) -> float:
    if len(prices) < 2:
        return 0.0
    total_days = (prices.index[-1] - prices.index[0]).days
    years = total_days / 365.25
    if years <= 0:
        return 0.0
    final_value = float(prices.iloc[-1])
    initial_value = float(prices.iloc[0])
    result = (final_value / initial_value) ** (1 / years) - 1
    return float(result)


def calculate_volatility(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(252))


def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    if excess_returns.std() == 0:
        return 0.0
    return float(excess_returns.mean() / excess_returns.std() * np.sqrt(252))


def calculate_max_drawdown(prices: pd.Series) -> float:
    if prices.empty:
        return 0.0
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return float(drawdown.min())
