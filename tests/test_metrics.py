import pandas as pd
from src.starter_finance.metrics import (
    calculate_cagr,
    calculate_volatility,
    calculate_sharpe,
    calculate_max_drawdown,
)


def test_calculate_cagr():
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    prices = pd.Series([100.0, 110.0], index=[dates[0], dates[-1]])
    cagr = calculate_cagr(prices)
    assert isinstance(cagr, float)


def test_calculate_volatility():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01], dtype=float)
    vol = calculate_volatility(returns)
    assert isinstance(vol, float)


def test_calculate_sharpe():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01], dtype=float)
    sharpe = calculate_sharpe(returns)
    assert isinstance(sharpe, float)


def test_calculate_max_drawdown():
    prices = pd.Series([100.0, 110.0, 105.0, 115.0], dtype=float)
    drawdown = calculate_max_drawdown(prices)
    assert isinstance(drawdown, float)
