from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_zscore_signals(
    basis_pct,
    window: int = 60,
    upper_threshold: float = 1.5,
    lower_threshold: float = -1.5,
    neutral_upper: float = 0.3,
    neutral_lower: float = -0.3,
):
    """
    Рассчитаем сигналы Z-score стратегии.

    """
    n = len(basis_pct)
    signals = np.zeros(n)
    zscore = np.full(n, np.nan)

    basis_clean = np.nan_to_num(basis_pct, nan=0.0)

    for i in range(window, n):
        window_data = basis_clean[i - window:i]
        mean_b = np.mean(window_data)
        std_b = np.std(window_data)

        if std_b > 1e-6:
            zscore[i] = (basis_clean[i] - mean_b) / std_b
        else:
            zscore[i] = 0

    current_position = 0

    for i in range(n):
        if np.isnan(zscore[i]):
            signals[i] = 0
        elif zscore[i] > upper_threshold:
            signals[i] = 1  # short futures + long spot
            current_position = 1
        elif zscore[i] < lower_threshold:
            signals[i] = -1  # long futures + short spot
            current_position = -1
        elif neutral_lower <= zscore[i] <= neutral_upper:
            signals[i] = 0
            current_position = 0
        else:
            signals[i] = current_position

    return signals, zscore


def calculate_strategy_returns(
    basis_pct,
    signals,
    transaction_cost: float = 0.001,
):
    """
    Рассчитаем доходность стратегии по изменению базиса.

    """
    basis_clean = np.nan_to_num(basis_pct, nan=0.0)

    basis_change = np.diff(basis_clean, prepend=basis_clean[0])
    basis_change = basis_change / 100

    strategy_returns = signals * basis_change

    position_changes = np.abs(np.diff(signals, prepend=0))
    transaction_costs = position_changes * transaction_cost

    net_returns = strategy_returns - transaction_costs

    return strategy_returns, net_returns


def backtest_zscore_strategy(
    df: pd.DataFrame,
    initial_capital: float = 10000,
    window: int = 60,
    upper_threshold: float = 1.5,
    lower_threshold: float = -1.5,
    neutral_upper: float = 0.3,
    neutral_lower: float = -0.3,
    transaction_cost: float = 0.001,
) -> dict:
    """
    Бэктестинг baseline Z-score стратегии.

    """
    basis_pct = df["basis_pct"].values
    basis_pct = np.nan_to_num(basis_pct, nan=0.0)

    signals, zscore = calculate_zscore_signals(
        basis_pct,
        window=window,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        neutral_upper=neutral_upper,
        neutral_lower=neutral_lower,
    )

    strategy_returns, net_returns = calculate_strategy_returns(
        basis_pct,
        signals,
        transaction_cost,
    )

    portfolio = initial_capital * (1 + net_returns).cumprod()

    total_return = (portfolio[-1] - initial_capital) / initial_capital

    if np.std(net_returns) > 0:
        sharpe_ratio = (np.mean(net_returns) / np.std(net_returns)) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    running_max = np.maximum.accumulate(portfolio)
    drawdown = (portfolio - running_max) / running_max
    max_drawdown = np.min(drawdown)

    win_rate = (
        np.mean(net_returns[net_returns != 0] > 0)
        if np.any(net_returns != 0)
        else 0
    )

    trades = np.sum(np.abs(np.diff(signals, prepend=0)) > 0)

    results = {
        "signals": signals,
        "zscore": zscore,
        "strategy_returns": strategy_returns,
        "net_returns": net_returns,
        "portfolio": portfolio,
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trades": trades,
        "transaction_costs_total": np.sum(
            np.abs(np.diff(signals, prepend=0)) * transaction_cost
        ),
    }

    return results


def format_baseline_results(results: dict) -> pd.DataFrame:
    """
    Таблица результатов baseline.
    """
    return pd.DataFrame(
        [
            {
                "total_return": results["total_return"],
                "sharpe_ratio": results["sharpe_ratio"],
                "max_drawdown": results["max_drawdown"],
                "win_rate": results["win_rate"],
                "trades": results["trades"],
                "transaction_costs_total": results["transaction_costs_total"],
            }
        ]
    )