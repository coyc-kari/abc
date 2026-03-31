from __future__ import annotations

import numpy as np

def get_state_at_step(
    features: np.ndarray,
    step: int,
    current_position: int,
) -> np.ndarray:
    """
    Возвращает состояние на текущем шаге
    """
    if len(features) == 0:
        return np.array([float(current_position)], dtype=float)

    if step >= len(features):
        obs = np.zeros(features.shape[1], dtype=float)
    else:
        obs = np.asarray(features[step], dtype=float)

    return np.concatenate([obs, np.array([float(current_position)], dtype=float)])


def action_to_position(action: int) -> int:
    """
    0 -> neutral
    1 -> long spot + short futures
    2 -> short spot + long futures
    """
    if action == 0:
        return 0
    if action == 1:
        return 1
    if action == 2:
        return -1
    raise ValueError(f"Unknown action: {action}")


def calculate_position_pnl(
    prev_portfolio_value: float,
    spot_price: float,
    fut_price: float,
    prev_spot: float,
    prev_fut: float,
    position: int,
    position_size_ratio: float = 0.5,
) -> float:
    """
    Рассчитываем PnL за шаг для уже открытой позиции
    """
    if position == 0:
        return 0

    if prev_spot > 0 and prev_fut > 0:
        spot_return = (spot_price - prev_spot) / prev_spot
        fut_return = (fut_price - prev_fut) / prev_fut
    else:
        return 0

    position_size = prev_portfolio_value * position_size_ratio

    if position == 1:  # Long spot + Short futures
        pnl_amount = position_size * (spot_return - fut_return)
    else:  # Short spot + Long futures
        pnl_amount = position_size * (-spot_return + fut_return)

    pnl = pnl_amount / max(prev_portfolio_value, 1e-6)
    return np.clip(pnl, -0.5, 0.5)


def calculate_reward(
    prev_portfolio_value: float,
    curr_portfolio_value: float,
    prev_position: int,
    new_position: int,
    transaction_cost: float = 0.001,
) -> float:
    """
    Возвращает:
    - reward
    - net_portfolio_value после списания комиссий
    """

    if prev_portfolio_value > 0:
        pnl = (curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value
    else:
        pnl = 0

    pnl = np.clip(pnl, -0.2, 0.2)

    transaction_cost_amount = transaction_cost * (prev_position != new_position)
    bonus = 0.01 * pnl if pnl > 0 else 0
    reward = pnl - transaction_cost_amount + bonus

    if prev_position != new_position and abs(pnl) < 0.005:
        reward -= 0.002

    return np.clip(reward, -0.5, 0.5)


def rl_step(
    data: dict,
    step: int,
    prev_position: int,
    prev_portfolio_value: float,
    action: int,
    transaction_cost: float = 0.001,
    position_size_ratio: float = 0.5,
) -> tuple[int, float, float]:
    """
    Один шаг среды.
    Возвращает:
    - новую позицию
    - новое значение портфеля
    - reward
    """

    # action: 0 -> нейтрально, 1 -> long spot/short futures, 2 -> short spot/long futures
    if action == 0:
        new_position = 0
    elif action == 1:
        new_position = 1
    else:
        new_position = -1

    if step == 0:
        return new_position, prev_portfolio_value, 0

    try:
        spot_price = data['df']['spot_close'].iloc[step]
        fut_price = data['df']['fut_close'].iloc[step]
        prev_spot = data['df']['spot_close'].iloc[step - 1]
        prev_fut = data['df']['fut_close'].iloc[step - 1]
    except (IndexError, KeyError):
        return new_position, prev_portfolio_value, 0

    pnl = calculate_position_pnl(
        prev_portfolio_value, spot_price, fut_price,
        prev_spot, prev_fut, prev_position, position_size_ratio
    )

    portfolio_value = prev_portfolio_value * (1 + pnl)

    reward = calculate_reward(
        prev_portfolio_value, portfolio_value,
        prev_position, new_position, transaction_cost
    )

    return new_position, portfolio_value, reward