from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import KuCoinHistoricalLoader, parse_dt, resolve_project_path
from .features import build_feature_dataset
from .env import get_state_at_step, rl_step
from .qlearning import QLearningAgent
from .config import load_config


def split_timeseries_data(
    df: pd.DataFrame,
    features_scaled: np.ndarray,
    train_ratio: float = 0.7,
) -> tuple[dict, dict]:
    """
    Разделяем данные по времени без перемешивания
    """
    if len(df) != len(features_scaled):
        raise ValueError("df and features_scaled must have the same length")

    n = len(df)
    split_idx = int(n * train_ratio)

    train_data = {
        "df": df.iloc[:split_idx].reset_index(drop=True),
        "features": np.asarray(features_scaled[:split_idx], dtype=float),
    }

    test_data = {
        "df": df.iloc[split_idx:].reset_index(drop=True),
        "features": np.asarray(features_scaled[split_idx:], dtype=float),
    }

    return train_data, test_data


def run_training_episode(
    agent: QLearningAgent,
    data: dict,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,
    position_size_ratio: float = 0.5,
) -> dict:
    """
    Один обучающий эпизод.
    """
    n_steps = len(data["df"])

    portfolio_value = initial_capital
    position = 0

    portfolio_history = [portfolio_value]
    position_history = [position]
    reward_history = [0.0]
    action_history = []

    total_reward = 0.0

    for step in range(n_steps):
        state = get_state_at_step(
            features=data["features"],
            step=step,
            current_position=position,
        )

        action = agent.choose_action(state, greedy=False)

        new_position, new_portfolio, reward = rl_step(
            data=data,
            step=step,
            prev_position=position,
            prev_portfolio_value=portfolio_value,
            action=action,
            transaction_cost=transaction_cost,
            position_size_ratio=position_size_ratio,
        )

        next_state = get_state_at_step(
            features=data["features"],
            step=step + 1,
            current_position=new_position,
        )

        done = step == (n_steps - 1)

        agent.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        action_history.append(action)
        position = new_position
        portfolio_value = new_portfolio
        total_reward += reward

        portfolio_history.append(portfolio_value)
        position_history.append(position)
        reward_history.append(reward)

    return {
        "portfolio": np.array(portfolio_history, dtype=float),
        "positions": np.array(position_history, dtype=int),
        "rewards": np.array(reward_history, dtype=float),
        "actions": np.array(action_history, dtype=int),
        "total_reward": float(total_reward),
        "final_portfolio": float(portfolio_value),
    }


def train_qlearning_agent(
    train_data: dict,
    state_dim: int,
    episodes: int = 100,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 0.20,
    epsilon_end: float = 0.02,
    epsilon_decay: float = 0.97,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,
    position_size_ratio: float = 0.5,
    n_bins: int = 7,
    random_state: int = 42,
) -> tuple[QLearningAgent, pd.DataFrame]:
    """
    Обучаем агента по эпизодам.
    """
    agent = QLearningAgent(
        state_dim=state_dim,
        n_actions=3,
        n_bins=n_bins,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        random_state=random_state,
    )

    history = []

    for episode in range(episodes):
        result = run_training_episode(
            agent=agent,
            data=train_data,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            position_size_ratio=position_size_ratio,
        )

        history.append(
            {
                "episode": episode + 1,
                "epsilon": agent.epsilon,
                "total_reward": result["total_reward"],
                "final_portfolio": result["final_portfolio"],
            }
        )

        agent.decay_epsilon()

    history_df = pd.DataFrame(history)
    return agent, history_df


def evaluate_qlearning_agent(
    agent: QLearningAgent,
    data: dict,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,
    position_size_ratio: float = 0.5,
) -> dict:
    """
    Оценка агента на train или test 
    """
    n_steps = len(data["df"])

    portfolio_value = initial_capital
    position = 0

    portfolio_history = [portfolio_value]
    position_history = [position]
    reward_history = [0.0]
    action_history = []

    for step in range(n_steps):
        state = get_state_at_step(
            features=data["features"],
            step=step,
            current_position=position,
        )

        action = agent.choose_action(state, greedy=True)

        new_position, new_portfolio, reward = rl_step(
            data=data,
            step=step,
            prev_position=position,
            prev_portfolio_value=portfolio_value,
            action=action,
            transaction_cost=transaction_cost,
            position_size_ratio=position_size_ratio,
        )

        action_history.append(action)
        position = new_position
        portfolio_value = new_portfolio

        portfolio_history.append(portfolio_value)
        position_history.append(position)
        reward_history.append(reward)

    portfolio = np.array(portfolio_history, dtype=float)
    positions = np.array(position_history, dtype=int)
    rewards = np.array(reward_history, dtype=float)
    actions = np.array(action_history, dtype=int)

    net_returns = pd.Series(portfolio).pct_change().fillna(0.0).to_numpy()
    total_return = float((portfolio[-1] - initial_capital) / initial_capital)

    if np.std(net_returns) > 0:
        sharpe_ratio = float((np.mean(net_returns) / np.std(net_returns)) * np.sqrt(252))
    else:
        sharpe_ratio = 0.0

    running_max = np.maximum.accumulate(portfolio)
    drawdown = (portfolio - running_max) / running_max
    max_drawdown = float(np.min(drawdown))

    non_zero_returns = net_returns[net_returns != 0]
    win_rate = float(np.mean(non_zero_returns > 0)) if len(non_zero_returns) > 0 else 0.0

    position_changes = np.abs(np.diff(positions))
    trades = int(np.sum(position_changes > 0))
    transaction_costs_total = float(np.sum(position_changes) * transaction_cost)

    years = max(len(data["df"]) / 365.0, 1e-9)
    cagr = float((portfolio[-1] / initial_capital) ** (1 / years) - 1)

    return {
        "portfolio": portfolio,
        "positions": positions,
        "rewards": rewards,
        "actions": actions,
        "net_returns": net_returns,
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trades": trades,
        "cagr": cagr,
        "transaction_costs_total": transaction_costs_total,
    }


def _save_training_artifacts(
    model_path: Path,
    agent: QLearningAgent,
    scaler: object,
    feature_columns: list[str],
    metadata: dict[str, object],
) -> dict[str, str]:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)

    scaler_path = model_path.with_suffix(".scaler.pkl")
    features_path = model_path.with_suffix(".features.json")
    metadata_path = model_path.with_suffix(".metadata.json")

    with scaler_path.open("wb") as handle:
        pickle.dump(scaler, handle)
    features_path.write_text(
        json.dumps(feature_columns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "features_path": str(features_path),
        "metadata_path": str(metadata_path),
    }


def run_training(
    config_path: str,
    model_out: str,
    source_csv: str | None = None,
    start_iso: str | None = None,
    end_iso: str | None = None,
    features_out: str | None = None,
    episodes_override: int | None = None,
) -> dict[str, float | int | str]:
    cfg = load_config(config_path)
    if episodes_override is not None:
        cfg.rl.episodes = int(episodes_override)

    if source_csv:
        raw = pd.read_csv(source_csv, parse_dates=["timestamp"])
        if "timestamp" in raw.columns:
            raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    else:
        loader = KuCoinHistoricalLoader(
            spot_base_url=cfg.api.spot_base_url,
            futures_base_url=cfg.api.futures_base_url,
        )
        if start_iso and end_iso:
            start_dt = parse_dt(start_iso)
            end_dt = parse_dt(end_iso)
        else:
            end_dt = datetime.now(timezone.utc)
            history_hours = max(cfg.data.rl_history_years * 365 * 24, 24)
            start_dt = end_dt - pd.Timedelta(hours=history_hours)
        _, _, raw = loader.fetch_merged_candles(
            spot_symbol=cfg.data.spot_symbol,
            futures_symbol=cfg.data.futures_symbol,
            interval=cfg.data.interval,
            futures_granularity_minutes=int(cfg.data.futures_granularity_minutes or 0),
            start_dt=start_dt,
            end_dt=end_dt,
        )

    renamed = raw.rename(columns={"futures_close": "fut_close", "futures_volume": "fut_volume"})
    feature_frame, features_scaled, scaler, feature_columns = build_feature_dataset(renamed)
    if len(feature_frame) < 50:
        raise RuntimeError("Not enough rows after feature engineering for RL training.")

    train_data, test_data = split_timeseries_data(feature_frame, features_scaled, train_ratio=0.7)
    agent, history_df = train_qlearning_agent(
        train_data=train_data,
        state_dim=features_scaled.shape[1] + 1,
        episodes=cfg.rl.episodes,
        alpha=cfg.rl.alpha,
        gamma=cfg.rl.gamma,
        epsilon_start=cfg.rl.epsilon_start,
        epsilon_end=cfg.rl.epsilon_end,
        epsilon_decay=cfg.rl.epsilon_decay,
        initial_capital=cfg.baseline.initial_capital,
        transaction_cost=cfg.baseline.transaction_cost,
    )
    evaluation = evaluate_qlearning_agent(
        agent=agent,
        data=test_data,
        initial_capital=cfg.baseline.initial_capital,
        transaction_cost=cfg.baseline.transaction_cost,
    )

    model_path = Path(model_out).resolve()
    metadata = {
        "config_path": str(Path(config_path).resolve()),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_used": int(len(feature_frame)),
        "feature_columns": feature_columns,
        "episodes": int(cfg.rl.episodes),
        "train_rows": int(len(train_data["df"])),
        "test_rows": int(len(test_data["df"])),
    }
    artifact_paths = _save_training_artifacts(
        model_path=model_path,
        agent=agent,
        scaler=scaler,
        feature_columns=feature_columns,
        metadata=metadata,
    )

    if features_out:
        features_path = resolve_project_path(config_path, features_out)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        feature_frame.to_csv(features_path, index=False)

    return {
        "episodes": int(cfg.rl.episodes),
        "rows": int(len(feature_frame)),
        "train_rows": int(len(train_data["df"])),
        "test_rows": int(len(test_data["df"])),
        "avg_reward_last_10": float(history_df["total_reward"].tail(10).mean()),
        "best_reward": float(history_df["total_reward"].max()),
        "test_total_return": float(evaluation["total_return"]),
        "test_sharpe_ratio": float(evaluation["sharpe_ratio"]),
        "model_path": artifact_paths["model_path"],
    }
