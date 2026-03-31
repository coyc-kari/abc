from __future__ import annotations

import json
import pickle
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .config import load_config
from .env import action_to_position, get_state_at_step
from .features import build_feature_dataset
from .kucoin_api import KuCoinExecutionClient, KuCoinPublicDataClient
from .qlearning import QLearningAgent


def _load_model_bundle(model_path: str | Path) -> tuple[QLearningAgent, object, list[str], dict]:
    model_path = Path(model_path).resolve()
    scaler_path = model_path.with_suffix(".scaler.pkl")
    features_path = model_path.with_suffix(".features.json")
    metadata_path = model_path.with_suffix(".metadata.json")

    agent = QLearningAgent.load(model_path)
    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)
    feature_columns = json.loads(features_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return agent, scaler, feature_columns, metadata


def _resolve_live_window(interval: str, lookback_bars: int) -> tuple[datetime, datetime]:
    interval_seconds = {
        "1min": 60,
        "3min": 180,
        "5min": 300,
        "15min": 900,
        "30min": 1800,
        "1hour": 3600,
        "2hour": 7200,
        "4hour": 14400,
        "6hour": 21600,
        "8hour": 28800,
        "12hour": 43200,
        "1day": 86400,
    }
    end_dt = datetime.now(timezone.utc)
    seconds = interval_seconds[interval] * max(lookback_bars, 120)
    start_dt = end_dt - timedelta(seconds=seconds)
    return start_dt, end_dt


def run_live(config_path: str, model_path: str, paper: bool, once: bool) -> None:
    cfg = load_config(config_path)
    agent, scaler, feature_columns, metadata = _load_model_bundle(model_path)

    data_client = KuCoinPublicDataClient(cfg.api)
    execution_client = KuCoinExecutionClient(cfg.api, dry_run=paper)

    current_position = -execution_client.get_futures_position_direction(cfg.data.futures_symbol)
    print(
        "startup",
        {
            "paper_mode": paper,
            "initial_futures_position": current_position,
            "model_rows_used": metadata.get("rows_used"),
            "symbol": cfg.data.futures_symbol,
        },
    )

    while True:
        try:
            start_dt, end_dt = _resolve_live_window(cfg.data.interval, cfg.execution.lookback_bars)
            raw = data_client.fetch_merged_candles(cfg.data, start_dt=start_dt, end_dt=end_dt)
            renamed = raw.rename(
                columns={"futures_close": "fut_close", "futures_volume": "fut_volume"}
            )
            feature_frame, _, _, _ = build_feature_dataset(renamed)
            if feature_frame.empty:
                raise RuntimeError("Feature frame is empty in live loop.")

            latest_features = feature_frame[feature_columns].tail(1)
            scaled_latest = scaler.transform(latest_features)
            observation = get_state_at_step(
                features=scaled_latest,
                step=0,
                current_position=current_position,
            )
            action = agent.choose_action(observation, greedy=True)

            target_position_raw = action_to_position(action)
            target_position = target_position_raw
            if target_position == -1 and not cfg.execution.allow_spot_short:
                target_position = 0

            prices = data_client.fetch_price_snapshot(
                spot_symbol=cfg.data.spot_symbol,
                futures_symbol=cfg.data.futures_symbol,
            )
            spot_size = max(0.0, cfg.execution.quote_notional_usdt / prices.spot_price)
            futures_size = max(
                1,
                int(round(spot_size / max(cfg.execution.futures_contract_multiplier, 1e-9))),
            )

            result = execution_client.rebalance_basis_position(
                current_position=current_position,
                target_position=target_position,
                spot_symbol=cfg.data.spot_symbol,
                futures_symbol=cfg.data.futures_symbol,
                spot_size=spot_size,
                futures_size=futures_size,
                leverage=cfg.execution.leverage,
            )
            if result["changed"]:
                current_position = target_position

            row = feature_frame.iloc[-1]
            print(
                "tick",
                {
                    "timestamp": str(row["timestamp"]),
                    "basis_pct": float(row["basis_pct"]),
                    "basis_zscore": float(row["basis_zscore"]),
                    "action": int(action),
                    "target_position_raw": int(target_position_raw),
                    "target_position": int(target_position),
                    "paper_mode": paper,
                    "orders": result["orders"],
                },
            )
        except Exception as exc:
            print(
                "tick_error",
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "paper_mode": paper,
                    "current_position": current_position,
                    "error": str(exc),
                },
            )
            if once:
                raise

        if once:
            return
        time.sleep(cfg.execution.poll_seconds)
