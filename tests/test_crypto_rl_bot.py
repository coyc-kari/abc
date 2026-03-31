from __future__ import annotations

import json
import pickle

import numpy as np

from crypto_rl_bot.env import action_to_position
from crypto_rl_bot.qlearning import QLearningAgent
from crypto_rl_bot.runtime_env import load_env_file


def test_load_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / "kucoin.env"
    env_file.write_text(
        "KUCOIN_API_KEY=test_key\nKUCOIN_API_SECRET=test_secret\n",
        encoding="utf-8",
    )

    loaded = load_env_file(env_file)

    assert loaded["KUCOIN_API_KEY"] == "test_key"
    assert loaded["KUCOIN_API_SECRET"] == "test_secret"


def test_action_to_position_mapping():
    assert action_to_position(0) == 0
    assert action_to_position(1) == 1
    assert action_to_position(2) == -1


def test_qlearning_save_and_sidecars(tmp_path):
    agent = QLearningAgent(state_dim=3)
    agent.q_table[(1, 2, 3)] = np.array([0.1, 0.2, 0.3], dtype=float)

    model_path = tmp_path / "model.json"
    agent.save(model_path)

    scaler_path = model_path.with_suffix(".scaler.pkl")
    features_path = model_path.with_suffix(".features.json")
    with scaler_path.open("wb") as handle:
        pickle.dump({"ok": True}, handle)
    features_path.write_text(json.dumps(["a", "b"]), encoding="utf-8")

    loaded = QLearningAgent.load(model_path)
    assert loaded.state_dim == 3
    assert tuple(loaded.q_table[(1, 2, 3)]) == (0.1, 0.2, 0.3)
