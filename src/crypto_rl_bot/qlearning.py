from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class QLearningAgent:
    """
    Табличный Q-learning с дискретизацией непрерывного состояния
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 3,
        n_bins: int = 7,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 0.20,
        epsilon_end: float = 0.02,
        epsilon_decay: float = 0.97,
        random_state: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(random_state)
        self.q_table: dict[tuple[int, ...], np.ndarray] = {}

    def discretize_state(self, state: np.ndarray) -> tuple[int, ...]:
        """
        Преобразуем непрерывный state в дискретный ключ
        """
        if state is None or len(state) == 0:
            return tuple([0] * self.state_dim)

        state = np.asarray(state, dtype=float)

        if len(state) != self.state_dim:
            raise ValueError(
                f"Expected state_dim={self.state_dim}, got {len(state)}"
            )

        clipped = np.clip(state, -3.0, 3.0)
        bins = np.linspace(-3.0, 3.0, self.n_bins - 1)
        discrete = np.digitize(clipped, bins)

        return tuple(int(x) for x in discrete)

    def _ensure_state(self, state_key: tuple[int, ...]) -> None:
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions, dtype=float)

    def choose_action(self, state: np.ndarray, greedy: bool = False) -> int:
        state_key = self.discretize_state(state)
        self._ensure_state(state_key)

        if (not greedy) and (self.rng.random() < self.epsilon):
            return int(self.rng.integers(0, self.n_actions))

        q_values = self.q_table[state_key]
        max_q = np.max(q_values)

        # Находим все действия с максимальным Q-значением
        best_actions = np.flatnonzero(q_values == max_q)

        # Если лучших действий несколько, выбираем случайно одно из них
        return int(self.rng.choice(best_actions))
    

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)

        self._ensure_state(state_key)
        self._ensure_state(next_state_key)

        current_q = self.q_table[state_key][action]
        max_next_q = 0.0 if done else np.max(self.q_table[next_state_key])

        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q

        self.q_table[state_key][action] = current_q + self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "n_bins": self.n_bins,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "q_table": {
                "|".join(map(str, state_key)): values.tolist()
                for state_key, values in self.q_table.items()
            },
        }

        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))

        agent = cls(
            state_dim=payload["state_dim"],
            n_actions=payload["n_actions"],
            n_bins=payload["n_bins"],
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon_start=payload["epsilon_start"],
            epsilon_end=payload["epsilon_end"],
            epsilon_decay=payload["epsilon_decay"],
        )
        agent.epsilon = payload["epsilon"]
        agent.q_table = {
            tuple(map(int, key.split("|"))): np.array(values, dtype=float)
            for key, values in payload["q_table"].items()
        }
        return agent