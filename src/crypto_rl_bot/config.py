from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Поддерживаемые интервалы 
SUPPORTED_INTERVALS = {
    "1min",
    "3min",
    "5min",
    "15min",
    "30min",
    "1hour",
    "2hour",
    "4hour",
    "6hour",
    "8hour",
    "12hour",
    "1day",
}
INTERVAL_TO_FUTURES_GRANULARITY = {
    "1min": 1,
    "3min": 3,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1hour": 60,
    "2hour": 120,
    "4hour": 240,
    "6hour": 360,
    "8hour": 480,
    "12hour": 720,
    "1day": 1440,
}


@dataclass
class ApiConfig:
    api_key_env: str = "KUCOIN_API_KEY"
    api_secret_env: str = "KUCOIN_API_SECRET"
    api_passphrase_env: str = "KUCOIN_API_PASSPHRASE"
    is_sandbox: bool = False
    spot_base_url: str = "https://api.kucoin.com"
    futures_base_url: str = "https://api-futures.kucoin.com"


@dataclass
class DataConfig:
    spot_symbol: str = "BTC-USDT"
    futures_symbol: str = "XBTUSDTM"

    # Добавим выбор интервала из поддерживамых
    # 1hour / 4hour / 1day
    interval: str = "1hour"

    # Если не указать в json, подставим автоматически из interval
    futures_granularity_minutes: int | None = None

    # Горизонт по требованиям дз: для бэйзлайна 6 мес, для rl агента 2 года
    baseline_history_months: int = 6
    rl_history_years: int = 2

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"

    def __post_init__(self) -> None:
        if self.interval not in SUPPORTED_INTERVALS:
            raise ValueError(
                f"Unsupported interval: {self.interval}. "
                f"Supported: {sorted(SUPPORTED_INTERVALS)}"
            )

        expected = INTERVAL_TO_FUTURES_GRANULARITY[self.interval]

        if self.futures_granularity_minutes is None:
            self.futures_granularity_minutes = expected
        elif self.futures_granularity_minutes != expected:
            raise ValueError(
                f"For interval='{self.interval}' expected "
                f"futures_granularity_minutes={expected}, "
                f"got {self.futures_granularity_minutes}"
            )

        if self.baseline_history_months <= 0:
            raise ValueError("baseline_history_months must be > 0")

        if self.rl_history_years <= 0:
            raise ValueError("rl_history_years must be > 0")


@dataclass
class FeatureConfig:
    zscore_window: int = 60
    volatility_window: int = 20
    volume_window: int = 20
    basis_momentum_lag: int = 5


@dataclass
class BaselineConfig:
    enter_zscore: float = 1.5
    exit_zscore: float = 0.3
    transaction_cost: float = 0.001
    initial_capital: float = 10_000.0


@dataclass
class RlConfig:
    episodes: int = 100
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 0.20
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.97


@dataclass
class ExecutionConfig:
    poll_seconds: int = 60
    lookback_bars: int = 240
    quote_notional_usdt: float = 100.0
    leverage: int = 2
    futures_contract_multiplier: float = 0.001
    allow_spot_short: bool = False


@dataclass
class AppConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    rl: RlConfig = field(default_factory=RlConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppConfig":
        return cls(
            api=ApiConfig(**payload.get("api", {})),
            data=DataConfig(**payload.get("data", {})),
            features=FeatureConfig(**payload.get("features", {})),
            baseline=BaselineConfig(**payload.get("baseline", {})),
            rl=RlConfig(**payload.get("rl", {})),
            execution=ExecutionConfig(**payload.get("execution", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return AppConfig.from_dict(payload)
