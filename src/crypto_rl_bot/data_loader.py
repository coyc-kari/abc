from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .config import (
    AppConfig,
    INTERVAL_TO_FUTURES_GRANULARITY,
    load_config,
)


SPOT_INTERVAL_SECONDS: dict[str, int] = {
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


def resolve_project_path(config_path: str | Path, target_path: str | Path) -> Path:
    """
    Если путь относительный, считаем его относительно корня проекта.
    Корень проекта определяем как родитель директории config-файла.
    """
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent

    target_path = Path(target_path)
    if target_path.is_absolute():
        return target_path

    return project_root / target_path


def parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def align_datetime_to_interval(dt: datetime, interval: str) -> datetime:
    """
    Приводим время к границе свечи.
    Это делает число строк более предсказуемым и уменьшает расхождения
    между спотовым и фьючерсным рядами на границах периода.
    """
    ts = pd.Timestamp(dt)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    if interval == "1day":
        ts = ts.floor("D")
    elif interval == "4hour":
        ts = ts.floor("4h")
    elif interval == "1hour":
        ts = ts.floor("h")
    else:
        raise ValueError(f"Unsupported interval for alignment: {interval}")

    return ts.to_pydatetime()


def resolve_period(
    cfg: AppConfig,
    dataset: str,
    start_iso: str | None = None,
    end_iso: str | None = None,
) -> tuple[datetime, datetime]:
    if start_iso and end_iso:
        start_dt = parse_dt(start_iso)
        end_dt = parse_dt(end_iso)

        if start_dt >= end_dt:
            raise ValueError("start_dt must be earlier than end_dt")

        start_dt = align_datetime_to_interval(start_dt, cfg.data.interval)
        end_dt = align_datetime_to_interval(end_dt, cfg.data.interval)
        return start_dt, end_dt

    end_dt = utc_now()

    if dataset == "baseline":
        start_dt = (
            pd.Timestamp(end_dt) - pd.DateOffset(months=cfg.data.baseline_history_months)
        ).to_pydatetime()
    elif dataset == "rl":
        start_dt = (
            pd.Timestamp(end_dt) - pd.DateOffset(years=cfg.data.rl_history_years)
        ).to_pydatetime()
    else:
        raise ValueError("dataset must be one of {'baseline', 'rl'}")

    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)

    start_dt = align_datetime_to_interval(start_dt, cfg.data.interval)
    end_dt = align_datetime_to_interval(end_dt, cfg.data.interval)

    if start_dt >= end_dt:
        raise ValueError("Resolved start_dt must be earlier than end_dt")

    return start_dt, end_dt


def validate_interval_config(cfg: AppConfig) -> None:
    expected = INTERVAL_TO_FUTURES_GRANULARITY[cfg.data.interval]
    actual = cfg.data.futures_granularity_minutes

    if actual != expected:
        raise ValueError(
            f"Invalid config: interval={cfg.data.interval} requires "
            f"futures_granularity_minutes={expected}, got {actual}"
        )


class KuCoinHistoricalLoader:
    def __init__(
        self,
        spot_base_url: str = "https://api.kucoin.com",
        futures_base_url: str = "https://api-futures.kucoin.com",
        timeout_sec: int = 20,
    ) -> None:
        self.spot_base_url = spot_base_url
        self.futures_base_url = futures_base_url
        self.timeout_sec = timeout_sec
        self.session = requests.Session()

    def _get(self, base_url: str, path: str, params: dict) -> dict:
        response = self.session.get(
            f"{base_url}{path}",
            params=params,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()

        payload = response.json()
        if payload.get("code") != "200000":
            raise RuntimeError(f"KuCoin request failed: {payload}")

        return payload["data"]

    def fetch_spot_candles(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        interval_sec = SPOT_INTERVAL_SECONDS.get(interval)
        if interval_sec is None:
            raise ValueError(f"Unsupported spot interval: {interval}")

        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())

        # Загружаем окнами, чтобы не упираться в ограничения API
        window_sec = interval_sec * 1200
        cursor_start = start_ts
        rows: list[dict] = []

        while cursor_start <= end_ts:
            cursor_end = min(end_ts, cursor_start + window_sec)

            data = self._get(
                self.spot_base_url,
                "/api/v1/market/candles",
                {
                    "symbol": symbol,
                    "type": interval,
                    "startAt": cursor_start,
                    "endAt": cursor_end,
                },
            )

            if data:
                rows.extend(self._parse_spot_row(raw) for raw in data)

            cursor_start = cursor_end + interval_sec

        if not rows:
            raise RuntimeError("No spot candles received from KuCoin")

        frame = (
            pd.DataFrame(rows)
            .drop_duplicates(subset=["ts_sec"])
            .sort_values("ts_sec")
            .reset_index(drop=True)
        )

        frame["timestamp"] = pd.to_datetime(frame["ts_sec"], unit="s", utc=True)

        frame = frame[
            (frame["ts_sec"] >= start_ts) & (frame["ts_sec"] <= end_ts)
        ].reset_index(drop=True)

        return frame

    def fetch_futures_candles(
        self,
        symbol: str,
        granularity_minutes: int,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        granularity_ms = granularity_minutes * 60_000
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        # Загружаем окнами для стабильной пагинации
        window_ms = granularity_ms * 180
        cursor_from = start_ms
        rows: list[dict] = []

        while cursor_from <= end_ms:
            cursor_to = min(end_ms, cursor_from + window_ms)

            data = self._get(
                self.futures_base_url,
                "/api/v1/kline/query",
                {
                    "symbol": symbol,
                    "granularity": granularity_minutes,
                    "from": cursor_from,
                    "to": cursor_to,
                },
            )

            if data:
                rows.extend(self._parse_futures_row(raw) for raw in data)

            cursor_from = cursor_to + granularity_ms

        if not rows:
            raise RuntimeError("No futures candles received from KuCoin")

        frame = (
            pd.DataFrame(rows)
            .drop_duplicates(subset=["ts_ms"])
            .sort_values("ts_ms")
            .reset_index(drop=True)
        )

        frame["timestamp"] = pd.to_datetime(frame["ts_ms"], unit="ms", utc=True)

        frame = frame[
            (frame["ts_ms"] >= start_ms) & (frame["ts_ms"] <= end_ms)
        ].reset_index(drop=True)

        return frame

    def fetch_merged_candles(
        self,
        spot_symbol: str,
        futures_symbol: str,
        interval: str,
        futures_granularity_minutes: int,
        start_dt: datetime,
        end_dt: datetime,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        spot = self.fetch_spot_candles(
            symbol=spot_symbol,
            interval=interval,
            start_dt=start_dt,
            end_dt=end_dt,
        )

        fut = self.fetch_futures_candles(
            symbol=futures_symbol,
            granularity_minutes=futures_granularity_minutes,
            start_dt=start_dt,
            end_dt=end_dt,
        )

        spot["timestamp"] = pd.to_datetime(spot["timestamp"], utc=True).dt.floor("min")
        fut["timestamp"] = pd.to_datetime(fut["timestamp"], utc=True).dt.floor("min")

        merged = pd.merge(
            spot[["timestamp", "open", "high", "low", "close", "volume"]].rename(
                columns={
                    "open": "spot_open",
                    "high": "spot_high",
                    "low": "spot_low",
                    "close": "spot_close",
                    "volume": "spot_volume",
                }
            ),
            fut[["timestamp", "open", "high", "low", "close", "volume"]].rename(
                columns={
                    "open": "fut_open",
                    "high": "fut_high",
                    "low": "fut_low",
                    "close": "fut_close",
                    "volume": "fut_volume",
                }
            ),
            on="timestamp",
            how="inner",
        ).sort_values("timestamp").reset_index(drop=True)

        if merged.empty:
            raise RuntimeError("Merged dataframe is empty")

        return spot, fut, merged

    @staticmethod
    def _parse_spot_row(raw: list) -> dict:
        # Spot format: [time, open, close, high, low, volume, turnover]
        return {
            "ts_sec": int(raw[0]),
            "open": float(raw[1]),
            "close": float(raw[2]),
            "high": float(raw[3]),
            "low": float(raw[4]),
            "volume": float(raw[5]),
            "turnover": float(raw[6]) if len(raw) > 6 else 0.0,
        }

    @staticmethod
    def _parse_futures_row(raw: list) -> dict:
        # Futures format: [time, open, high, low, close, volume, turnover]
        ts = int(raw[0])
        if ts < 10**12:
            ts *= 1000

        close_idx = 4 if len(raw) > 4 else 2
        vol_idx = 5 if len(raw) > 5 else len(raw) - 1

        return {
            "ts_ms": ts,
            "open": float(raw[1]),
            "high": float(raw[2]),
            "low": float(raw[3]),
            "close": float(raw[close_idx]),
            "volume": float(raw[vol_idx]),
            "turnover": float(raw[6]) if len(raw) > 6 else 0.0,
        }


def build_output_prefix(
    cfg: AppConfig,
    dataset: str,
    start_dt: datetime,
    end_dt: datetime,
) -> str:
    spot = cfg.data.spot_symbol.replace("-", "").lower()
    fut = cfg.data.futures_symbol.replace("-", "").lower()
    interval = cfg.data.interval.lower()
    start_str = start_dt.strftime("%Y%m%d")
    end_str = end_dt.strftime("%Y%m%d")

    return f"{dataset}_{spot}_{fut}_{interval}_{start_str}_{end_str}"


def save_history_bundle(
    raw_dir: str | Path,
    prefix: str,
    spot_df: pd.DataFrame,
    fut_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    save_spot_futures: bool = True,
) -> dict[str, Path]:
    raw_dir = Path(raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    if save_spot_futures:
        spot_parquet = raw_dir / f"{prefix}_spot.parquet"
        fut_parquet = raw_dir / f"{prefix}_futures.parquet"

        spot_df.to_parquet(spot_parquet, index=False)
        fut_df.to_parquet(fut_parquet, index=False)

        paths["spot_parquet"] = spot_parquet
        paths["futures_parquet"] = fut_parquet

    merged_parquet = raw_dir / f"{prefix}_merged.parquet"
    merged_df.to_parquet(merged_parquet, index=False)
    paths["merged_parquet"] = merged_parquet

    meta = pd.DataFrame(
        [
            {
                "prefix": prefix,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "rows_spot": len(spot_df),
                "rows_futures": len(fut_df),
                "rows_merged": len(merged_df),
                "start_utc": str(merged_df["timestamp"].min()),
                "end_utc": str(merged_df["timestamp"].max()),
                "interval": prefix.split("_")[3] if len(prefix.split("_")) > 3 else None,
            }
        ]
    )

    meta_path = raw_dir / f"{prefix}_meta.csv"
    meta.to_csv(meta_path, index=False)
    paths["meta_csv"] = meta_path

    return paths


def download_history(
    config_path: str,
    dataset: str,
    start_iso: str | None = None,
    end_iso: str | None = None,
) -> dict:
    cfg = load_config(config_path)
    validate_interval_config(cfg)

    start_dt, end_dt = resolve_period(
        cfg,
        dataset=dataset,
        start_iso=start_iso,
        end_iso=end_iso,
    )

    loader = KuCoinHistoricalLoader(
        spot_base_url=cfg.api.spot_base_url,
        futures_base_url=cfg.api.futures_base_url,
    )

    spot_df, fut_df, merged_df = loader.fetch_merged_candles(
        spot_symbol=cfg.data.spot_symbol,
        futures_symbol=cfg.data.futures_symbol,
        interval=cfg.data.interval,
        futures_granularity_minutes=cfg.data.futures_granularity_minutes,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    prefix = build_output_prefix(cfg, dataset, start_dt, end_dt)
    resolved_raw_dir = resolve_project_path(config_path, cfg.data.raw_dir)

    paths = save_history_bundle(
        raw_dir=resolved_raw_dir,
        prefix=prefix,
        spot_df=spot_df,
        fut_df=fut_df,
        merged_df=merged_df,
        save_spot_futures=True,
    )

    return {
        "config": cfg,
        "dataset": dataset,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "prefix": prefix,
        "spot_rows": len(spot_df),
        "futures_rows": len(fut_df),
        "merged_rows": len(merged_df),
        "paths": paths,
        "data": {
            "spot": spot_df,
            "futures": fut_df,
            "merged": merged_df,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download KuCoin historical data for baseline / RL"
    )
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["baseline", "rl"],
        help="baseline -> 6 months, rl -> 2 years",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="UTC ISO timestamp, example: 2024-03-01T00:00:00Z",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="UTC ISO timestamp, example: 2026-03-01T00:00:00Z",
    )
    args = parser.parse_args()

    result = download_history(
        config_path=args.config,
        dataset=args.dataset,
        start_iso=args.start,
        end_iso=args.end,
    )

    print(f"Набор данных: {result['dataset']}")
    print(f"Интервал: {result['config'].data.interval}")
    print(
        f"Гранулярность фьючерса: "
        f"{result['config'].data.futures_granularity_minutes} мин"
    )
    print(f"Период: {result['start_dt']} -> {result['end_dt']}")
    print(
        f"Число строк: merged={result['merged_rows']}, "
        f"spot={result['spot_rows']}, futures={result['futures_rows']}"
    )
    print("Сохранённые файлы:")
    for key, value in result["paths"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()