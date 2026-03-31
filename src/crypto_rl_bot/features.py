from __future__ import annotations

import json
import pickle
import sklearn
from pathlib import Path 

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def calculate_core_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитаем базовые метрики по паре:
    - доходности спота и фьючерса
    - лог-доходности спота
    - rolling волатильность
    - rolling корреляцию
    - логарифмический спред

    """
    out = df.copy()

    # Доходности
    out["spot_returns"] = out["spot_close"].pct_change()
    out["fut_returns"] = out["fut_close"].pct_change()
    out["log_returns"] = np.log(out["spot_close"] / out["spot_close"].shift(1))

    # Волатильность (21 день)
    out["spot_volatility"] = out["spot_returns"].rolling(21).std() * np.sqrt(365)
    out["fut_volatility"] = out["fut_returns"].rolling(21).std() * np.sqrt(365)

    # Корреляция
    out["rolling_corr"] = out["spot_returns"].rolling(30).corr(out["fut_returns"])

    # Логарифмический спред
    out["log_spread"] = np.log(out["fut_close"] / out["spot_close"])

    return out


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Рассчитаем RSI 
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитаем признаки для baseline и RL-агента

    """
    out = df.copy()

    # 1. Доходности
    out["spot_returns"] = out["spot_close"].pct_change()
    out["fut_returns"] = out["fut_close"].pct_change()
    out["log_returns"] = np.log(out["spot_close"] / out["spot_close"].shift(1))

    # 2. Волатильность
    out["volatility_20"] = out["spot_returns"].rolling(20).std() * np.sqrt(365)
    out["volatility_60"] = out["spot_returns"].rolling(60).std() * np.sqrt(365)
    out["vol_ratio"] = (
        out["fut_returns"].rolling(20).std()
        / out["spot_returns"].rolling(20).std()
    )

    # 3. Базис
    out["basis"] = out["fut_close"] - out["spot_close"]
    out["basis_pct"] = (out["fut_close"] - out["spot_close"]) / out["spot_close"] * 100
    out["basis_sma_20"] = out["basis_pct"].rolling(20).mean()
    out["basis_std_20"] = out["basis_pct"].rolling(20).std()
    out["basis_zscore"] = (
        out["basis_pct"] - out["basis_pct"].rolling(60).mean()
    ) / out["basis_pct"].rolling(60).std()
    out["basis_momentum"] = out["basis_pct"].diff(5) / out["basis_pct"].shift(5)

    # 3b. Лаги ключевых рядов, чтобы агент видел недавнюю динамику, а не только
    # текущее состояние рынка
    out["basis_pct_lag_1"] = out["basis_pct"].shift(1)
    out["basis_zscore_lag_1"] = out["basis_zscore"].shift(1)

    # 4. Ценовые признаки
    out["sma_20"] = out["spot_close"].rolling(20).mean()
    out["sma_50"] = out["spot_close"].rolling(50).mean()
    out["price_to_sma_20"] = out["spot_close"] / out["sma_20"]
    out["price_to_sma_50"] = out["spot_close"] / out["sma_50"]
    out["above_sma_20"] = (out["spot_close"] > out["sma_20"]).astype(int)
    out["above_sma_50"] = (out["spot_close"] > out["sma_50"]).astype(int)

    # 5. RSI
    out["rsi"] = calculate_rsi(out["spot_close"], 14)

    # 6. MACD
    out["ema_12"] = out["spot_close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["spot_close"].ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # 7. Моментум
    out["momentum_10"] = out["spot_close"].pct_change(10) * 100

    # 
    out["rolling_corr"] = out["spot_returns"].rolling(30).corr(out["fut_returns"])
    out["log_spread"] = np.log(out["fut_close"] / out["spot_close"])

    return out


def get_feature_columns() -> list[str]:
    """
    Финальный набор признаков 
    """
    return [
        "spot_returns",
        "fut_returns",
        "volatility_20",
        "basis_pct",
        "basis_pct_lag_1",
        "basis_zscore",
        "basis_zscore_lag_1",
        "basis_momentum",
        "price_to_sma_20",
        "price_to_sma_50",
        "rsi",
        "macd_hist",
        "momentum_10",
        "above_sma_20",
        "above_sma_50",
        "rolling_corr",
        "log_spread",
    ]


def clean_data_for_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Очистим данные перед обучением
    """
    df_clean = df.copy()
    feature_cols = list(feature_cols)

    for col in feature_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)

    for col in feature_cols:
        if col in df_clean.columns:
            if col in ["above_sma_20", "above_sma_50"]:
                df_clean[col] = df_clean[col].fillna(0)
            else:
                if df_clean[col].notna().sum() > 0:
                    mean_val = df_clean[col].mean()
                    df_clean[col] = df_clean[col].fillna(mean_val)
                else:
                    df_clean[col] = df_clean[col].fillna(0)

    return df_clean


def normalize_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
) -> tuple[np.ndarray, StandardScaler]:
    """
    Нормализация признаков для RL-агента
    """
    feature_cols = list(feature_cols)
    df_features = df[feature_cols].copy()

    for col in feature_cols:
        mean_val = df_features[col].mean()
        std_val = df_features[col].std()
        if std_val > 0:
            upper_bound = mean_val + 3 * std_val
            lower_bound = mean_val - 3 * std_val
            df_features[col] = df_features[col].clip(lower_bound, upper_bound)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_features)

    return features_scaled, scaler


def build_feature_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, StandardScaler, list[str]]:
    """
    Полный пайплайн подготовки признаков
    """
    df_features = calculate_features(df)
    feature_cols = get_feature_columns()
    df_clean = clean_data_for_features(df_features, feature_cols)
    features_scaled, scaler = normalize_features(df_clean, feature_cols)

    return df_clean, features_scaled, scaler, feature_cols

def save_processed_dataset(
    df_clean: pd.DataFrame,
    feature_cols: list[str],
    processed_dir: str | Path,
    dataset_name: str,
    scaler: StandardScaler | None = None,
) -> dict[str, Path]:
    """
    Сохраним подготовленный датасет в data/processed.

    Что сохраняем:
    - parquet с очищенными признаками
    - json со списком feature columns
    - pickle со scaler, если он передан
    """
    processed_dir = Path(processed_dir).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = processed_dir / f"{dataset_name}_feature_dataset.parquet"
    feature_cols_path = processed_dir / f"{dataset_name}_feature_columns.json"

    df_clean.to_parquet(dataset_path, index=False)
    feature_cols_path.write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    paths = {
        "dataset_parquet": dataset_path,
        "feature_columns_json": feature_cols_path,
    }

    if scaler is not None:
        scaler_path = processed_dir / f"{dataset_name}_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        paths["scaler_pkl"] = scaler_path

    return paths


def load_processed_dataset(
    processed_dir: str | Path,
    dataset_name: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Загрузим подготовленный датасет и список признаков
    """
    processed_dir = Path(processed_dir).resolve()

    dataset_path = processed_dir / f"{dataset_name}_feature_dataset.parquet"
    feature_cols_path = processed_dir / f"{dataset_name}_feature_columns.json"

    df = pd.read_parquet(dataset_path)
    feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))

    return df, feature_cols
