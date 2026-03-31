# Crypto RL Bot

Репозиторий для RL-стратегии basis trading между:
- спотом: `BTC-USDT`
- бессрочным фьючерсом: `XBTUSDTM`

В проекте есть:
- загрузка исторических данных KuCoin,
- построение признаков,
- обучение Q-learning агента,
- `shadow`-режим для paper trading,
- `live`-режим для реальной торговли на KuCoin,
- воспроизводимый запуск локально и через Docker.

## Структура репозитория

- `src/crypto_rl_bot/*.py` - основная логика: данные, признаки, RL, KuCoin API, обучение, live execution.
- `config/basis_strategy_config.json` - основной конфиг стратегии и исполнения.
- `trade_signal_executor_kucoin.py` - главный executor с режимами `train` / `shadow` / `live`.
- `run_trade_signal.py` - кроссплатформенный launcher.
- `run_kucoin_trade_signal.ps1` - обёртка для Windows PowerShell с логами.
- `run_kucoin_trade_signal.sh` - обёртка для macOS/Linux с логами.
- `Dockerfile` - контейнер для воспроизводимого запуска.
- `docker-compose.yml` - готовые сервисы для `live`, `shadow` и быстрого `train`.
- `examples/kucoin.env.example` - пример файла с переменными окружения для KuCoin.

## 1) Локальная установка

Создание виртуального окружения:

```bash
python -m venv .venv
```

Windows:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install -r requirements.txt
```

Если активация PowerShell не нужна, можно использовать Python из окружения напрямую:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel "setuptools<81"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install -r requirements.txt
chmod +x run_kucoin_trade_signal.sh
```

## 2) Ключи KuCoin

Создайте локальный runtime-файл:

```powershell
New-Item -ItemType Directory -Force .runtime
Copy-Item examples\kucoin.env.example .runtime\kucoin.env
```

или на macOS/Linux:

```bash
mkdir -p .runtime
cp examples/kucoin.env.example .runtime/kucoin.env
```

Ожидаемые переменные:

```env
KUCOIN_API_KEY=...
KUCOIN_API_SECRET=...
KUCOIN_API_PASSPHRASE=...
```

`trade_signal_executor_kucoin.py` автоматически подгружает `.runtime/kucoin.env`.

## 3) Обучение модели

Обучение на исторических данных KuCoin:

```powershell
python run_trade_signal.py --mode train --config config/basis_strategy_config.json --model-path models/btc_basis_qlearning.json --features-out reports/btc_basis_features.csv
```

Обучение на заданном периоде:

```powershell
python run_trade_signal.py --mode train --start "2026-01-01T00:00:00Z" --end "2026-03-01T00:00:00Z" --config config/basis_strategy_config.json --model-path models/btc_basis_qlearning.json
```

Быстрый smoke-train:

```powershell
python run_trade_signal.py --mode train --episodes 10 --start "2026-03-01T00:00:00Z" --end "2026-03-10T00:00:00Z" --config config/basis_strategy_config.json --model-path models/btc_basis_qlearning.json
```

Артефакты сохраняются рядом с моделью:
- `models/btc_basis_qlearning.json`
- `models/btc_basis_qlearning.scaler.pkl`
- `models/btc_basis_qlearning.features.json`
- `models/btc_basis_qlearning.metadata.json`

## 4) Shadow-режим

Один цикл:

```powershell
python run_trade_signal.py --mode shadow --once --train-if-missing --config config/basis_strategy_config.json --model-path models/btc_basis_qlearning.json
```

Непрерывный paper loop:

```powershell
python run_trade_signal.py --mode shadow --train-if-missing --config config/basis_strategy_config.json --model-path models/btc_basis_qlearning.json
```

Через PowerShell-обёртку:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_kucoin_trade_signal.ps1 -Mode shadow -TrainIfMissing
```

## 5) Live-режим

Реальная торговля:

```powershell
python run_trade_signal.py --mode live --run-real-order --config config/basis_strategy_config.json --model-path models/btc_basis_qlearning.json --env-file .runtime/kucoin.env
```

Через PowerShell-обёртку:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_kucoin_trade_signal.ps1 -Mode live -RunRealOrder -TrainIfMissing
```

## 6) Docker

Сборка образа:

```bash
docker build -t crypto-rl-bot .
```

Обучение в контейнере:

```powershell
docker run --rm -it -v ${PWD}/models:/app/models -v ${PWD}/reports:/app/reports -v ${PWD}/data:/app/data -v ${PWD}/.runtime:/app/.runtime crypto-rl-bot python run_trade_signal.py --mode train --config config/basis_strategy_config.json --model-path models/btc_basis_qlearning.json --features-out reports/btc_basis_features.csv --env-file .runtime/kucoin.env
```

Один `shadow`-запуск в контейнере:

```powershell
docker run --rm -it -v ${PWD}/models:/app/models -v ${PWD}/reports:/app/reports -v ${PWD}/data:/app/data -v ${PWD}/.runtime:/app/.runtime crypto-rl-bot python run_trade_signal.py --mode shadow --once --train-if-missing --config config/basis_strategy_config.json --model-path models/btc_basis_qlearning.json --env-file .runtime/kucoin.env
```

Непрерывный `live`-контейнер:

```powershell
docker run -d --name btc-basis-live --restart unless-stopped -v ${PWD}/models:/app/models -v ${PWD}/reports:/app/reports -v ${PWD}/logs:/app/logs -v ${PWD}/data:/app/data -v ${PWD}/.runtime:/app/.runtime crypto-rl-bot python run_trade_signal.py --mode live --run-real-order --train-if-missing --config config/basis_strategy_config.json --model-path models/btc_basis_qlearning.json --env-file .runtime/kucoin.env
```

Логи и остановка:

```bash
docker logs -f btc-basis-live
docker rm -f btc-basis-live
```

Запуск через compose:

```bash
docker compose up -d btc-basis-live
docker compose logs -f btc-basis-live
docker compose down
```

Ручные compose-профили:

```bash
docker compose --profile manual run --rm btc-basis-train-fast
docker compose --profile manual run --rm btc-basis-shadow-once
```

## 7) Воспроизводимый сценарий запуска

Минимальный локальный сценарий:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install -r requirements.txt
New-Item -ItemType Directory -Force .runtime,models,reports,logs
Copy-Item examples\kucoin.env.example .runtime\kucoin.env
python run_trade_signal.py --mode train --train-if-missing --config config/basis_strategy_config.json --model-path models/btc_basis_qlearning.json
python run_trade_signal.py --mode shadow --once --train-if-missing --config config/basis_strategy_config.json --model-path models/btc_basis_qlearning.json
```

Минимальный Docker-сценарий:

```powershell
New-Item -ItemType Directory -Force .runtime,models,reports,logs,data
Copy-Item examples\kucoin.env.example .runtime\kucoin.env
docker build -t crypto-rl-bot .
docker compose --profile manual run --rm btc-basis-train-fast
docker compose --profile manual run --rm btc-basis-shadow-once
```

## 8) Быстрая проверка после установки

Проверка help-команд:

```powershell
python run_trade_signal.py --help
python trade_signal_executor_kucoin.py --help
```

Запуск тестов:

```powershell
python -m pytest tests\test_crypto_rl_bot.py
```

## 9) Jupyter и ноутбуки

Запуск JupyterLab:

```powershell
python -m jupyter lab
```

или

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```
