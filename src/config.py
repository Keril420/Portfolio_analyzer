import os
from pathlib import Path

# Пути к директориям
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
PORTFOLIO_DIR = DATA_DIR / "portfolios"

# Создаем директории, если они не существуют
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, PORTFOLIO_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API-ключи (в реальном проекте следует использовать .env файл)
ALPHA_VANTAGE_API_KEY = os.environ.get("0VJE73FSCQIPH601", "")

# Настройки кеширования
CACHE_EXPIRY_DAYS = 1  # Срок годности кеша в днях

# Настройки портфеля
DEFAULT_BENCHMARK = "SPY"  # Индекс S&P 500 по умолчанию
RISK_FREE_RATE = 0.0  # Безрисковая ставка для расчетов