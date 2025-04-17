import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variable from .env file
load_dotenv()

# Paths to directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
PORTFOLIO_DIR = DATA_DIR / "portfolios"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, PORTFOLIO_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API keys with environment variable priority
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')

# Caching settings
CACHE_EXPIRY_DAYS = 1  # Cache expiration date in days

# Portfolio settings
DEFAULT_BENCHMARK = "SPY"  # Default S&P 500 Index
RISK_FREE_RATE = 0.0 # Risk-free rate for calculations