import os
import time
import logging
import pickle
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional, Union, Any

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_fetcher')


class DataFetcher:
    """
        The main class for retrieving financial data from various sources.
        Supports data caching for performance optimization.
    """

    def __init__(self, cache_dir: str = './cache', cache_expiry_days: int = 1):
        """
        Initializing the data loader

        Args:
            cache_dir: Directory for caching data
            cache_expiry_days: Cache expiration date in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_expiry_days = cache_expiry_days
        self.api_call_counts = {'yfinance': 0, 'alpha_vantage': 0}
        self.api_limits = {'alpha_vantage': 500}

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load API keys from the configuration file or environment variables
        self.api_keys = {
            'alpha_vantage': os.environ.get('ALPHA_VANTAGE_API_KEY', '')
        }

        # Dictionary of supported data providers
        self.providers = {
            'yfinance': self._fetch_yfinance,
            'alpha_vantage': self._fetch_alpha_vantage
        }

        #Preloading benchmark data
        self.benchmark_data = {}
        self.benchmark_tickers = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']
        self.preload_benchmarks()

        logger.info(f"Initialized DataFetcher with cache in {self.cache_dir}")

        # Preloading benchmark data
        self.benchmark_data = {}
        self.benchmark_tickers = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']
        self.preload_benchmarks()

    # Added location: new method in DataFetcher class

    def preload_benchmarks(self):
        """Preloading major benchmark data with a long historical period"""
        start_date = '1990-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')

        for ticker in self.benchmark_tickers:
            try:
                data = self.get_historical_prices(ticker, start_date, end_date, force_refresh=False)
                if not data.empty:
                    self.benchmark_data[ticker] = data
                    logger.info(f"Historical data for benchmark preloaded {ticker}")
            except Exception as e:
                logger.warning(f"Failed to preload benchmark data {ticker}: {e}")

    def get_historical_prices(
            self,
            ticker: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            provider: str = 'yfinance',
            interval: str = '1d',
            force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get historical prices for the specified ticker
        Args:
            ticker: Stock/ETF ticker
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            provider: Data provider ('yfinance', 'alpha_vantage')
            interval: Data interval ('1d', '1wk', '1mo')
            force_refresh: Force data refresh
        Returns:
            DataFrame with historical prices
        """

        # Check if the ticker contains a dot and create a corrected version for the API
        original_ticker = ticker
        corrected_ticker = ticker.replace('.', '-') if '.' in ticker else ticker

        if corrected_ticker != original_ticker:
            logger.info(f"Use corrected ticker {corrected_ticker} to query {original_ticker}")

        # Use original ticker for cache
        cache_key = original_ticker

        # Set default values for dates
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date is None:

            start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

        # Check cache expiration date
        cache_file = self.cache_dir / f"{cache_key}_{start_date}_{end_date}_{interval}_{provider}.pkl"

        if not force_refresh and cache_file.exists():
            # Check cache age
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < self.cache_expiry_days:
                try:
                    with open(cache_file, 'rb') as f:
                        logger.info(f"Loading {original_ticker} data from cache")
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cache for {original_ticker}: {e}")

        # If the cache is missing or outdated, we get new data
        if provider in self.providers:
            try:

                data = self.providers[provider](corrected_ticker, start_date, end_date, interval)

                if data is not None and not data.empty:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(data, f)
                    logger.info(f"Saved data {original_ticker} to cache")

                return data
            except Exception as e:
                logger.error(f"Error retrieving data for {original_ticker} from {provider}: {e}")

                fallback_provider = next((p for p in self.providers.keys() if p != provider), None)
                if fallback_provider:
                    logger.info(f"Let's try an alternative provider: {fallback_provider}")
                    return self.get_historical_prices(
                        ticker, start_date, end_date, fallback_provider, interval, force_refresh
                    )
        else:
            raise ValueError(f"Unsupported data provider: {provider}")

        return pd.DataFrame()

    def get_company_info(self, ticker: str, provider: str = 'yfinance') -> Dict:
        """
        Getting information about the company

        Args:
            ticker: Stock/ETF ticker
            provider: Data provider

        Returns:
            Dictionary with information about the company
        """
        cache_file = self.cache_dir / f"{ticker}_info_{provider}.json"

        # Check cache (company information expires after 7 days)
        if cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 7:
                try:
                    with open(cache_file, 'r') as f:
                        logger.info(f"Loading information about {ticker} from cache")
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cache information for {ticker}: {e}")

        info = {}

        try:
            if provider == 'yfinance':
                info = self._get_yfinance_company_info(ticker)
            elif provider == 'alpha_vantage':
                info = self._get_alpha_vantage_company_info(ticker)
            else:
                raise ValueError(f"Unsupported provider for company information: {provider}")

            if info:
                with open(cache_file, 'w') as f:
                    json.dump(info, f)
                logger.info(f"Saved information about {ticker} to cache")

            return info
        except Exception as e:
            logger.error(f"Error getting information about {ticker}: {e}")

            # Trying an alternative provider
            if provider != 'yfinance':
                logger.info("We are trying to get information through yfinance")
                return self.get_company_info(ticker, 'yfinance')

            return {}

    def search_tickers(self, query: str, limit: int = 10, provider: str = 'alpha_vantage') -> List[Dict]:
        """
        Search tickers by request

        Args:
            query: Search query
            limit: Maximum number of results
            provider: Data provider

        Returns:
            List of dictionaries with information about found tickers
        """
        cache_key = f"search_{query.lower()}_{provider}.json"
        cache_file = self.cache_dir / cache_key

        # Check cache (search results expire after 3 days)
        if cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 3:
                try:
                    with open(cache_file, 'r') as f:
                        results = json.load(f)
                        logger.info(f"Loading search results for '{query}' from cache")
                        return results[:limit]
                except Exception as e:
                    logger.warning(f"Failed to load search cache: {e}")

        results = []

        try:
            if provider == 'alpha_vantage':
                results = self._search_alpha_vantage(query, limit)
            elif provider == 'yfinance':

                results = self._search_alternative(query, limit)
            else:
                raise ValueError(f"Unsupported provider for search: {provider}")

            if results:
                with open(cache_file, 'w') as f:
                    json.dump(results, f)
                logger.info(f"Search results for '{query}' saved to cache")

            return results[:limit]
        except Exception as e:
            logger.error(f"Error searching for tickers on request '{query}': {e}")

            # Trying an alternative provider
            if provider != 'alpha_vantage' and self.api_keys['alpha_vantage']:
                return self.search_tickers(query, limit, 'alpha_vantage')

            return []

    def validate_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Проверка валидности списка тикеров

        Args:
            tickers: List of tickers to check

        Returns:
           Tuple (valid_tickers, invalid_tickers)
        """

        # List of popular tickers that we consider valid by default
        popular_tickers = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'META', 'NVDA',
            'JPM', 'V', 'WMT', 'SPY', 'QQQ', 'VTI', 'VOO', 'BRK.B'
        ]

        valid_tickers = []
        invalid_tickers = []

        for ticker in tickers:
            if ticker in popular_tickers:
                valid_tickers.append(ticker)
                continue

            try:
                # Trying to get basic information about a ticker
                data = self.get_historical_prices(ticker,
                                                  start_date=(datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d'),
                                                  end_date=datetime.now().strftime('%Y-%m-%d'))

                if data is not None and not data.empty:
                    valid_tickers.append(ticker)
                else:
                    invalid_tickers.append(ticker)
            except Exception:
                invalid_tickers.append(ticker)

        return valid_tickers, invalid_tickers

    def get_macro_indicators(self, indicators: List[str], start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
      Obtaining macroeconomic indicators

        Args:
            indicators: List of indicators
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary {indicator: DataFrame}
        """
        results = {}

        # Map indicator codes to their FRED codes
        indicator_mapping = {
            'INFLATION': 'CPIAUCSL',  # Consumer Price Index
            'GDP': 'GDP',  # Gross Domestic Product
            'UNEMPLOYMENT': 'UNRATE',  # Unemployment Rate
            'INTEREST_RATE': 'FEDFUNDS',  # Federal Funds Rate
            'RETAIL_SALES': 'RSXFS',  # Retail Sales
            'INDUSTRIAL_PRODUCTION': 'INDPRO',  # Industrial Production Index
            'HOUSE_PRICE_INDEX': 'CSUSHPISA',  # Case-Shiller Home Price Index
            'CONSUMER_SENTIMENT': 'UMCSENT'  # University of Michigan Consumer Sentiment
        }

        try:
            import pandas_datareader.data as web

            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            if start_date is None:
                start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

            # Convert date strings to datetime
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)

            for indicator in indicators:
                try:

                    fred_code = indicator_mapping.get(indicator, indicator)

                    data = web.DataReader(fred_code, 'fred', start_date_dt, end_date_dt)

                    results[indicator] = data

                    logger.info(f"Macroeconomic indicator loaded {indicator} ({fred_code})")
                except Exception as e:
                    logger.error(f"Error loading indicator {indicator}: {e}")
                    results[indicator] = pd.DataFrame()

            return results
        except ImportError:
            logger.error("pandas_datareader is not installed. Install it with pip install pandas-datareader")
            return {indicator: pd.DataFrame() for indicator in indicators}
        except Exception as e:
            logger.error(f"Error while getting macroeconomic indicators: {e}")
            return {indicator: pd.DataFrame() for indicator in indicators}

    def get_etf_constituents(self, etf_ticker: str) -> List[Dict]:
        """
        Getting the ETF composition

        Args:
            etf_ticker: ETF ticker

        Returns:
            List of ETF components with weights
        """
        # Cache results for popular ETFs
        etf_cache_file = self.cache_dir / f"{etf_ticker}_constituents.json"

        # Checking the cache
        if etf_cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(etf_cache_file.stat().st_mtime)
            if file_age.days < 30:  # Updated once a month
                try:
                    with open(etf_cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cache for ETF {etf_ticker}: {e}")

        constituents = []

        try:
            import yfinance as yf
            import requests
            from bs4 import BeautifulSoup

            # Trying to get the composition via Yahoo Finance
            etf_info = yf.Ticker(etf_ticker)
            holdings = etf_info.holdings

            # If you managed to get data via yfinance
            if holdings is not None and hasattr(holdings, 'to_dict'):
                top_holdings = holdings.get('holdings', [])

                for i, (symbol, data) in enumerate(top_holdings.items()):
                    if i >= 100:
                        break

                    weight = data.get('percent_of_fund', 0)
                    constituent = {
                        'ticker': symbol,
                        'name': data.get('name', ''),
                        'weight': weight / 100 if weight else 0,
                        'sector': data.get('sector', '')
                    }
                    constituents.append(constituent)

            # If yfinance didn't work, let's try web scraping
            if not constituents:

                if etf_ticker.upper() in ['SPY', 'VOO', 'QQQ', 'VTI', 'IWM']:
                    url = f"https://www.etf.com/{etf_ticker}"
                    headers = {'User-Agent': 'Mozilla/5.0'}

                    response = requests.get(url, headers=headers)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')

                        tables = soup.find_all('table')
                        for table in tables:
                            if 'holdings' in table.get('class', []):
                                rows = table.find_all('tr')
                                for row in rows[1:]:
                                    cells = row.find_all('td')
                                    if len(cells) >= 3:
                                        ticker = cells[0].text.strip()
                                        name = cells[1].text.strip()
                                        weight_text = cells[2].text.strip().replace('%', '')

                                        try:
                                            weight = float(weight_text) / 100
                                        except ValueError:
                                            weight = 0

                                        constituents.append({
                                            'ticker': ticker,
                                            'name': name,
                                            'weight': weight
                                        })

            if constituents:
                with open(etf_cache_file, 'w') as f:
                    json.dump(constituents, f)

            return constituents
        except Exception as e:
            logger.error(f"Error getting ETF composition {etf_ticker}: {e}")
            return []

    def get_sector_performance(self) -> pd.DataFrame:
        """
        Obtaining data on the performance of various market sectors

        Returns:
            DataFrame with sector returns
        """

        cache_file = self.cache_dir / "sector_performance.pkl"


        if cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 1:
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load performance sector cache: {e}")

        try:
            # List of ETF sectors
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Consumer Discretionary': 'XLY',
                'Consumer Staples': 'XLP',
                'Energy': 'XLE',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Materials': 'XLB',
                'Industrials': 'XLI',
                'Communication Services': 'XLC'
            }

            # We define intervals for calculating profitability
            periods = {
                '1D': timedelta(days=1),
                '1W': timedelta(weeks=1),
                '1M': timedelta(days=30),
                '3M': timedelta(days=90),
                'YTD': timedelta(days=(datetime.now() - datetime(datetime.now().year, 1, 1)).days),
                '1Y': timedelta(days=365)
            }

            # Get historical data for all ETF sectors
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365 + 10)).strftime('%Y-%m-%d')

            etf_data = self.get_batch_data(list(sector_etfs.values()), start_date, end_date)

            # Create a DataFrame with the results
            result_data = []

            for sector_name, ticker in sector_etfs.items():
                if ticker in etf_data and not etf_data[ticker].empty:
                    price_data = etf_data[ticker]

                    price_col = 'Adj Close' if 'Adj Close' in price_data.columns else 'Close'

                    latest_price = price_data[price_col].iloc[-1]

                    returns = {}
                    for period_name, period_delta in periods.items():
                        start_idx = price_data.index[-1] - period_delta
                        historical_data = price_data[price_data.index >= start_idx]

                        if not historical_data.empty and len(historical_data) > 1:
                            start_price = historical_data[price_col].iloc[0]
                            returns[period_name] = (latest_price / start_price - 1) * 100
                        else:
                            returns[period_name] = None

                    # Add information to the result
                    sector_info = {
                        'Sector': sector_name,
                        'Ticker': ticker,
                        'Latest Price': latest_price
                    }
                    sector_info.update(returns)

                    result_data.append(sector_info)

            result_df = pd.DataFrame(result_data)

            with open(cache_file, 'wb') as f:
                pickle.dump(result_df, f)

            return result_df
        except Exception as e:
            logger.error(f"Error getting sector performance: {e}")
            return pd.DataFrame()

    # Methods for specific data providers

    def _fetch_yfinance(self, ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
        """Getting data via yfinance"""
        try:
            import yfinance as yf

            original_ticker = ticker
            corrected_ticker = ticker.replace('.', '-') if '.' in ticker else ticker

            if corrected_ticker != original_ticker:
                logger.info(f"Use corrected ticker {corrected_ticker} to query {original_ticker}")

            # Increase the API call counter
            self.api_call_counts['yfinance'] += 1

            try:

                ticker_obj = yf.Ticker(corrected_ticker)
                data = ticker_obj.history(start=start_date, end=end_date, interval=interval)

                if data is None or data.empty:

                    data = yf.download(
                        corrected_ticker,
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        progress=False,
                        show_errors=False
                    )

                if data is None or data.empty:
                    logger.warning(
                        f"No data found for {original_ticker} (query as {corrected_ticker}) via yfinance")
                    return pd.DataFrame()

                # Check and adjust index if it is not DatetimeIndex
                if not isinstance(data.index, pd.DatetimeIndex):
                    try:
                        data.index = pd.to_datetime(data.index)
                    except Exception as e:
                        logger.warning(f"Failed to convert index to DatetimeIndex: {e}")

                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_columns:
                    if col not in data.columns:
                        if col == 'Volume' and 'volume' in data.columns:
                            data['Volume'] = data['volume']
                        else:
                            data[col] = np.nan

                if 'Adj Close' not in data.columns:
                    data['Adj Close'] = data['Close']

                return data

            except Exception as e:
                logger.error(
                    f"Error retrieving data via yfinance for {original_ticker} (request as {corrected_ticker}): {e}")
                return pd.DataFrame()
        except ImportError:
            logger.error("yfinance is not installed. Install it with pip install yfinance")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error retrieving data via yfinance for {ticker}: {e}")
            return pd.DataFrame()

    def _fetch_alpha_vantage(self, ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
        """Getting data via Alpha Vantage API"""
        if not self.api_keys['alpha_vantage']:
            logger.error("API key for Alpha Vantage not installed")
            return pd.DataFrame()

        if self.api_call_counts['alpha_vantage'] >= self.api_limits['alpha_vantage']:
            logger.warning("Alpha Vantage API call limit reached")
            return pd.DataFrame()

        original_ticker = ticker
        corrected_ticker = ticker.replace('.', '-') if '.' in ticker else ticker

        if corrected_ticker != original_ticker:
            logger.info(
                f"Use corrected ticker {corrected_ticker} to query Alpha Vantage {original_ticker}")

        interval_map = {
            '1d': 'daily',
            '1wk': 'weekly',
            '1mo': 'monthly'
        }

        function = f"TIME_SERIES_{interval_map.get(interval, 'daily').upper()}"

        try:
            # Forming URL
            base_url = "https://www.alphavantage.co/query"
            params = {
                "function": function,
                "symbol": corrected_ticker,
                "apikey": self.api_keys['alpha_vantage'],
                "outputsize": "full"
            }

            self.api_call_counts['alpha_vantage'] += 1


            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()

            if "Error Message" in data:
                logger.error(
                    f"Alpha Vantage API returned an error for {original_ticker} (queried as {corrected_ticker}): {data['Error Message']}")
                return pd.DataFrame()

            # Determine the time series key depending on the function
            time_series_key = next((k for k in data.keys() if k.startswith("Time Series")), None)

            if not time_series_key:
                logger.error(f"Unexpected Alpha Vantage response format for {original_ticker}: {data.keys()}")
                return pd.DataFrame()

            time_series_data = data[time_series_key]

            df = pd.DataFrame.from_dict(time_series_data, orient='index')

            column_map = {
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. adjusted close': 'Adj Close',
                '5. volume': 'Volume',
                '6. volume': 'Volume'
            }

            df = df.rename(columns=column_map)

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.index = pd.to_datetime(df.index)

            df = df.sort_index()

            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Error requesting Alpha Vantage API for {original_ticker}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing Alpha Vantage data for {original_ticker}: {e}")
            return pd.DataFrame()

    def _get_yfinance_company_info(self, ticker: str) -> Dict:
        """Getting information about a company through yfinance"""
        try:
            import yfinance as yf

            # Check if the ticker contains a dot and create a corrected version for the API
            original_ticker = ticker
            corrected_ticker = ticker.replace('.', '-') if '.' in ticker else ticker

            if corrected_ticker != original_ticker:
                logger.info(
                    f"Use the corrected ticker {corrected_ticker} to query information about the company {original_ticker}")

            self.api_call_counts['yfinance'] += 1

            ticker_obj = yf.Ticker(corrected_ticker)
            info = ticker_obj.info

            normalized_info = {
                'symbol': original_ticker,
                'name': info.get('longName', info.get('shortName', original_ticker)),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'country': info.get('country', 'N/A'),
                'exchange': info.get('exchange', 'N/A'),
                'currency': info.get('currency', 'USD'),
                'market_cap': info.get('marketCap', None),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'pb_ratio': info.get('priceToBook', None),
                'dividend_yield': info.get('dividendYield', None) if info.get('dividendYield') else None,
                'beta': info.get('beta', None),
                'description': info.get('longBusinessSummary', 'N/A'),
                'website': info.get('website', 'N/A'),
                'employees': info.get('fullTimeEmployees', None),
                'logo_url': info.get('logo_url', None),
                'type': self._determine_asset_type(info),
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }

            for metric in ['returnOnEquity', 'returnOnAssets', 'profitMargins', 'operatingMargins', 'grossMargins']:
                if metric in info:
                    normalized_info[self._camel_to_snake(metric)] = info[metric]

            return normalized_info
        except ImportError:
            logger.error("yfinance is not installed. Install it with pip install yfinance")
            return {}
        except Exception as e:
            logger.error(f"Error when getting company information via yfinance for {original_ticker}: {e}")
            return {}

    def _get_alpha_vantage_company_info(self, ticker: str) -> Dict:
        """Obtaining information about a company through Alpha Vantage"""
        if not self.api_keys['alpha_vantage']:
            logger.error("API key for Alpha Vantage not installed")
            return {}

        # Checking API Limits
        if self.api_call_counts['alpha_vantage'] >= self.api_limits['alpha_vantage']:
            logger.warning("Alpha Vantage API call limit reached")
            return {}

        # Check if the ticker contains a dot and create a corrected version for the API
        original_ticker = ticker
        corrected_ticker = ticker.replace('.', '-') if '.' in ticker else ticker

        if corrected_ticker != original_ticker:
            logger.info(
                f"We use the corrected ticker {corrected_ticker} to query Alpha Vantage for company {original_ticker}")

        try:

            base_url = "https://www.alphavantage.co/query"
            params = {
                "function": "OVERVIEW",
                "symbol": corrected_ticker,
                "apikey": self.api_keys['alpha_vantage']
            }

            self.api_call_counts['alpha_vantage'] += 1

            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()

            if not data or ("Error Message" in data) or len(data.keys()) <= 1:
                logger.warning(f"Alpha Vantage did not return data for {original_ticker} (queried as {corrected_ticker})")
                return {}

            # Transform and normalize data
            normalized_info = {
                'symbol': original_ticker,
                'name': data.get('Name', original_ticker),
                'sector': data.get('Sector', 'N/A'),
                'industry': data.get('Industry', 'N/A'),
                'country': data.get('Country', 'N/A'),
                'exchange': data.get('Exchange', 'N/A'),
                'currency': data.get('Currency', 'USD'),
                'market_cap': self._safe_convert(data.get('MarketCapitalization'), float),
                'pe_ratio': self._safe_convert(data.get('PERatio'), float),
                'forward_pe': self._safe_convert(data.get('ForwardPE'), float),
                'pb_ratio': self._safe_convert(data.get('PriceToBookRatio'), float),
                'dividend_yield': self._safe_convert(data.get('DividendYield'), float),
                'beta': self._safe_convert(data.get('Beta'), float),
                'description': data.get('Description', 'N/A'),
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }

            for key in ['ReturnOnEquityTTM', 'ReturnOnAssetsTTM', 'ProfitMargin', 'OperatingMarginTTM',
                        'GrossProfitTTM']:
                if key in data:
                    normalized_key = self._camel_to_snake(key.replace('TTM', ''))
                    normalized_info[normalized_key] = self._safe_convert(data[key], float)

            return normalized_info
        except requests.exceptions.RequestException as e:
            logger.error(f"Error requesting Alpha Vantage API for company information {original_ticker}: {e} company information")
            return {}
        except Exception as e:
            logger.error(f"Error processing Alpha Vantage information for {original_ticker}: {e}")
            return {}

    def _search_alpha_vantage(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for tickers via Alpha Vantage"""
        if not self.api_keys['alpha_vantage']:
            logger.error("API key for Alpha Vantage not installed")
            return []

        # Checking API Limits
        if self.api_call_counts['alpha_vantage'] >= self.api_limits['alpha_vantage']:
            logger.warning("Alpha Vantage API call limit reached")
            return []

        try:

            base_url = "https://www.alphavantage.co/query"
            params = {
                "function": "SYMBOL_SEARCH",
                "keywords": query,
                "apikey": self.api_keys['alpha_vantage']
            }

            self.api_call_counts['alpha_vantage'] += 1

            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()

            results = []
            if 'bestMatches' in data:
                for match in data['bestMatches'][:limit]:

                    symbol = match.get('1. symbol', '')

                    display_symbol = symbol
                    if '-' in symbol:

                        known_patterns = ['BRK-B', 'BF-B']
                        if symbol in known_patterns or (len(symbol.split('-')) == 2 and len(symbol.split('-')[1]) == 1):
                            display_symbol = symbol.replace('-', '.')
                            logger.info(f"Converted ticker in search results: {symbol} -> {display_symbol}")

                    results.append({
                        'symbol': display_symbol,
                        'original_symbol': symbol,
                        'name': match.get('2. name', ''),
                        'type': match.get('3. type', ''),
                        'region': match.get('4. region', ''),
                        'currency': match.get('8. currency', 'USD'),
                        'exchange': match.get('5. marketOpen', '') + '-' + match.get('6. marketClose', ''),
                        'timezone': match.get('7. timezone', '')
                    })

            return results
        except requests.exceptions.RequestException as e:
            logger.error(f"Error requesting Alpha Vantage API for search: {e}")
            return []
        except Exception as e:
            logger.error(f"Error searching for tickers via Alpha Vantage: {e}")
            return []

    def _search_alternative(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Alternative Method of Finding Tickers via Yahoo Finance API

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of dictionaries with information about found tickers
        """
        try:
            import yfinance as yf
            import json
            import requests

            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount={limit}&newsCount=0"
            headers = {'User-Agent': 'Mozilla/5.0'}

            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                logger.error(f"Error requesting Yahoo Finance API: {response.status_code}")
                return []

            data = response.json()

            results = []
            if 'quotes' in data and data['quotes']:
                for quote in data['quotes'][:limit]:

                    ticker = quote.get('symbol', '')

                    display_ticker = ticker
                    if '-' in ticker:
                        if len(ticker.split('-')) == 2 and len(ticker.split('-')[1]) == 1:
                            display_ticker = ticker.replace('-', '.')

                    results.append({
                        'symbol': display_ticker,
                        'original_symbol': ticker,
                        'name': quote.get('shortname', quote.get('longname', '')),
                        'type': quote.get('quoteType', ''),
                        'region': quote.get('region', 'US'),
                        'currency': quote.get('currency', 'USD'),
                        'exchange': quote.get('exchange', '')
                    })

            return results
        except Exception as e:
            logger.error(f"Error searching for tickers via Yahoo Finance: {e}")
            return []

    def _determine_asset_type(self, info: Dict) -> str:
        """Determine the asset type based on the information"""
        if not info:
            return 'Unknown'

        if 'quoteType' in info:
            quote_type = info['quoteType'].lower()
            if quote_type in ['equity', 'stock']:
                return 'Stock'
            elif quote_type == 'etf':
                return 'ETF'
            elif quote_type == 'index':
                return 'Index'
            elif quote_type in ['cryptocurrency', 'crypto']:
                return 'Crypto'
            elif quote_type == 'mutualfund':
                return 'Mutual Fund'
            else:
                return quote_type.capitalize()

        if 'fundFamily' in info and info['fundFamily']:
            return 'ETF' if 'ETF' in info.get('longName', '') else 'Mutual Fund'

        if 'industry' in info and info['industry']:
            return 'Stock'

        return 'Unknown'

    def _camel_to_snake(self, name: str) -> str:
        """Convert camelCase to snake_case"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _safe_convert(self, value: Any, convert_type) -> Optional[Any]:
        """Safe type conversion with error handling"""
        if value is None:
            return None

        try:
            return convert_type(value)
        except (ValueError, TypeError):
            return None

    def clear_cache(self, tickers: Optional[List[str]] = None):
        """
        Clearing data cache

        Args:
            tickers: List of tickers to clear. If None, clears the entire cache.
        """
        if tickers is None:

            for file in self.cache_dir.glob('*.*'):
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {file}: {e}")
            logger.info("The cache has been completely cleared.")
        else:

            for ticker in tickers:
                for file in self.cache_dir.glob(f"{ticker}_*.*"):
                    try:
                        file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {file}: {e}")
            logger.info(f"Cache cleared for tickers: {tickers}")

    def get_batch_data(self, tickers: List[str], start_date: Optional[str] = None,
                       end_date: Optional[str] = None, provider: str = 'yfinance') -> Dict[str, pd.DataFrame]:
        """
        Getting data for multiple tickers at once

        Args:
            tickers: List of tickers
            start_date: Start date
            end_date: End date
            provider: Data provider

        Returns:
            Dictionary {ticker: DataFrame}
        """
        results = {}

        ticker_mapping = {}
        corrected_tickers = []

        for ticker in tickers:
            if '.' in ticker:

                corrected_ticker = ticker.replace('.', '-')
                ticker_mapping[corrected_ticker] = ticker
                corrected_tickers.append(corrected_ticker)
            else:
                corrected_tickers.append(ticker)
                ticker_mapping[ticker] = ticker

        if ticker_mapping:
            logger.info(f"Tickers have been replaced for data requests: {ticker_mapping}")

        if provider == 'yfinance' and len(corrected_tickers) > 1:
            try:
                import yfinance as yf

                if end_date is None:
                    end_date = datetime.now().strftime('%Y-%m-%d')

                if start_date is None:
                    start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

                tickers_to_download = []
                download_mapping = {}

                for original_ticker in tickers:
                    cache_file = self.cache_dir / f"{original_ticker}_{start_date}_{end_date}_1d_{provider}.pkl"
                    if cache_file.exists():
                        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                        if file_age.days < self.cache_expiry_days:
                            try:
                                with open(cache_file, 'rb') as f:
                                    results[original_ticker] = pickle.load(f)
                                    logger.info(f"Loading {original_ticker} data from cache")
                                    continue
                            except Exception as e:
                                logger.warning(f"Failed to load cache for {original_ticker}: {e}")

                    corrected_ticker = original_ticker.replace('.', '-') if '.' in original_ticker else original_ticker
                    tickers_to_download.append(corrected_ticker)
                    download_mapping[corrected_ticker] = original_ticker

                if tickers_to_download:
                    self.api_call_counts['yfinance'] += 1

                    data = yf.download(
                        tickers_to_download,
                        start=start_date,
                        end=end_date,
                        interval='1d',
                        group_by='ticker',
                        progress=False,
                        show_errors=False
                    )

                    for corrected_ticker, original_ticker in download_mapping.items():
                        if len(tickers_to_download) == 1:

                            ticker_data = data
                        else:

                            ticker_data = data[corrected_ticker].copy() if corrected_ticker in data else pd.DataFrame()

                        if not ticker_data.empty:

                            cache_file = self.cache_dir / f"{original_ticker}_{start_date}_{end_date}_1d_{provider}.pkl"
                            with open(cache_file, 'wb') as f:
                                pickle.dump(ticker_data, f)

                            results[original_ticker] = ticker_data
                            logger.info(
                                f"Loaded and saved data for {original_ticker} (queried as {corrected_ticker})")

                return results
            except ImportError:
                logger.error("yfinance is not installed. Install it with pip install yfinance")
            except Exception as e:
                logger.error(f"Error while loading data in batch: {e}")

        # Sequential download for each ticker
        for original_ticker in tickers:
            try:

                corrected_ticker = original_ticker.replace('.', '-') if '.' in original_ticker else original_ticker

                data = self.get_historical_prices(corrected_ticker, start_date, end_date, provider)

                if not data.empty:

                    results[original_ticker] = data
            except Exception as e:
                logger.error(f"Error loading data for {original_ticker}: {e}")

        return results

    def get_fundamental_data(self, ticker: str, data_type: str = 'income') -> pd.DataFrame:
        """
        Obtaining fundamental financial data

        Args:
            ticker: Company ticker
            data_type: Data type ('income', 'balance', 'cash', 'earnings')

        Returns:
            DataFrame with fundamental data
        """
        cache_file = self.cache_dir / f"{ticker}_fundamental_{data_type}.pkl"

        # Cache check (fundamental data expires after 30 days)
        if cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 30:
                try:
                    with open(cache_file, 'rb') as f:
                        logger.info(f"Loading fundamental data {data_type} for {ticker} from cache")
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load fundamental data cache for {ticker}: {e}")

        try:
            import yfinance as yf

            # Increase the API call counter
            self.api_call_counts['yfinance'] += 1

            ticker_obj = yf.Ticker(ticker)

            if data_type == 'income':
                df = ticker_obj.income_stmt
            elif data_type == 'balance':
                df = ticker_obj.balance_sheet
            elif data_type == 'cash':
                df = ticker_obj.cashflow
            elif data_type == 'earnings':
                df = ticker_obj.earnings
            else:
                raise ValueError(f"Unsupported fundamental data type: {data_type}")

            if df is None or df.empty:
                logger.warning(f"No fundamental data {data_type} found for {ticker}")
                return pd.DataFrame()

            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)

            return df
        except ImportError:
            logger.error("yfinance is not installed. Install it with pip install yfinance")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error in getting fundamental data for {ticker}: {e}")
            return pd.DataFrame()


# Class for working with portfolio data
class PortfolioDataManager:
    """
        Class for working with portfolio data.
        Provides loading, saving and exchanging portfolio data.
    """

    def __init__(self, data_fetcher: DataFetcher, storage_dir: str = './portfolios'):
        """
        Initializing the Portfolio Data Manager

        Args:
            data_fetcher: DataFetcher instance for loading market data
            storage_dir: Directory for storing portfolios
        """
        self.data_fetcher = data_fetcher
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized PortfolioDataManager with storage in {self.storage_dir}")

    def save_portfolio(self, portfolio_data: Dict, filename: Optional[str] = None) -> str:
        """
        Saving portfolio data to a file

        Args:
           portfolio_data: Dictionary with portfolio data
            filename: File name (without extension). If None, uses portfolio_data['name']

        Returns:
            Path to the saved file
        """
        if 'name' not in portfolio_data:
            raise ValueError("Portfolio data is missing a required field 'name'")

        # Defining a file name
        if filename is None:
            filename = self._sanitize_filename(portfolio_data['name'])

        file_path = self.storage_dir / f"{filename}.json"

        try:

            if 'last_updated' not in portfolio_data:
                portfolio_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(portfolio_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Portfolio '{portfolio_data['name']}' saved to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
            raise

    def load_portfolio(self, filename: str) -> Dict:
        """
        Loading portfolio data from file

        Args:
            filename: File name (with or without extension)

        Returns:
            Portfolio data dictionary
        """

        if not filename.endswith('.json'):
            filename += '.json'

        file_path = self.storage_dir / filename

        if not file_path.exists():

            potential_files = list(self.storage_dir.glob('*.json'))

            normalized_name = filename.lower().replace(' ', '').replace('&', 'and')

            for potential_file in potential_files:

                potential_normalized = potential_file.name.lower().replace(' ', '').replace('&', 'and')

                if normalized_name == potential_normalized:
                    file_path = potential_file
                    logger.info(f"Alternative file name found: {potential_file}")
                    break

                if len(normalized_name) > 10 and normalized_name[:10] == potential_normalized[:10]:
                    file_path = potential_file
                    logger.info(f"Similar file name found: {potential_file}")
                    break

        if not file_path.exists():

            for json_file in self.storage_dir.glob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        if 'name' in data and data['name'] == filename.replace('.json', ''):
                            file_path = json_file
                            logger.info(f"Найден файл по имени портфеля: {json_file}")
                            break
                except Exception:

                    continue

        if not file_path.exists():
            raise FileNotFoundError(f"Portfolio file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                portfolio_data = json.load(f)

            logger.info(f"Portfolio loaded from {file_path}")
            return portfolio_data
        except Exception as e:
            logger.error(f"Error loading portfolio from {file_path}: {e}")
            raise

    def list_portfolios(self) -> List[Dict]:
        """
        Getting a list of all available portfolios

        Returns:
            List of dictionaries with basic information about portfolios
        """
        portfolios = []

        try:
            for file_path in self.storage_dir.glob('*.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    portfolio_info = {
                        'name': data.get('name', file_path.stem),
                        'filename': file_path.name,
                        'last_updated': data.get('last_updated', 'Unknown'),
                        'asset_count': len(data.get('assets', [])),
                        'description': data.get('description', ''),
                        'tags': data.get('tags', [])
                    }

                    portfolios.append(portfolio_info)
                except Exception as e:
                    logger.warning(f"Failed to read portfolio data from {file_path}: {e}")

            return sorted(portfolios, key=lambda x: x['name'])
        except Exception as e:
            logger.error(f"Error getting list of portfolios: {e}")
            return []

    def delete_portfolio(self, filename: str) -> bool:
        """
        Deleting a portfolio

        Args:
            filename: File name (with or without extension)

        Returns:
            True if the deletion was successful, otherwise False
        """

        if not filename.endswith('.json'):
            filename += '.json'

        file_path = self.storage_dir / filename

        if not file_path.exists():
            logger.warning(f"Briefcase file not found to delete: {file_path}")
            return False

        try:
            file_path.unlink()
            logger.info(f"Portfolio removed: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting portfolio {file_path}: {e}")
            return False

    def import_from_csv(self, file_path: str, portfolio_name: Optional[str] = None) -> Dict:
        """
        Import portfolio from CSV file

        Args:
            file_path: Path to CSV file
            portfolio_name: Name of portfolio (if None, file name is used)

        Returns:
            Portfolio data dictionary
        """
        try:
            df = pd.read_csv(file_path)

            # Checking mandatory columns
            required_columns = ['ticker']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"The CSV file is missing required columns: {missing_columns}")

            # Preparing a portfolio name
            if portfolio_name is None:
                file_name = Path(file_path).stem
                portfolio_name = file_name

            # Creating a portfolio structure
            portfolio_data = {
                'name': portfolio_name,
                'description': f"Imported from {Path(file_path).name}",
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'assets': []
            }

            for _, row in df.iterrows():
                asset = {'ticker': row['ticker']}

                optional_fields = ['weight', 'quantity', 'purchase_price', 'purchase_date',
                                   'sector', 'asset_class', 'region', 'currency']

                for field in optional_fields:
                    if field in df.columns and not pd.isna(row[field]):
                        asset[field] = row[field]

                if 'weight' not in asset and 'quantity' in asset:
                    asset['weight'] = 0.0

                portfolio_data['assets'].append(asset)

            if all('quantity' in asset for asset in portfolio_data['assets']) and \
                    not all('weight' in asset for asset in portfolio_data['assets']):
                self._calculate_weights_from_quantities(portfolio_data)

            self._normalize_weights(portfolio_data)

            self._enrich_portfolio_data(portfolio_data)

            logger.info(f"Portfolio '{portfolio_name}' imported from {file_path}")
            return portfolio_data
        except Exception as e:
            logger.error(f"Error importing portfolio from CSV {file_path}: {e}")
            raise

    def export_to_csv(self, portfolio_data: Dict, file_path: Optional[str] = None) -> str:
        """
        Export portfolio to CSV file

        Args:
            portfolio_data: Dictionary with portfolio data
            file_path: Path to save to (if None, created based on portfolio name)

        Returns:
            Path to the saved file
        """
        if 'name' not in portfolio_data or 'assets' not in portfolio_data:
            raise ValueError("Invalid portfolio data format")

        # Defining the path to save
        if file_path is None:
            filename = self._sanitize_filename(portfolio_data['name'])
            file_path = self.storage_dir / f"{filename}.csv"

        try:
            # Create a DataFrame from assets
            assets_data = []

            for asset in portfolio_data['assets']:
                asset_row = {'ticker': asset['ticker']}

                # Add all available fields
                for key, value in asset.items():
                    if key != 'ticker':
                        asset_row[key] = value

                assets_data.append(asset_row)

            # Create a DataFrame and save it
            df = pd.DataFrame(assets_data)
            df.to_csv(file_path, index=False)

            logger.info(f"Portfolio '{portfolio_data['name']}' exported to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Error exporting portfolio to CSV: {e}")
            raise

    def parse_ticker_weights_text(self, text: str) -> List[Dict]:
        """
        Parsing a text list of tickers and scales

        Args:
            text: Text in the format "TICKER:weight, TICKER:weight" or "TICKER weight, TICKER weight"

        Returns:
            List of dictionaries with asset data
        """
        assets = []

        text = text.strip()

        lines = re.split(r'[,\n]+', text)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            colon_match = re.match(r'^([A-Za-z0-9.-]+):([0-9.]+)$', line)

            space_match = re.match(r'^([A-Za-z0-9.-]+)\s+([0-9.]+)$', line)

            ticker_only_match = re.match(r'^([A-Za-z0-9.-]+)$', line)

            if colon_match:
                ticker, weight = colon_match.groups()
                assets.append({
                    'ticker': ticker.strip().upper(),
                    'weight': float(weight.strip())
                })
            elif space_match:
                ticker, weight = space_match.groups()
                assets.append({
                    'ticker': ticker.strip().upper(),
                    'weight': float(weight.strip())
                })
            elif ticker_only_match:
                ticker = ticker_only_match.group(1)
                assets.append({
                    'ticker': ticker.strip().upper(),
                    'weight': 0.0
                })

        if assets and all(asset['weight'] == 0.0 for asset in assets):
            equal_weight = 1.0 / len(assets)
            for asset in assets:
                asset['weight'] = equal_weight

        return assets

    def create_portfolio_from_text(self, text: str, portfolio_name: str, description: str = "") -> Dict:
        """
        Creating a portfolio from a text list of tickers

        Args:
            text: Text with tickers and weights
            portfolio_name: Portfolio name
            description: Portfolio description

        Returns:
            Portfolio data dictionary
        """
        assets = self.parse_ticker_weights_text(text)

        if not assets:
            raise ValueError("Could not recognize any ticker in the text")

        portfolio_data = {
            'name': portfolio_name,
            'description': description,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'assets': assets
        }

        self._normalize_weights(portfolio_data)

        valid_tickers, invalid_tickers = self.data_fetcher.validate_tickers([a['ticker'] for a in assets])

        if invalid_tickers:
            logger.warning(f"Invalid tickers found: {invalid_tickers}")

            portfolio_data['assets'] = [a for a in portfolio_data['assets']
                                        if a['ticker'] in valid_tickers]

            self._normalize_weights(portfolio_data)

        self._enrich_portfolio_data(portfolio_data)

        logger.info(f"Portfolio '{portfolio_name}' created with {len(portfolio_data['assets'])} assets")
        return portfolio_data

    def update_portfolio_prices(self, portfolio_data: Dict) -> Dict:
        """
        Updating current prices of assets in the portfolio

        Args:
            portfolio_data: Dictionary with portfolio data

        Returns:
            Updated dictionary with portfolio data
        """
        if 'assets' not in portfolio_data or not portfolio_data['assets']:
            return portfolio_data

        tickers = [asset['ticker'] for asset in portfolio_data['assets']]

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')

        price_data = self.data_fetcher.get_batch_data(tickers, start_date, end_date)

        for asset in portfolio_data['assets']:
            ticker = asset['ticker']

            if ticker in price_data and not price_data[ticker].empty:

                latest_data = price_data[ticker].iloc[-1]

                if 'Adj Close' in latest_data:
                    current_price = latest_data['Adj Close']
                elif 'Close' in latest_data:
                    current_price = latest_data['Close']
                else:
                    continue

                asset['current_price'] = current_price
                asset['price_date'] = price_data[ticker].index[-1].strftime('%Y-%m-%d')

                if 'purchase_price' in asset and asset['purchase_price']:
                    asset['price_change_pct'] = ((current_price / float(asset['purchase_price'])) - 1) * 100

        portfolio_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"Asset prices for the portfolio have been updated '{portfolio_data['name']}'")
        return portfolio_data

    def _sanitize_filename(self, filename: str) -> str:
        """Cleaning a file name from invalid characters"""

        filename = re.sub(r'[\\/*?:"<>|%\\\\\s]', '_', filename)

        filename = filename.replace('&', 'and')

        filename = filename.replace('\\', '_').replace('/', '_').replace(':', '_')

        if len(filename) > 200:
            filename = filename[:197] + '...'

        return filename

    def _normalize_weights(self, portfolio_data: Dict) -> None:
        """Normalize the weights of assets in a portfolio so that the sum is equal to 1.0"""
        if 'assets' not in portfolio_data or not portfolio_data['assets']:
            return

        total_weight = sum(asset.get('weight', 0.0) for asset in portfolio_data['assets'])

        if total_weight > 0:
            for asset in portfolio_data['assets']:
                if 'weight' in asset:
                    asset['weight'] = asset['weight'] / total_weight

    def _calculate_weights_from_quantities(self, portfolio_data: Dict) -> None:
        """Calculate weights based on quantities and current prices"""
        if 'assets' not in portfolio_data or not portfolio_data['assets']:
            return

        tickers = [asset['ticker'] for asset in portfolio_data['assets'] if 'quantity' in asset]
        price_data = self.data_fetcher.get_batch_data(tickers)

        total_value = 0.0

        for asset in portfolio_data['assets']:
            if 'quantity' in asset and asset['ticker'] in price_data:
                df = price_data[asset['ticker']]

                if not df.empty:
                    if 'Adj Close' in df.columns:
                        price = df['Adj Close'].iloc[-1]
                    elif 'Close' in df.columns:
                        price = df['Close'].iloc[-1]
                    else:
                        continue

                    position_value = float(asset['quantity']) * price
                    asset['position_value'] = position_value
                    total_value += position_value

        if total_value > 0:
            for asset in portfolio_data['assets']:
                if 'position_value' in asset:
                    asset['weight'] = asset['position_value'] / total_value

                    del asset['position_value']

    def _enrich_portfolio_data(self, portfolio_data: Dict) -> None:
        """Enriching portfolio data with additional information"""
        if 'assets' not in portfolio_data or not portfolio_data['assets']:
            return

        for asset in portfolio_data['assets']:
            try:
                info = self.data_fetcher.get_company_info(asset['ticker'])

                if info:

                    for field in ['name', 'sector', 'industry', 'asset_class', 'currency']:
                        if field in info and field not in asset:
                            asset[field] = info[field]

                    if 'asset_class' not in asset and 'type' in info:
                        asset['asset_class'] = info['type']
            except Exception as e:
                logger.warning(f"Unable to retrieve information for {asset['ticker']}: {e}")

        self.update_portfolio_prices(portfolio_data)