import os
import time
import logging
import pickle
import json
import re  # Обязательно включаем на верхнем уровне модуля
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional, Union, Any

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_fetcher')


class DataFetcher:
    """
    Основной класс для получения финансовых данных из различных источников.
    Поддерживает кеширование данных для оптимизации производительности.
    """

    def __init__(self, cache_dir: str = './cache', cache_expiry_days: int = 1):
        """
        Инициализация загрузчика данных

        Args:
            cache_dir: Директория для кеширования данных
            cache_expiry_days: Срок действия кеша в днях
        """
        self.cache_dir = Path(cache_dir)
        self.cache_expiry_days = cache_expiry_days
        self.api_call_counts = {'yfinance': 0, 'alpha_vantage': 0}
        self.api_limits = {'alpha_vantage': 500}  # Лимиты API-вызовов

        # Создаем директорию кеша, если она не существует
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Загружаем API ключи из конфигурационного файла или переменных окружения
        self.api_keys = {
            'alpha_vantage': os.environ.get('ALPHA_VANTAGE_API_KEY', '')
        }

        # Словарь поддерживаемых провайдеров данных
        self.providers = {
            'yfinance': self._fetch_yfinance,
            'alpha_vantage': self._fetch_alpha_vantage
        }

        # Предзагрузка данных бенчмарков
        self.benchmark_data = {}
        self.benchmark_tickers = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']
        self.preload_benchmarks()

        logger.info(f"Инициализирован DataFetcher с кешем в {self.cache_dir}")

        # Предзагрузка данных бенчмарков
        self.benchmark_data = {}
        self.benchmark_tickers = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']
        self.preload_benchmarks()

    # Место добавления: новый метод в классе DataFetcher

    def preload_benchmarks(self):
        """Предзагрузка данных основных бенчмарков с большим историческим периодом"""
        start_date = '1990-01-01'  # Достаточно давно для большинства бенчмарков
        end_date = datetime.now().strftime('%Y-%m-%d')

        for ticker in self.benchmark_tickers:
            try:
                data = self.get_historical_prices(ticker, start_date, end_date, force_refresh=False)
                if not data.empty:
                    self.benchmark_data[ticker] = data
                    logger.info(f"Предзагружены исторические данные для бенчмарка {ticker}")
            except Exception as e:
                logger.warning(f"Не удалось предзагрузить данные для бенчмарка {ticker}: {e}")

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
        Получение исторических цен для указанного тикера
        Args:
            ticker: Тикер акции/ETF
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'
            provider: Провайдер данных ('yfinance', 'alpha_vantage')
            interval: Интервал данных ('1d', '1wk', '1mo')
            force_refresh: Принудительное обновление данных
        Returns:
            DataFrame с историческими ценами
        """
        # Проверяем, содержит ли тикер точку, и создаем скорректированную версию для API
        original_ticker = ticker
        corrected_ticker = ticker.replace('.', '-') if '.' in ticker else ticker

        if corrected_ticker != original_ticker:
            logger.info(f"Используем скорректированный тикер {corrected_ticker} для запроса {original_ticker}")

        # Используем оригинальный тикер для кеша
        cache_key = original_ticker

        # Установка значений по умолчанию для дат
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date is None:
            # По умолчанию 5 лет исторических данных
            start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

        # Проверка срока годности кеша
        cache_file = self.cache_dir / f"{cache_key}_{start_date}_{end_date}_{interval}_{provider}.pkl"

        if not force_refresh and cache_file.exists():
            # Проверка возраста кеша
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < self.cache_expiry_days:
                try:
                    with open(cache_file, 'rb') as f:
                        logger.info(f"Загрузка данных {original_ticker} из кеша")
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Не удалось загрузить кеш для {original_ticker}: {e}")

        # Если кеш отсутствует или устарел, получаем новые данные
        if provider in self.providers:
            try:
                # Получение данных от провайдера, используя скорректированный тикер
                data = self.providers[provider](corrected_ticker, start_date, end_date, interval)

                # Сохраняем в кеш, если получены данные
                if data is not None and not data.empty:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(data, f)
                    logger.info(f"Сохранены данные {original_ticker} в кеш")

                return data
            except Exception as e:
                logger.error(f"Ошибка получения данных для {original_ticker} от {provider}: {e}")
                # Пробуем альтернативный провайдер, если текущий не работает
                fallback_provider = next((p for p in self.providers.keys() if p != provider), None)
                if fallback_provider:
                    logger.info(f"Пробуем альтернативный провайдер: {fallback_provider}")
                    return self.get_historical_prices(
                        ticker, start_date, end_date, fallback_provider, interval, force_refresh
                    )
        else:
            raise ValueError(f"Неподдерживаемый провайдер данных: {provider}")

        # Если не удалось получить данные
        return pd.DataFrame()

    def get_company_info(self, ticker: str, provider: str = 'yfinance') -> Dict:
        """
        Получение информации о компании

        Args:
            ticker: Тикер акции/ETF
            provider: Провайдер данных

        Returns:
            Словарь с информацией о компании
        """
        cache_file = self.cache_dir / f"{ticker}_info_{provider}.json"

        # Проверка кеша (срок годности информации о компании - 7 дней)
        if cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 7:  # Информация о компании обновляется реже
                try:
                    with open(cache_file, 'r') as f:
                        logger.info(f"Загрузка информации о {ticker} из кеша")
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Не удалось загрузить кеш информации для {ticker}: {e}")

        info = {}

        try:
            if provider == 'yfinance':
                info = self._get_yfinance_company_info(ticker)
            elif provider == 'alpha_vantage':
                info = self._get_alpha_vantage_company_info(ticker)
            else:
                raise ValueError(f"Неподдерживаемый провайдер для информации о компании: {provider}")

            # Сохраняем в кеш
            if info:
                with open(cache_file, 'w') as f:
                    json.dump(info, f)
                logger.info(f"Сохранена информация о {ticker} в кеш")

            return info
        except Exception as e:
            logger.error(f"Ошибка получения информации о {ticker}: {e}")

            # Пробуем альтернативный провайдер
            if provider != 'yfinance':
                logger.info("Пробуем получить информацию через yfinance")
                return self.get_company_info(ticker, 'yfinance')

            return {}

    def search_tickers(self, query: str, limit: int = 10, provider: str = 'alpha_vantage') -> List[Dict]:
        """
        Поиск тикеров по запросу

        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            provider: Провайдер данных

        Returns:
            Список словарей с информацией о найденных тикерах
        """
        cache_key = f"search_{query.lower()}_{provider}.json"
        cache_file = self.cache_dir / cache_key

        # Проверка кеша (срок годности поисковых результатов - 3 дня)
        if cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 3:
                try:
                    with open(cache_file, 'r') as f:
                        results = json.load(f)
                        logger.info(f"Загрузка результатов поиска '{query}' из кеша")
                        return results[:limit]
                except Exception as e:
                    logger.warning(f"Не удалось загрузить кеш поиска: {e}")

        results = []

        try:
            if provider == 'alpha_vantage':
                results = self._search_alpha_vantage(query, limit)
            elif provider == 'yfinance':
                # yfinance не имеет прямого API для поиска, используем альтернативу
                results = self._search_alternative(query, limit)
            else:
                raise ValueError(f"Неподдерживаемый провайдер для поиска: {provider}")

            # Сохраняем в кеш
            if results:
                with open(cache_file, 'w') as f:
                    json.dump(results, f)
                logger.info(f"Сохранены результаты поиска '{query}' в кеш")

            return results[:limit]
        except Exception as e:
            logger.error(f"Ошибка поиска тикеров по запросу '{query}': {e}")

            # Пробуем альтернативный провайдер
            if provider != 'alpha_vantage' and self.api_keys['alpha_vantage']:
                return self.search_tickers(query, limit, 'alpha_vantage')

            return []

    def validate_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Проверка валидности списка тикеров

        Args:
            tickers: Список тикеров для проверки

        Returns:
            Кортеж (valid_tickers, invalid_tickers)
        """
        # Список популярных тикеров, которые считаем валидными по умолчанию
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
                # Пытаемся получить базовую информацию о тикере
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
        Получение макроэкономических индикаторов

        Args:
            indicators: Список индикаторов
            start_date: Начальная дата
            end_date: Конечная дата

        Returns:
            Словарь {indicator: DataFrame}
        """
        results = {}

        # Отображаем коды индикаторов на их FRED-коды
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

            # Настройка дат
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            if start_date is None:
                start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

            # Конвертируем строки дат в datetime
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)

            # Получаем данные для каждого индикатора
            for indicator in indicators:
                try:
                    # Преобразуем индикатор в код FRED
                    fred_code = indicator_mapping.get(indicator, indicator)

                    # Загружаем данные из FRED
                    data = web.DataReader(fred_code, 'fred', start_date_dt, end_date_dt)

                    # Сохраняем в результат
                    results[indicator] = data

                    logger.info(f"Загружен макроэкономический индикатор {indicator} ({fred_code})")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке индикатора {indicator}: {e}")
                    results[indicator] = pd.DataFrame()

            return results
        except ImportError:
            logger.error("pandas_datareader не установлен. Установите его с помощью pip install pandas-datareader")
            return {indicator: pd.DataFrame() for indicator in indicators}
        except Exception as e:
            logger.error(f"Ошибка при получении макроэкономических индикаторов: {e}")
            return {indicator: pd.DataFrame() for indicator in indicators}

    def get_etf_constituents(self, etf_ticker: str) -> List[Dict]:
        """
        Получение состава ETF

        Args:
            etf_ticker: Тикер ETF

        Returns:
            Список компонентов ETF с весами
        """
        # Кешируем результаты для популярных ETF
        etf_cache_file = self.cache_dir / f"{etf_ticker}_constituents.json"

        # Проверяем кеш
        if etf_cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(etf_cache_file.stat().st_mtime)
            if file_age.days < 30:  # Обновляем раз в месяц
                try:
                    with open(etf_cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Не удалось загрузить кеш для ETF {etf_ticker}: {e}")

        constituents = []

        try:
            import yfinance as yf
            import requests
            from bs4 import BeautifulSoup

            # Пробуем получить состав через Yahoo Finance
            etf_info = yf.Ticker(etf_ticker)
            holdings = etf_info.holdings

            # Если удалось получить данные через yfinance
            if holdings is not None and hasattr(holdings, 'to_dict'):
                top_holdings = holdings.get('holdings', [])

                for i, (symbol, data) in enumerate(top_holdings.items()):
                    if i >= 100:  # Ограничиваем количество компонентов
                        break

                    weight = data.get('percent_of_fund', 0)
                    constituent = {
                        'ticker': symbol,
                        'name': data.get('name', ''),
                        'weight': weight / 100 if weight else 0,
                        'sector': data.get('sector', '')
                    }
                    constituents.append(constituent)

            # Если через yfinance не удалось, попробуем через веб-скрапинг
            if not constituents:
                # Для популярных ETFs можно использовать данные с ETF.com или других источников
                if etf_ticker.upper() in ['SPY', 'VOO', 'QQQ', 'VTI', 'IWM']:
                    url = f"https://www.etf.com/{etf_ticker}"
                    headers = {'User-Agent': 'Mozilla/5.0'}

                    response = requests.get(url, headers=headers)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')

                        # Пример скрапинга - реальная реализация будет зависеть от структуры страницы
                        tables = soup.find_all('table')
                        for table in tables:
                            if 'holdings' in table.get('class', []):
                                rows = table.find_all('tr')
                                for row in rows[1:]:  # Пропускаем заголовок
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

            # Сохраняем результаты в кеш
            if constituents:
                with open(etf_cache_file, 'w') as f:
                    json.dump(constituents, f)

            return constituents
        except Exception as e:
            logger.error(f"Ошибка при получении состава ETF {etf_ticker}: {e}")
            return []

    def get_sector_performance(self) -> pd.DataFrame:
        """
        Получение данных о производительности различных секторов рынка

        Returns:
            DataFrame с доходностью секторов
        """
        # Кеширование результатов
        cache_file = self.cache_dir / "sector_performance.pkl"

        # Проверяем кеш
        if cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 1:  # Обновляем раз в день
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Не удалось загрузить кеш производительности секторов: {e}")

        try:
            # Список ETF секторов
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

            # Определяем интервалы для расчета доходности
            periods = {
                '1D': timedelta(days=1),
                '1W': timedelta(weeks=1),
                '1M': timedelta(days=30),
                '3M': timedelta(days=90),
                'YTD': timedelta(days=(datetime.now() - datetime(datetime.now().year, 1, 1)).days),
                '1Y': timedelta(days=365)
            }

            # Получаем исторические данные для всех ETF секторов
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365 + 10)).strftime('%Y-%m-%d')

            etf_data = self.get_batch_data(list(sector_etfs.values()), start_date, end_date)

            # Создаем DataFrame с результатами
            result_data = []

            for sector_name, ticker in sector_etfs.items():
                if ticker in etf_data and not etf_data[ticker].empty:
                    price_data = etf_data[ticker]

                    # Проверяем, есть ли 'Adj Close' или 'Close'
                    price_col = 'Adj Close' if 'Adj Close' in price_data.columns else 'Close'

                    # Получаем последнюю цену
                    latest_price = price_data[price_col].iloc[-1]

                    # Рассчитываем доходность для каждого периода
                    returns = {}
                    for period_name, period_delta in periods.items():
                        start_idx = price_data.index[-1] - period_delta
                        historical_data = price_data[price_data.index >= start_idx]

                        if not historical_data.empty and len(historical_data) > 1:
                            start_price = historical_data[price_col].iloc[0]
                            returns[period_name] = (latest_price / start_price - 1) * 100
                        else:
                            returns[period_name] = None

                    # Добавляем информацию в результат
                    sector_info = {
                        'Sector': sector_name,
                        'Ticker': ticker,
                        'Latest Price': latest_price
                    }
                    sector_info.update(returns)

                    result_data.append(sector_info)

            # Создаем DataFrame
            result_df = pd.DataFrame(result_data)

            # Сохраняем в кеш
            with open(cache_file, 'wb') as f:
                pickle.dump(result_df, f)

            return result_df
        except Exception as e:
            logger.error(f"Ошибка при получении производительности секторов: {e}")
            return pd.DataFrame()

    # Методы для конкретных провайдеров данных

    def _fetch_yfinance(self, ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
        """Получение данных через yfinance"""
        try:
            import yfinance as yf

            # Проверяем, содержит ли тикер точку, и создаем скорректированную версию для API
            original_ticker = ticker
            corrected_ticker = ticker.replace('.', '-') if '.' in ticker else ticker

            if corrected_ticker != original_ticker:
                logger.info(f"Используем скорректированный тикер {corrected_ticker} для запроса {original_ticker}")

            # Увеличиваем счетчик API-вызовов
            self.api_call_counts['yfinance'] += 1

            try:
                # Первый способ - используем Ticker.history с исправленным тикером
                ticker_obj = yf.Ticker(corrected_ticker)
                data = ticker_obj.history(start=start_date, end=end_date, interval=interval)

                if data is None or data.empty:
                    # Если не получилось, пробуем через download
                    data = yf.download(
                        corrected_ticker,
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        progress=False,
                        show_errors=False
                    )

                # Проверка и обработка пустых данных
                if data is None or data.empty:
                    logger.warning(
                        f"Не найдены данные для {original_ticker} (запрос как {corrected_ticker}) через yfinance")
                    return pd.DataFrame()

                # Проверка и корректировка индекса, если это не DatetimeIndex
                if not isinstance(data.index, pd.DatetimeIndex):
                    try:
                        data.index = pd.to_datetime(data.index)
                    except Exception as e:
                        logger.warning(f"Не удалось конвертировать индекс в DatetimeIndex: {e}")

                # Убедимся, что все наши колонки существуют
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_columns:
                    if col not in data.columns:
                        if col == 'Volume' and 'volume' in data.columns:
                            data['Volume'] = data['volume']
                        else:
                            data[col] = np.nan

                # Если нет 'Adj Close', используем 'Close'
                if 'Adj Close' not in data.columns:
                    data['Adj Close'] = data['Close']

                return data

            except Exception as e:
                logger.error(
                    f"Ошибка при получении данных через yfinance для {original_ticker} (запрос как {corrected_ticker}): {e}")
                return pd.DataFrame()
        except ImportError:
            logger.error("yfinance не установлен. Установите его с помощью pip install yfinance")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Ошибка при получении данных через yfinance для {ticker}: {e}")
            return pd.DataFrame()

    def _fetch_alpha_vantage(self, ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
        """Получение данных через Alpha Vantage API"""
        if not self.api_keys['alpha_vantage']:
            logger.error("API ключ для Alpha Vantage не установлен")
            return pd.DataFrame()

        # Проверка лимитов API
        if self.api_call_counts['alpha_vantage'] >= self.api_limits['alpha_vantage']:
            logger.warning("Достигнут лимит API-вызовов Alpha Vantage")
            return pd.DataFrame()

        # Проверяем, содержит ли тикер точку, и создаем скорректированную версию для API
        original_ticker = ticker
        corrected_ticker = ticker.replace('.', '-') if '.' in ticker else ticker

        if corrected_ticker != original_ticker:
            logger.info(
                f"Используем скорректированный тикер {corrected_ticker} для запроса Alpha Vantage {original_ticker}")

        # Маппинг интервалов
        interval_map = {
            '1d': 'daily',
            '1wk': 'weekly',
            '1mo': 'monthly'
        }

        function = f"TIME_SERIES_{interval_map.get(interval, 'daily').upper()}"

        try:
            # Формируем URL
            base_url = "https://www.alphavantage.co/query"
            params = {
                "function": function,
                "symbol": corrected_ticker,  # Используем исправленный тикер
                "apikey": self.api_keys['alpha_vantage'],
                "outputsize": "full"
            }

            # Увеличиваем счетчик API-вызовов
            self.api_call_counts['alpha_vantage'] += 1

            # Делаем запрос
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Проверка на ошибки HTTP

            data = response.json()

            # Проверяем наличие ошибок в ответе
            if "Error Message" in data:
                logger.error(
                    f"Alpha Vantage API вернул ошибку для {original_ticker} (запрос как {corrected_ticker}): {data['Error Message']}")
                return pd.DataFrame()

            # Определяем ключ временного ряда в зависимости от функции
            time_series_key = next((k for k in data.keys() if k.startswith("Time Series")), None)

            if not time_series_key:
                logger.error(f"Неожиданный формат ответа Alpha Vantage для {original_ticker}: {data.keys()}")
                return pd.DataFrame()

            # Преобразуем данные в DataFrame
            time_series_data = data[time_series_key]

            # Создаем DataFrame
            df = pd.DataFrame.from_dict(time_series_data, orient='index')

            # Преобразуем названия столбцов
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

            # Преобразуем типы данных
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Устанавливаем индекс как DatetimeIndex
            df.index = pd.to_datetime(df.index)

            # Сортируем по дате
            df = df.sort_index()

            # Фильтруем по заданным датам
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса к Alpha Vantage API для {original_ticker}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Ошибка при обработке данных Alpha Vantage для {original_ticker}: {e}")
            return pd.DataFrame()

    def _get_yfinance_company_info(self, ticker: str) -> Dict:
        """Получение информации о компании через yfinance"""
        try:
            import yfinance as yf

            # Проверяем, содержит ли тикер точку, и создаем скорректированную версию для API
            original_ticker = ticker
            corrected_ticker = ticker.replace('.', '-') if '.' in ticker else ticker

            if corrected_ticker != original_ticker:
                logger.info(
                    f"Используем скорректированный тикер {corrected_ticker} для запроса информации о компании {original_ticker}")

            # Увеличиваем счетчик API-вызовов
            self.api_call_counts['yfinance'] += 1

            # Получаем данные
            ticker_obj = yf.Ticker(corrected_ticker)
            info = ticker_obj.info

            # Обработка и нормализация данных
            # Некоторые ключи могут отсутствовать, добавляем базовые проверки
            normalized_info = {
                'symbol': original_ticker,  # Сохраняем оригинальный тикер
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

            # Добавляем финансовые показатели, если доступны
            for metric in ['returnOnEquity', 'returnOnAssets', 'profitMargins', 'operatingMargins', 'grossMargins']:
                if metric in info:
                    normalized_info[self._camel_to_snake(metric)] = info[metric]

            return normalized_info
        except ImportError:
            logger.error("yfinance не установлен. Установите его с помощью pip install yfinance")
            return {}
        except Exception as e:
            logger.error(f"Ошибка при получении информации о компании через yfinance для {original_ticker}: {e}")
            return {}

    def _get_alpha_vantage_company_info(self, ticker: str) -> Dict:
        """Получение информации о компании через Alpha Vantage"""
        if not self.api_keys['alpha_vantage']:
            logger.error("API ключ для Alpha Vantage не установлен")
            return {}

        # Проверка лимитов API
        if self.api_call_counts['alpha_vantage'] >= self.api_limits['alpha_vantage']:
            logger.warning("Достигнут лимит API-вызовов Alpha Vantage")
            return {}

        # Проверяем, содержит ли тикер точку, и создаем скорректированную версию для API
        original_ticker = ticker
        corrected_ticker = ticker.replace('.', '-') if '.' in ticker else ticker

        if corrected_ticker != original_ticker:
            logger.info(
                f"Используем скорректированный тикер {corrected_ticker} для запроса Alpha Vantage о компании {original_ticker}")

        try:
            # Формируем URL для overview
            base_url = "https://www.alphavantage.co/query"
            params = {
                "function": "OVERVIEW",
                "symbol": corrected_ticker,  # Используем исправленный тикер
                "apikey": self.api_keys['alpha_vantage']
            }

            # Увеличиваем счетчик API-вызовов
            self.api_call_counts['alpha_vantage'] += 1

            # Делаем запрос
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()

            # Проверяем наличие данных
            if not data or ("Error Message" in data) or len(data.keys()) <= 1:
                logger.warning(f"Alpha Vantage не вернул данные для {original_ticker} (запрос как {corrected_ticker})")
                return {}

            # Преобразуем и нормализуем данные
            normalized_info = {
                'symbol': original_ticker,  # Сохраняем оригинальный тикер
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

            # Добавляем финансовые показатели
            for key in ['ReturnOnEquityTTM', 'ReturnOnAssetsTTM', 'ProfitMargin', 'OperatingMarginTTM',
                        'GrossProfitTTM']:
                if key in data:
                    normalized_key = self._camel_to_snake(key.replace('TTM', ''))
                    normalized_info[normalized_key] = self._safe_convert(data[key], float)

            return normalized_info
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса к Alpha Vantage API для информации о компании {original_ticker}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Ошибка при обработке информации Alpha Vantage для {original_ticker}: {e}")
            return {}

    def _search_alpha_vantage(self, query: str, limit: int = 10) -> List[Dict]:
        """Поиск тикеров через Alpha Vantage"""
        if not self.api_keys['alpha_vantage']:
            logger.error("API ключ для Alpha Vantage не установлен")
            return []

        # Проверка лимитов API
        if self.api_call_counts['alpha_vantage'] >= self.api_limits['alpha_vantage']:
            logger.warning("Достигнут лимит API-вызовов Alpha Vantage")
            return []

        try:
            # Формируем URL
            base_url = "https://www.alphavantage.co/query"
            params = {
                "function": "SYMBOL_SEARCH",
                "keywords": query,
                "apikey": self.api_keys['alpha_vantage']
            }

            # Увеличиваем счетчик API-вызовов
            self.api_call_counts['alpha_vantage'] += 1

            # Делаем запрос
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()

            results = []
            if 'bestMatches' in data:
                for match in data['bestMatches'][:limit]:
                    # Получаем символ из результата
                    symbol = match.get('1. symbol', '')

                    # Проверяем, содержит ли символ дефис, который может быть представлением точки
                    # Заменяем дефис на точку для отображения пользователю, если это соответствует
                    # известному формату тикеров с точками (например, BRK-B -> BRK.B)
                    display_symbol = symbol
                    if '-' in symbol:
                        # Только для известных паттернов (например, BRK-B, BF-B)
                        known_patterns = ['BRK-B', 'BF-B']
                        if symbol in known_patterns or (len(symbol.split('-')) == 2 and len(symbol.split('-')[1]) == 1):
                            display_symbol = symbol.replace('-', '.')
                            logger.info(f"Преобразован тикер в результатах поиска: {symbol} -> {display_symbol}")

                    results.append({
                        'symbol': display_symbol,  # Отображаем с точкой, если это подходящий формат
                        'original_symbol': symbol,  # Сохраняем оригинальный символ
                        'name': match.get('2. name', ''),
                        'type': match.get('3. type', ''),
                        'region': match.get('4. region', ''),
                        'currency': match.get('8. currency', 'USD'),
                        'exchange': match.get('5. marketOpen', '') + '-' + match.get('6. marketClose', ''),
                        'timezone': match.get('7. timezone', '')
                    })

            return results
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса к Alpha Vantage API для поиска: {e}")
            return []
        except Exception as e:
            logger.error(f"Ошибка при поиске тикеров через Alpha Vantage: {e}")
            return []

    def _search_alternative(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Альтернативный метод поиска тикеров через Yahoo Finance API

        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов

        Returns:
            Список словарей с информацией о найденных тикерах
        """
        try:
            import yfinance as yf
            import json
            import requests

            # Используем Yahoo Finance API для поиска
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount={limit}&newsCount=0"
            headers = {'User-Agent': 'Mozilla/5.0'}

            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                logger.error(f"Ошибка запроса к Yahoo Finance API: {response.status_code}")
                return []

            data = response.json()

            results = []
            if 'quotes' in data and data['quotes']:
                for quote in data['quotes'][:limit]:
                    # Преобразуем формат в тот же, что и Alpha Vantage
                    ticker = quote.get('symbol', '')

                    # Проверяем, содержит ли тикер дефис, который может быть представлением точки
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
            logger.error(f"Ошибка при поиске тикеров через Yahoo Finance: {e}")
            return []

    # Вспомогательные методы

    def _determine_asset_type(self, info: Dict) -> str:
        """Определить тип актива на основе информации"""
        if not info:
            return 'Unknown'

        # Определяем тип по доступным данным
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

        # Если нет прямых указаний, пытаемся определить по другим признакам
        if 'fundFamily' in info and info['fundFamily']:
            return 'ETF' if 'ETF' in info.get('longName', '') else 'Mutual Fund'

        if 'industry' in info and info['industry']:
            return 'Stock'

        return 'Unknown'

    def _camel_to_snake(self, name: str) -> str:
        """Преобразование camelCase в snake_case"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _safe_convert(self, value: Any, convert_type) -> Optional[Any]:
        """Безопасное преобразование типов с обработкой ошибок"""
        if value is None:
            return None

        try:
            return convert_type(value)
        except (ValueError, TypeError):
            return None

    def clear_cache(self, tickers: Optional[List[str]] = None):
        """
        Очистка кеша данных

        Args:
            tickers: Список тикеров для очистки. Если None, очищается весь кеш.
        """
        if tickers is None:
            # Очистка всего кеша
            for file in self.cache_dir.glob('*.*'):
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Не удалось удалить файл кеша {file}: {e}")
            logger.info("Кеш полностью очищен")
        else:
            # Очистка кеша только для указанных тикеров
            for ticker in tickers:
                for file in self.cache_dir.glob(f"{ticker}_*.*"):
                    try:
                        file.unlink()
                    except Exception as e:
                        logger.warning(f"Не удалось удалить файл кеша {file}: {e}")
            logger.info(f"Кеш очищен для тикеров: {tickers}")

    def get_batch_data(self, tickers: List[str], start_date: Optional[str] = None,
                       end_date: Optional[str] = None, provider: str = 'yfinance') -> Dict[str, pd.DataFrame]:
        """
        Получение данных для нескольких тикеров одновременно

        Args:
            tickers: Список тикеров
            start_date: Начальная дата
            end_date: Конечная дата
            provider: Провайдер данных

        Returns:
            Словарь {ticker: DataFrame}
        """
        results = {}

        # Обработка специальных тикеров с точками
        ticker_mapping = {}
        corrected_tickers = []

        for ticker in tickers:
            if '.' in ticker:
                # Заменяем точку на дефис для запроса данных
                corrected_ticker = ticker.replace('.', '-')
                ticker_mapping[corrected_ticker] = ticker
                corrected_tickers.append(corrected_ticker)
            else:
                corrected_tickers.append(ticker)
                ticker_mapping[ticker] = ticker

        if ticker_mapping:
            logger.info(f"Выполнена замена тикеров для запроса данных: {ticker_mapping}")

        # Реализуем многопоточную загрузку для yfinance
        if provider == 'yfinance' and len(corrected_tickers) > 1:
            try:
                import yfinance as yf

                # Установка значений по умолчанию для дат
                if end_date is None:
                    end_date = datetime.now().strftime('%Y-%m-%d')

                if start_date is None:
                    start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

                # Проверяем кеш для каждого тикера (используем оригинальные тикеры для имен файлов)
                tickers_to_download = []
                download_mapping = {}  # Маппинг для загрузки: {corrected_ticker: original_ticker}

                for original_ticker in tickers:
                    cache_file = self.cache_dir / f"{original_ticker}_{start_date}_{end_date}_1d_{provider}.pkl"
                    if cache_file.exists():
                        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                        if file_age.days < self.cache_expiry_days:
                            try:
                                with open(cache_file, 'rb') as f:
                                    results[original_ticker] = pickle.load(f)
                                    logger.info(f"Загрузка данных {original_ticker} из кеша")
                                    continue
                            except Exception as e:
                                logger.warning(f"Не удалось загрузить кеш для {original_ticker}: {e}")

                    # Находим скорректированный тикер для загрузки
                    corrected_ticker = original_ticker.replace('.', '-') if '.' in original_ticker else original_ticker
                    tickers_to_download.append(corrected_ticker)
                    download_mapping[corrected_ticker] = original_ticker

                # Загружаем недостающие данные
                if tickers_to_download:
                    self.api_call_counts['yfinance'] += 1

                    # Загружаем данные одним запросом
                    data = yf.download(
                        tickers_to_download,
                        start=start_date,
                        end=end_date,
                        interval='1d',
                        group_by='ticker',
                        progress=False,
                        show_errors=False
                    )

                    # Обрабатываем и сохраняем данные для каждого тикера
                    for corrected_ticker, original_ticker in download_mapping.items():
                        if len(tickers_to_download) == 1:
                            # Если только один тикер, данные не группируются
                            ticker_data = data
                        else:
                            # Извлекаем данные для конкретного тикера
                            ticker_data = data[corrected_ticker].copy() if corrected_ticker in data else pd.DataFrame()

                        # Проверяем и сохраняем данные
                        if not ticker_data.empty:
                            # Сохраняем в кеш под оригинальным именем
                            cache_file = self.cache_dir / f"{original_ticker}_{start_date}_{end_date}_1d_{provider}.pkl"
                            with open(cache_file, 'wb') as f:
                                pickle.dump(ticker_data, f)

                            results[original_ticker] = ticker_data
                            logger.info(
                                f"Загружены и сохранены данные для {original_ticker} (запрос как {corrected_ticker})")

                return results
            except ImportError:
                logger.error("yfinance не установлен. Установите его с помощью pip install yfinance")
            except Exception as e:
                logger.error(f"Ошибка при пакетной загрузке данных: {e}")
                # Продолжаем с последовательной загрузкой как резервным вариантом

        # Последовательная загрузка для каждого тикера
        for original_ticker in tickers:
            try:
                # Для одиночной загрузки используем модифицированный метод get_historical_prices
                corrected_ticker = original_ticker.replace('.', '-') if '.' in original_ticker else original_ticker

                # Временно заменяем тикер для запроса
                data = self.get_historical_prices(corrected_ticker, start_date, end_date, provider)

                if not data.empty:
                    # Сохраняем под оригинальным тикером
                    results[original_ticker] = data
            except Exception as e:
                logger.error(f"Ошибка при загрузке данных для {original_ticker}: {e}")

        return results

    def get_fundamental_data(self, ticker: str, data_type: str = 'income') -> pd.DataFrame:
        """
        Получение фундаментальных финансовых данных

        Args:
            ticker: Тикер компании
            data_type: Тип данных ('income', 'balance', 'cash', 'earnings')

        Returns:
            DataFrame с фундаментальными данными
        """
        cache_file = self.cache_dir / f"{ticker}_fundamental_{data_type}.pkl"

        # Проверка кеша (срок годности фундаментальных данных - 30 дней)
        if cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.days < 30:  # Фундаментальные данные обновляются реже
                try:
                    with open(cache_file, 'rb') as f:
                        logger.info(f"Загрузка фундаментальных данных {data_type} для {ticker} из кеша")
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Не удалось загрузить кеш фундаментальных данных для {ticker}: {e}")

        try:
            import yfinance as yf

            # Увеличиваем счетчик API-вызовов
            self.api_call_counts['yfinance'] += 1

            # Получаем данные
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
                raise ValueError(f"Неподдерживаемый тип фундаментальных данных: {data_type}")

            # Проверка данных
            if df is None or df.empty:
                logger.warning(f"Не найдены фундаментальные данные {data_type} для {ticker}")
                return pd.DataFrame()

            # Сохраняем в кеш
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)

            return df
        except ImportError:
            logger.error("yfinance не установлен. Установите его с помощью pip install yfinance")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Ошибка при получении фундаментальных данных для {ticker}: {e}")
            return pd.DataFrame()


# Класс для работы с данными портфеля
class PortfolioDataManager:
    """
    Класс для работы с данными портфеля.
    Обеспечивает загрузку, сохранение и обмен данными портфелей.
    """

    def __init__(self, data_fetcher: DataFetcher, storage_dir: str = './portfolios'):
        """
        Инициализация менеджера данных портфеля

        Args:
            data_fetcher: Экземпляр DataFetcher для загрузки рыночных данных
            storage_dir: Директория для хранения портфелей
        """
        self.data_fetcher = data_fetcher
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Инициализирован PortfolioDataManager с хранилищем в {self.storage_dir}")

    def save_portfolio(self, portfolio_data: Dict, filename: Optional[str] = None) -> str:
        """
        Сохранение данных портфеля в файл

        Args:
            portfolio_data: Словарь с данными портфеля
            filename: Имя файла (без расширения). Если None, использует portfolio_data['name']

        Returns:
            Путь к сохраненному файлу
        """
        if 'name' not in portfolio_data:
            raise ValueError("В данных портфеля отсутствует обязательное поле 'name'")

        # Определение имени файла
        if filename is None:
            filename = self._sanitize_filename(portfolio_data['name'])

        file_path = self.storage_dir / f"{filename}.json"

        try:
            # Добавляем метаданные
            if 'last_updated' not in portfolio_data:
                portfolio_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Сохраняем данные
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(portfolio_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Портфель '{portfolio_data['name']}' сохранен в {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Ошибка при сохранении портфеля: {e}")
            raise

    def load_portfolio(self, filename: str) -> Dict:
        """
        Загрузка данных портфеля из файла

        Args:
            filename: Имя файла (с расширением или без)

        Returns:
            Словарь с данными портфеля
        """
        # Обработка имени файла
        if not filename.endswith('.json'):
            filename += '.json'

        # Проверяем, существует ли файл напрямую
        file_path = self.storage_dir / filename

        # Если файл не найден по прямому имени, попробуем поискать по альтернативным именам
        if not file_path.exists():
            # Ищем файлы с похожими именами
            potential_files = list(self.storage_dir.glob('*.json'))

            # Нормализуем имя файла для сравнения (удаление пробелов и перевод в нижний регистр)
            normalized_name = filename.lower().replace(' ', '').replace('&', 'and')

            for potential_file in potential_files:
                # Нормализуем имя потенциального файла
                potential_normalized = potential_file.name.lower().replace(' ', '').replace('&', 'and')

                # Проверяем на совпадение после нормализации
                if normalized_name == potential_normalized:
                    file_path = potential_file
                    logger.info(f"Найдено альтернативное имя файла: {potential_file}")
                    break

                # Также проверяем по началу имени
                if len(normalized_name) > 10 and normalized_name[:10] == potential_normalized[:10]:
                    file_path = potential_file
                    logger.info(f"Найдено похожее имя файла: {potential_file}")
                    break

        if not file_path.exists():
            # Если файл все еще не найден, попробуем поискать имя внутри JSON файлов
            for json_file in self.storage_dir.glob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Проверяем, совпадает ли имя портфеля с запрошенным
                        if 'name' in data and data['name'] == filename.replace('.json', ''):
                            file_path = json_file
                            logger.info(f"Найден файл по имени портфеля: {json_file}")
                            break
                except Exception:
                    # Игнорируем ошибки при чтении файлов
                    continue

        if not file_path.exists():
            raise FileNotFoundError(f"Файл портфеля не найден: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                portfolio_data = json.load(f)

            logger.info(f"Портфель загружен из {file_path}")
            return portfolio_data
        except Exception as e:
            logger.error(f"Ошибка при загрузке портфеля из {file_path}: {e}")
            raise

    def list_portfolios(self) -> List[Dict]:
        """
        Получение списка всех доступных портфелей

        Returns:
            Список словарей с базовой информацией о портфелях
        """
        portfolios = []

        try:
            for file_path in self.storage_dir.glob('*.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Формируем базовую информацию
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
                    logger.warning(f"Не удалось прочитать данные портфеля из {file_path}: {e}")

            return sorted(portfolios, key=lambda x: x['name'])
        except Exception as e:
            logger.error(f"Ошибка при получении списка портфелей: {e}")
            return []

    def delete_portfolio(self, filename: str) -> bool:
        """
        Удаление портфеля

        Args:
            filename: Имя файла (с расширением или без)

        Returns:
            True, если удаление прошло успешно, иначе False
        """
        # Обработка имени файла
        if not filename.endswith('.json'):
            filename += '.json'

        file_path = self.storage_dir / filename

        if not file_path.exists():
            logger.warning(f"Файл портфеля для удаления не найден: {file_path}")
            return False

        try:
            file_path.unlink()
            logger.info(f"Портфель удален: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при удалении портфеля {file_path}: {e}")
            return False

    def import_from_csv(self, file_path: str, portfolio_name: Optional[str] = None) -> Dict:
        """
        Импорт портфеля из CSV-файла

        Args:
            file_path: Путь к CSV-файлу
            portfolio_name: Название портфеля (если None, используется имя файла)

        Returns:
            Словарь с данными портфеля
        """
        try:
            df = pd.read_csv(file_path)

            # Проверка обязательных колонок
            required_columns = ['ticker']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"В CSV-файле отсутствуют обязательные колонки: {missing_columns}")

            # Подготовка имени портфеля
            if portfolio_name is None:
                file_name = Path(file_path).stem
                portfolio_name = file_name

            # Создание структуры портфеля
            portfolio_data = {
                'name': portfolio_name,
                'description': f"Импортирован из {Path(file_path).name}",
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'assets': []
            }

            # Заполнение данных об активах
            for _, row in df.iterrows():
                asset = {'ticker': row['ticker']}

                # Добавление опциональных полей, если они есть
                optional_fields = ['weight', 'quantity', 'purchase_price', 'purchase_date',
                                   'sector', 'asset_class', 'region', 'currency']

                for field in optional_fields:
                    if field in df.columns and not pd.isna(row[field]):
                        asset[field] = row[field]

                # Если нет веса, но есть количество, устанавливаем вес по умолчанию
                if 'weight' not in asset and 'quantity' in asset:
                    asset['weight'] = 0.0  # Будет пересчитано позже

                portfolio_data['assets'].append(asset)

            # Если не указаны веса, но указаны количества, расчитываем веса
            if all('quantity' in asset for asset in portfolio_data['assets']) and \
                    not all('weight' in asset for asset in portfolio_data['assets']):
                self._calculate_weights_from_quantities(portfolio_data)

            # Нормализация весов
            self._normalize_weights(portfolio_data)

            # Получение дополнительной информации о тикерах
            self._enrich_portfolio_data(portfolio_data)

            logger.info(f"Портфель '{portfolio_name}' импортирован из {file_path}")
            return portfolio_data
        except Exception as e:
            logger.error(f"Ошибка при импорте портфеля из CSV {file_path}: {e}")
            raise

    def export_to_csv(self, portfolio_data: Dict, file_path: Optional[str] = None) -> str:
        """
        Экспорт портфеля в CSV-файл

        Args:
            portfolio_data: Словарь с данными портфеля
            file_path: Путь для сохранения (если None, создается на основе имени портфеля)

        Returns:
            Путь к сохраненному файлу
        """
        if 'name' not in portfolio_data or 'assets' not in portfolio_data:
            raise ValueError("Неверный формат данных портфеля")

        # Определение пути для сохранения
        if file_path is None:
            filename = self._sanitize_filename(portfolio_data['name'])
            file_path = self.storage_dir / f"{filename}.csv"

        try:
            # Создаем DataFrame из активов
            assets_data = []

            for asset in portfolio_data['assets']:
                asset_row = {'ticker': asset['ticker']}

                # Добавляем все доступные поля
                for key, value in asset.items():
                    if key != 'ticker':
                        asset_row[key] = value

                assets_data.append(asset_row)

            # Создаем DataFrame и сохраняем
            df = pd.DataFrame(assets_data)
            df.to_csv(file_path, index=False)

            logger.info(f"Портфель '{portfolio_data['name']}' экспортирован в {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Ошибка при экспорте портфеля в CSV: {e}")
            raise

    def parse_ticker_weights_text(self, text: str) -> List[Dict]:
        """
        Парсинг текстового списка тикеров и весов

        Args:
            text: Текст в формате "TICKER:weight, TICKER:weight" или "TICKER weight, TICKER weight"

        Returns:
            Список словарей с данными активов
        """
        assets = []

        # Удаляем лишние пробелы и переводы строк
        text = text.strip()

        # Разделяем по запятым или переводам строк
        lines = re.split(r'[,\n]+', text)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Пробуем разные форматы
            # Формат "TICKER:weight"
            colon_match = re.match(r'^([A-Za-z0-9.-]+):([0-9.]+)$', line)

            # Формат "TICKER weight"
            space_match = re.match(r'^([A-Za-z0-9.-]+)\s+([0-9.]+)$', line)

            # Формат "TICKER" (без веса)
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
                    'weight': 0.0  # Будет нормализовано позже
                })

        # Если все веса равны 0, устанавливаем равные веса
        if assets and all(asset['weight'] == 0.0 for asset in assets):
            equal_weight = 1.0 / len(assets)
            for asset in assets:
                asset['weight'] = equal_weight

        return assets

    def create_portfolio_from_text(self, text: str, portfolio_name: str, description: str = "") -> Dict:
        """
        Создание портфеля из текстового списка тикеров

        Args:
            text: Текст с тикерами и весами
            portfolio_name: Название портфеля
            description: Описание портфеля

        Returns:
            Словарь с данными портфеля
        """
        assets = self.parse_ticker_weights_text(text)

        if not assets:
            raise ValueError("Не удалось распознать ни одного тикера в тексте")

        # Создание структуры портфеля
        portfolio_data = {
            'name': portfolio_name,
            'description': description,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'assets': assets
        }

        # Нормализация весов
        self._normalize_weights(portfolio_data)

        # Проверка валидности тикеров
        valid_tickers, invalid_tickers = self.data_fetcher.validate_tickers([a['ticker'] for a in assets])

        if invalid_tickers:
            logger.warning(f"Найдены невалидные тикеры: {invalid_tickers}")

            # Удаляем невалидные тикеры
            portfolio_data['assets'] = [a for a in portfolio_data['assets']
                                        if a['ticker'] in valid_tickers]

            # Перенормализация весов
            self._normalize_weights(portfolio_data)

        # Обогащение данных портфеля
        self._enrich_portfolio_data(portfolio_data)

        logger.info(f"Создан портфель '{portfolio_name}' с {len(portfolio_data['assets'])} активами")
        return portfolio_data

    def update_portfolio_prices(self, portfolio_data: Dict) -> Dict:
        """
        Обновление текущих цен активов в портфеле

        Args:
            portfolio_data: Словарь с данными портфеля

        Returns:
            Обновленный словарь с данными портфеля
        """
        if 'assets' not in portfolio_data or not portfolio_data['assets']:
            return portfolio_data

        # Получаем список тикеров
        tickers = [asset['ticker'] for asset in portfolio_data['assets']]

        # Получаем последние цены
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')

        price_data = self.data_fetcher.get_batch_data(tickers, start_date, end_date)

        # Обновляем данные портфеля
        for asset in portfolio_data['assets']:
            ticker = asset['ticker']

            if ticker in price_data and not price_data[ticker].empty:
                # Получаем последнюю доступную цену
                latest_data = price_data[ticker].iloc[-1]

                if 'Adj Close' in latest_data:
                    current_price = latest_data['Adj Close']
                elif 'Close' in latest_data:
                    current_price = latest_data['Close']
                else:
                    continue

                # Обновляем текущую цену
                asset['current_price'] = current_price
                asset['price_date'] = price_data[ticker].index[-1].strftime('%Y-%m-%d')

                # Рассчитываем изменение цены, если есть цена покупки
                if 'purchase_price' in asset and asset['purchase_price']:
                    asset['price_change_pct'] = ((current_price / float(asset['purchase_price'])) - 1) * 100

        # Обновляем дату последнего обновления
        portfolio_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"Обновлены цены активов для портфеля '{portfolio_data['name']}'")
        return portfolio_data

    def _sanitize_filename(self, filename: str) -> str:
        """Очистка имени файла от недопустимых символов"""
        # Заменяем недопустимые символы на '_'
        # Расширенный список символов, которые нужно заменить
        filename = re.sub(r'[\\/*?:"<>|%\\\\\s]', '_', filename)

        # Также заменяем амперсанд (&), который может вызывать проблемы
        filename = filename.replace('&', 'and')

        # Также заменяем слеши и двоеточия, которые могут вызывать проблемы
        filename = filename.replace('\\', '_').replace('/', '_').replace(':', '_')

        # Ограничиваем длину имени файла
        if len(filename) > 200:
            filename = filename[:197] + '...'

        return filename

    def _normalize_weights(self, portfolio_data: Dict) -> None:
        """Нормализация весов активов в портфеле, чтобы сумма была равна 1.0"""
        if 'assets' not in portfolio_data or not portfolio_data['assets']:
            return

        total_weight = sum(asset.get('weight', 0.0) for asset in portfolio_data['assets'])

        if total_weight > 0:
            for asset in portfolio_data['assets']:
                if 'weight' in asset:
                    asset['weight'] = asset['weight'] / total_weight

    def _calculate_weights_from_quantities(self, portfolio_data: Dict) -> None:
        """Расчет весов на основе количеств и текущих цен"""
        if 'assets' not in portfolio_data or not portfolio_data['assets']:
            return

        # Получаем текущие цены
        tickers = [asset['ticker'] for asset in portfolio_data['assets'] if 'quantity' in asset]
        price_data = self.data_fetcher.get_batch_data(tickers)

        # Рассчитываем стоимости позиций
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

        # Рассчитываем веса
        if total_value > 0:
            for asset in portfolio_data['assets']:
                if 'position_value' in asset:
                    asset['weight'] = asset['position_value'] / total_value
                    # Удаляем временное поле
                    del asset['position_value']

    def _enrich_portfolio_data(self, portfolio_data: Dict) -> None:
        """Обогащение данных портфеля дополнительной информацией"""
        if 'assets' not in portfolio_data or not portfolio_data['assets']:
            return

        # Получаем информацию о компаниях
        for asset in portfolio_data['assets']:
            try:
                info = self.data_fetcher.get_company_info(asset['ticker'])

                if info:
                    # Добавляем базовую информацию
                    for field in ['name', 'sector', 'industry', 'asset_class', 'currency']:
                        if field in info and field not in asset:
                            asset[field] = info[field]

                    # Если не указан тип актива, устанавливаем из информации
                    if 'asset_class' not in asset and 'type' in info:
                        asset['asset_class'] = info['type']
            except Exception as e:
                logger.warning(f"Не удалось получить информацию для {asset['ticker']}: {e}")

        # Обновляем текущие цены
        self.update_portfolio_prices(portfolio_data)