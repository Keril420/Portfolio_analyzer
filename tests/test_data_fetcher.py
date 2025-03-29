import unittest
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Добавляем путь к директории src для импорта
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Импортируем модуль из src/utils
from utils.data_fetcher import DataFetcher, PortfolioDataManager


class TestDataFetcher(unittest.TestCase):
    """Тестирование класса DataFetcher"""

    def setUp(self):
        """Настройка для каждого теста"""
        # Создаем временную директорию для кеша
        self.temp_dir = tempfile.mkdtemp()
        # Инициализируем DataFetcher с временной директорией
        self.data_fetcher = DataFetcher(cache_dir=self.temp_dir, cache_expiry_days=1)

    def tearDown(self):
        """Очистка после каждого теста"""
        # Удаляем временную директорию
        shutil.rmtree(self.temp_dir)

    @patch('data_fetcher.requests.get')
    def test_alpha_vantage_api_call(self, mock_get):
        """Тест запроса к Alpha Vantage API"""
        # Создаем мок-ответ
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "Time Series (Daily)": {
                "2023-01-03": {
                    "1. open": "130.28",
                    "2. high": "130.90",
                    "3. low": "124.17",
                    "4. close": "125.07",
                    "5. volume": "112117500"
                }
            }
        }
        mock_get.return_value = mock_response

        # Устанавливаем API ключ для теста
        self.data_fetcher.api_keys['alpha_vantage'] = 'test_key'

        # Вызываем метод
        result = self.data_fetcher._fetch_alpha_vantage('AAPL', '2023-01-01', '2023-01-05')

        # Проверяем, что метод get был вызван с правильными параметрами
        mock_get.assert_called_once()
        call_args = mock_get.call_args[1]['params']
        self.assertEqual(call_args['symbol'], 'AAPL')
        self.assertEqual(call_args['apikey'], 'test_key')

        # Проверяем результат
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

    @patch('data_fetcher.yf.Ticker')
    @patch('data_fetcher.yf.download')
    def test_fetch_yfinance(self, mock_download, mock_ticker):
        """Тест получения данных через yfinance"""
        # Настраиваем моки
        mock_ticker_instance = MagicMock()
        mock_history = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [102, 103],
            'Volume': [1000, 1100]
        }, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02']))
        mock_ticker_instance.history.return_value = mock_history
        mock_ticker.return_value = mock_ticker_instance

        # Если history не сработает, будет вызван download
        mock_download.return_value = pd.DataFrame()

        # Вызываем метод
        result = self.data_fetcher._fetch_yfinance('AAPL', '2023-01-01', '2023-01-02')

        # Проверяем результаты
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('Open', result.columns)
        self.assertIn('Close', result.columns)

    def test_parse_ticker_weights_text(self):
        """Тест парсинга текста с тикерами и весами"""
        # Инициализируем PortfolioDataManager
        portfolio_manager = PortfolioDataManager(self.data_fetcher, storage_dir=self.temp_dir)

        # Тестируем различные форматы ввода
        text1 = "AAPL:0.5, MSFT:0.5"
        assets1 = portfolio_manager.parse_ticker_weights_text(text1)
        self.assertEqual(len(assets1), 2)
        self.assertEqual(assets1[0]['ticker'], 'AAPL')
        self.assertEqual(assets1[0]['weight'], 0.5)

        text2 = "AAPL 0.3\nMSFT 0.3\nGOOG 0.4"
        assets2 = portfolio_manager.parse_ticker_weights_text(text2)
        self.assertEqual(len(assets2), 3)
        self.assertEqual(assets2[2]['ticker'], 'GOOG')
        self.assertEqual(assets2[2]['weight'], 0.4)

        text3 = "AAPL\nMSFT\nGOOG"
        assets3 = portfolio_manager.parse_ticker_weights_text(text3)
        self.assertEqual(len(assets3), 3)
        # Проверяем, что веса распределены равномерно
        self.assertAlmostEqual(assets3[0]['weight'], 1 / 3)
        self.assertAlmostEqual(assets3[1]['weight'], 1 / 3)
        self.assertAlmostEqual(assets3[2]['weight'], 1 / 3)

    @patch('data_fetcher.DataFetcher.get_historical_prices')
    def test_validate_tickers(self, mock_get_historical):
        """Тест валидации тикеров"""

        # Настраиваем мок для имитации успешного и неуспешного получения данных
        def mock_get_prices(ticker, *args, **kwargs):
            # Имитируем, что AAPL и MSFT существуют, а INVALID не существует
            if ticker in ['AAPL', 'MSFT']:
                return pd.DataFrame({'Close': [100, 101]})
            else:
                return pd.DataFrame()  # Пустой DataFrame для невалидного тикера

        mock_get_historical.side_effect = mock_get_prices

        # Проверяем валидацию
        valid, invalid = self.data_fetcher.validate_tickers(['AAPL', 'MSFT', 'INVALID'])
        self.assertEqual(valid, ['AAPL', 'MSFT'])
        self.assertEqual(invalid, ['INVALID'])

    @patch('data_fetcher.DataFetcher.get_historical_prices')
    @patch('data_fetcher.DataFetcher.get_company_info')
    def test_create_portfolio_from_text(self, mock_get_info, mock_get_prices):
        """Тест создания портфеля из текста"""

        # Настраиваем моки
        def mock_get_prices(ticker, *args, **kwargs):
            return pd.DataFrame({'Close': [100, 101]})

        mock_get_prices.side_effect = mock_get_prices

        mock_get_info.return_value = {
            'name': f'Company {ticker}',
            'sector': 'Technology',
            'industry': 'Software',
            'type': 'Stock',
            'currency': 'USD'
        }

        # Инициализируем PortfolioDataManager
        portfolio_manager = PortfolioDataManager(self.data_fetcher, storage_dir=self.temp_dir)

        # Создаем портфель
        portfolio = portfolio_manager.create_portfolio_from_text(
            "AAPL:0.6, MSFT:0.4",
            "Test Portfolio",
            "Test description"
        )

        # Проверяем результаты
        self.assertEqual(portfolio['name'], "Test Portfolio")
        self.assertEqual(portfolio['description'], "Test description")
        self.assertEqual(len(portfolio['assets']), 2)
        self.assertEqual(portfolio['assets'][0]['ticker'], 'AAPL')
        self.assertEqual(portfolio['assets'][0]['weight'], 0.6)
        self.assertEqual(portfolio['assets'][1]['ticker'], 'MSFT')
        self.assertEqual(portfolio['assets'][1]['weight'], 0.4)

    def test_normalize_weights(self):
        """Тест нормализации весов портфеля"""
        # Инициализируем PortfolioDataManager
        portfolio_manager = PortfolioDataManager(self.data_fetcher, storage_dir=self.temp_dir)

        # Создаем тестовый портфель с ненормализованными весами
        portfolio = {
            'name': 'Test Portfolio',
            'assets': [
                {'ticker': 'AAPL', 'weight': 5},
                {'ticker': 'MSFT', 'weight': 3},
                {'ticker': 'GOOG', 'weight': 2}
            ]
        }

        # Нормализуем веса
        portfolio_manager._normalize_weights(portfolio)

        # Проверяем, что сумма весов равна 1
        total_weight = sum(asset['weight'] for asset in portfolio['assets'])
        self.assertAlmostEqual(total_weight, 1.0)

        # Проверяем пропорции
        self.assertAlmostEqual(portfolio['assets'][0]['weight'], 0.5)  # 5/10
        self.assertAlmostEqual(portfolio['assets'][1]['weight'], 0.3)  # 3/10
        self.assertAlmostEqual(portfolio['assets'][2]['weight'], 0.2)  # 2/10


class TestPortfolioDataManager(unittest.TestCase):
    """Тестирование класса PortfolioDataManager"""

    def setUp(self):
        """Настройка для каждого теста"""
        # Создаем временные директории
        self.temp_cache_dir = tempfile.mkdtemp()
        self.temp_storage_dir = tempfile.mkdtemp()

        # Инициализируем классы
        self.data_fetcher = DataFetcher(cache_dir=self.temp_cache_dir)
        self.portfolio_manager = PortfolioDataManager(
            self.data_fetcher,
            storage_dir=self.temp_storage_dir
        )

    def tearDown(self):
        """Очистка после каждого теста"""
        # Удаляем временные директории
        shutil.rmtree(self.temp_cache_dir)
        shutil.rmtree(self.temp_storage_dir)

    def test_save_and_load_portfolio(self):
        """Тест сохранения и загрузки портфеля"""
        # Создаем тестовый портфель
        portfolio = {
            'name': 'Test Portfolio',
            'description': 'Test Description',
            'assets': [
                {'ticker': 'AAPL', 'weight': 0.6},
                {'ticker': 'MSFT', 'weight': 0.4}
            ]
        }

        # Сохраняем портфель
        file_path = self.portfolio_manager.save_portfolio(portfolio)
        self.assertTrue(os.path.exists(file_path))

        # Загружаем портфель
        loaded_portfolio = self.portfolio_manager.load_portfolio(os.path.basename(file_path))

        # Проверяем загруженные данные
        self.assertEqual(loaded_portfolio['name'], portfolio['name'])
        self.assertEqual(loaded_portfolio['description'], portfolio['description'])
        self.assertEqual(len(loaded_portfolio['assets']), len(portfolio['assets']))
        self.assertEqual(loaded_portfolio['assets'][0]['ticker'], 'AAPL')
        self.assertEqual(loaded_portfolio['assets'][1]['ticker'], 'MSFT')

    def test_list_portfolios(self):
        """Тест получения списка портфелей"""
        # Создаем несколько тестовых портфелей
        portfolios = [
            {
                'name': 'Portfolio 1',
                'description': 'Description 1',
                'assets': [{'ticker': 'AAPL', 'weight': 1.0}]
            },
            {
                'name': 'Portfolio 2',
                'description': 'Description 2',
                'assets': [{'ticker': 'MSFT', 'weight': 1.0}]
            }
        ]

        # Сохраняем портфели
        for portfolio in portfolios:
            self.portfolio_manager.save_portfolio(portfolio)

        # Получаем список портфелей
        portfolio_list = self.portfolio_manager.list_portfolios()

        # Проверяем результаты
        self.assertEqual(len(portfolio_list), 2)
        self.assertEqual(portfolio_list[0]['name'], 'Portfolio 1')
        self.assertEqual(portfolio_list[1]['name'], 'Portfolio 2')

    def test_delete_portfolio(self):
        """Тест удаления портфеля"""
        # Создаем тестовый портфель
        portfolio = {
            'name': 'Test Portfolio',
            'assets': [{'ticker': 'AAPL', 'weight': 1.0}]
        }

        # Сохраняем портфель
        file_path = self.portfolio_manager.save_portfolio(portfolio)
        filename = os.path.basename(file_path)

        # Проверяем, что файл существует
        self.assertTrue(os.path.exists(file_path))

        # Удаляем портфель
        result = self.portfolio_manager.delete_portfolio(filename)
        self.assertTrue(result)

        # Проверяем, что файл удален
        self.assertFalse(os.path.exists(file_path))

    @patch('data_fetcher.DataFetcher.get_batch_data')
    def test_update_portfolio_prices(self, mock_get_batch_data):
        """Тест обновления цен портфеля"""
        # Настраиваем мок
        mock_get_batch_data.return_value = {
            'AAPL': pd.DataFrame({
                'Adj Close': [150.0],
                'Close': [149.0]
            }, index=pd.DatetimeIndex(['2023-01-01'])),
            'MSFT': pd.DataFrame({
                'Adj Close': [250.0],
                'Close': [249.0]
            }, index=pd.DatetimeIndex(['2023-01-01']))
        }

        # Создаем тестовый портфель
        portfolio = {
            'name': 'Test Portfolio',
            'assets': [
                {'ticker': 'AAPL', 'weight': 0.5, 'purchase_price': 140.0},
                {'ticker': 'MSFT', 'weight': 0.5, 'purchase_price': 240.0}
            ]
        }

        # Обновляем цены
        updated_portfolio = self.portfolio_manager.update_portfolio_prices(portfolio)

        # Проверяем результаты
        self.assertEqual(updated_portfolio['assets'][0]['current_price'], 150.0)
        self.assertEqual(updated_portfolio['assets'][1]['current_price'], 250.0)

        # Проверяем расчет изменения цены
        self.assertAlmostEqual(updated_portfolio['assets'][0]['price_change_pct'], 7.142857,
                               places=5)  # (150/140 - 1) * 100
        self.assertAlmostEqual(updated_portfolio['assets'][1]['price_change_pct'], 4.166667,
                               places=5)  # (250/240 - 1) * 100


if __name__ == '__main__':
    # Запускаем тесты
    unittest.main()