import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Добавляем путь к директории src для импорта
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Импортируем модуль из src/utils
from utils.data_fetcher import DataFetcher, PortfolioDataManager

def test_historical_data():
    """Тестирование получения исторических данных"""
    print("\n--- Тестирование получения исторических данных ---")

    # Создаем экземпляр DataFetcher
    fetcher = DataFetcher(cache_dir='../src/utils/cache')

    # Получаем исторические данные для популярного тикера
    ticker = 'AAPL'
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    print(f"Получение данных для {ticker} с {start_date} по {end_date}...")
    data = fetcher.get_historical_prices(ticker, start_date, end_date)

    if data is not None and not data.empty:
        print(f"Получено {len(data)} записей. Последние 5:")
        print(data.tail())

        # Построение графика
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['Close'])
        plt.title(f'{ticker} - Цена закрытия')
        plt.xlabel('Дата')
        plt.ylabel('Цена, USD')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{ticker}_close_price.png')
        print(f"График сохранен в {ticker}_close_price.png")
    else:
        print("Не удалось получить данные")


def test_company_info():
    """Тестирование получения информации о компании"""
    print("\n--- Тестирование получения информации о компании ---")

    # Создаем экземпляр DataFetcher
    fetcher = DataFetcher(cache_dir='../src/utils/cache')

    # Получаем информацию для нескольких тикеров
    tickers = ['AAPL', 'MSFT', 'GOOG']

    for ticker in tickers:
        print(f"\nПолучение информации о {ticker}...")
        info = fetcher.get_company_info(ticker)

        if info:
            print(f"Название: {info.get('name', 'Н/Д')}")
            print(f"Сектор: {info.get('sector', 'Н/Д')}")
            print(f"Индустрия: {info.get('industry', 'Н/Д')}")
            print(f"P/E: {info.get('pe_ratio', 'Н/Д')}")
            print(f"Капитализация: {info.get('market_cap', 'Н/Д')}")
        else:
            print("Не удалось получить информацию")


def test_portfolio_creation():
    """Тестирование создания и работы с портфелем"""
    print("\n--- Тестирование создания и работы с портфелем ---")

    # Создаем экземпляры классов
    fetcher = DataFetcher(cache_dir='../src/utils/cache')
    portfolio_manager = PortfolioDataManager(fetcher, storage_dir='../src/utils/portfolios')

    # Создаем портфель из текста
    portfolio_text = """
    AAPL: 0.25
    MSFT: 0.25
    GOOG: 0.2
    AMZN: 0.15
    TSLA: 0.15
    """

    print("Создание портфеля из текста...")
    portfolio = portfolio_manager.create_portfolio_from_text(
        portfolio_text,
        "Тестовый портфель",
        "Портфель технологических компаний для тестирования"
    )

    # Выводим информацию о портфеле
    print(f"\nИмя портфеля: {portfolio['name']}")
    print(f"Описание: {portfolio['description']}")
    print(f"Дата создания: {portfolio.get('created', 'Н/Д')}")

    print("\nАктивы портфеля:")
    for asset in portfolio['assets']:
        print(f"- {asset['ticker']}: {asset.get('weight', 0) * 100:.1f}% "
              f"({asset.get('name', asset['ticker'])})")

    # Сохраняем портфель
    file_path = portfolio_manager.save_portfolio(portfolio)
    print(f"\nПортфель сохранен в {file_path}")

    # Обновляем цены
    print("\nОбновление цен активов...")
    updated_portfolio = portfolio_manager.update_portfolio_prices(portfolio)

    print("\nТекущие цены:")
    for asset in updated_portfolio['assets']:
        price = asset.get('current_price', 'Н/Д')
        change = asset.get('price_change_pct', 'Н/Д')

        if isinstance(change, (int, float)):
            change_str = f"{change:.2f}%"
        else:
            change_str = change

        print(f"- {asset['ticker']}: ${price} (изменение: {change_str})")

    # Список всех портфелей
    print("\nСписок всех портфелей:")
    portfolios = portfolio_manager.list_portfolios()

    for p in portfolios:
        print(f"- {p['name']} ({p['filename']}): {p['asset_count']} активов")


def test_search_tickers():
    """Тестирование поиска тикеров"""
    print("\n--- Тестирование поиска тикеров ---")

    # Создаем экземпляр DataFetcher
    fetcher = DataFetcher(cache_dir='../src/utils/cache')

    # Проверяем, есть ли API ключ для Alpha Vantage
    if not fetcher.api_keys['alpha_vantage']:
        print("API ключ для Alpha Vantage не задан. Пропускаем тест поиска.")
        return

    # Поиск тикеров
    queries = ['Apple', 'Microsoft', 'Tesla']

    for query in queries:
        print(f"\nПоиск по запросу '{query}'...")
        results = fetcher.search_tickers(query, limit=5)

        if results:
            print(f"Найдено {len(results)} результатов:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['symbol']} - {result['name']} ({result.get('type', 'Н/Д')})")
        else:
            print("Результаты не найдены")


def main():
    """Основная функция для запуска тестов"""
    print("=== Ручное тестирование модуля data_fetcher ===")

    # Создаем директории, если их нет
    os.makedirs('../src/utils/cache', exist_ok=True)
    os.makedirs('../src/utils/portfolios', exist_ok=True)

    # Запускаем тесты
    try:
        test_historical_data()
    except Exception as e:
        print(f"Ошибка при тестировании получения исторических данных: {e}")

    try:
        test_company_info()
    except Exception as e:
        print(f"Ошибка при тестировании получения информации о компаниях: {e}")

    try:
        test_portfolio_creation()
    except Exception as e:
        print(f"Ошибка при тестировании создания портфеля: {e}")

    try:
        test_search_tickers()
    except Exception as e:
        print(f"Ошибка при тестировании поиска тикеров: {e}")

    print("\n=== Тестирование завершено ===")


if __name__ == "__main__":
    main()