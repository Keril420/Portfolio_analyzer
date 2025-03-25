import os
import pandas as pd
from data_fetcher import DataFetcher, PortfolioDataManager


def test_data_fetcher():
    print("Тестирование DataFetcher...")

    # Создаем экземпляр DataFetcher
    fetcher = DataFetcher(cache_dir='./test_cache')

    # Проверяем получение исторических цен
    print("\nПолучаем исторические данные для AAPL...")
    apple_data = fetcher.get_historical_prices('AAPL', start_date='2023-01-01', end_date='2023-01-31')
    print(f"Получено записей: {len(apple_data)}")
    print(apple_data.head())

    # Проверяем получение информации о компании
    print("\nПолучаем информацию о MSFT...")
    msft_info = fetcher.get_company_info('MSFT')
    print(f"Имя компании: {msft_info.get('name')}")
    print(f"Сектор: {msft_info.get('sector')}")

    # Проверяем получение данных для нескольких тикеров
    print("\nПолучаем данные для нескольких тикеров...")
    batch_data = fetcher.get_batch_data(['AAPL', 'MSFT', 'GOOGL'], start_date='2023-01-01', end_date='2023-01-31')
    print(f"Получено данных для {len(batch_data)} тикеров")

    # Проверяем валидацию тикеров
    print("\nПроверяем валидность тикеров...")
    valid, invalid = fetcher.validate_tickers(['AAPL', 'MSFT', 'INVALID_TICKER'])
    print(f"Валидные тикеры: {valid}")
    print(f"Невалидные тикеры: {invalid}")

    # Очищаем кеш для тестов
    fetcher.clear_cache(['AAPL'])
    print("\nКеш для AAPL очищен")

    return fetcher


def test_portfolio_manager(fetcher):
    print("\nТестирование PortfolioDataManager...")

    # Создаем экземпляр PortfolioDataManager
    manager = PortfolioDataManager(fetcher, storage_dir='./test_portfolios')

    # Тестируем создание портфеля из текста
    text_input = """
    AAPL: 0.4
    MSFT: 0.3
    GOOGL: 0.3
    """
    print("\nСоздаем портфель из текста...")
    portfolio = manager.create_portfolio_from_text(text_input, "Test Portfolio", "Тестовый портфель")
    print(f"Создан портфель '{portfolio['name']}' с {len(portfolio['assets'])} активами")

    # Выводим активы портфеля
    print("\nАктивы в портфеле:")
    for asset in portfolio['assets']:
        print(f"  {asset['ticker']}: {asset['weight']:.2f}")

    # Сохраняем портфель
    print("\nСохраняем портфель...")
    save_path = manager.save_portfolio(portfolio)
    print(f"Портфель сохранен в {save_path}")

    # Загружаем портфель
    print("\nЗагружаем портфель...")
    loaded_portfolio = manager.load_portfolio("Test Portfolio")
    print(f"Загружен портфель '{loaded_portfolio['name']}'")

    # Экспортируем в CSV
    print("\nЭкспортируем портфель в CSV...")
    csv_path = manager.export_to_csv(portfolio)
    print(f"Портфель экспортирован в {csv_path}")

    # Проверяем список портфелей
    print("\nСписок доступных портфелей:")
    portfolios = manager.list_portfolios()
    for p in portfolios:
        print(f"  {p['name']} ({p['asset_count']} активов)")

    # Обновляем цены
    print("\nОбновляем цены...")
    updated_portfolio = manager.update_portfolio_prices(portfolio)

    # Удаляем тестовый портфель
    print("\nУдаляем тестовый портфель...")
    manager.delete_portfolio("Test Portfolio")
    print("Портфель удален")


if __name__ == "__main__":
    fetcher = test_data_fetcher()
    test_portfolio_manager(fetcher)

    print("\nВсе тесты завершены!")