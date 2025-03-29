# src/pages/portfolio_creation.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os
from datetime import datetime
import pandas as pd

# Добавляем корень проекта в путь Python
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Используем абсолютные импорты
from src.utils.calculations import PortfolioAnalytics
from src.utils.risk_management import RiskManagement
from src.utils.visualization import PortfolioVisualization
import src.config as config


def run(data_fetcher, portfolio_manager):
    """
    Функция для отображения страницы создания портфеля

    Args:
        data_fetcher: Экземпляр DataFetcher для загрузки данных
        portfolio_manager: Экземпляр PortfolioDataManager для работы с портфелями
    """
    st.title("Создание портфеля")

    # Создаем вкладки для разных способов создания портфеля
    tabs = st.tabs([
        "Ручной ввод",
        "Импорт из CSV",
        "Шаблоны",
        "Управление портфелями"
    ])

    # Вкладка "Ручной ввод"
    with tabs[0]:
        create_portfolio_manually(data_fetcher, portfolio_manager)

    # Вкладка "Импорт из CSV"
    with tabs[1]:
        import_portfolio_from_csv(data_fetcher, portfolio_manager)

    # Вкладка "Шаблоны"
    with tabs[2]:
        create_portfolio_from_template(data_fetcher, portfolio_manager)

    # Вкладка "Управление портфелями"
    with tabs[3]:
        manage_portfolios(data_fetcher, portfolio_manager)


def create_portfolio_manually(data_fetcher, portfolio_manager):
    """
    Функция для ручного создания портфеля

    Args:
        data_fetcher: Экземпляр DataFetcher для загрузки данных
        portfolio_manager: Экземпляр PortfolioDataManager для работы с портфелями
    """
    st.header("Ручное создание портфеля")

    # Базовая информация о портфеле
    portfolio_name = st.text_input("Название портфеля", key="manual_portfolio_name")
    portfolio_description = st.text_area("Описание (опционально)", key="manual_description")

    # Ввод тикеров и весов
    st.subheader("Добавление активов")

    # Метод ввода активов
    input_method = st.radio(
        "Способ ввода активов",
        ["Текстовый ввод", "Пошаговое добавление", "Поиск активов"]
    )

    if input_method == "Текстовый ввод":
        st.write("Введите тикеры и веса в формате:")
        st.code("AAPL:0.4, MSFT:0.3, GOOGL:0.3")
        st.write("или")
        st.code("AAPL 0.4, MSFT 0.3, GOOGL 0.3")
        st.write("или по одному на строку:")
        st.code("""
        AAPL:0.4
        MSFT:0.3
        GOOGL:0.3
        """)

        tickers_text = st.text_area("Список тикеров и весов", height=200, key="manual_tickers_text")

        if st.button("Проверить тикеры") and tickers_text.strip():
            try:
                parsed_assets = portfolio_manager.parse_ticker_weights_text(tickers_text)

                if not parsed_assets:
                    st.error("Не удалось распознать ни одного тикера. Проверьте формат ввода.")
                else:
                    st.success(f"Распознано {len(parsed_assets)} активов.")

                    # Проверяем валидность тикеров
                    tickers = [asset['ticker'] for asset in parsed_assets]
                    valid_tickers, invalid_tickers = data_fetcher.validate_tickers(tickers)

                    if invalid_tickers:
                        st.warning(f"Следующие тикеры не найдены: {', '.join(invalid_tickers)}")

                    # Показываем таблицу с распознанными активами
                    assets_df = pd.DataFrame({
                        'Тикер': [asset['ticker'] for asset in parsed_assets],
                        'Вес': [f"{asset['weight'] * 100:.2f}%" for asset in parsed_assets],
                        'Статус': ['✅ Найден' if asset['ticker'] in valid_tickers else '❌ Не найден' for asset in
                                   parsed_assets]
                    })

                    st.dataframe(assets_df, use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка при разборе тикеров: {e}")

        # Создание портфеля по текстовому списку
        if st.button("Создать портфель") and portfolio_name and tickers_text.strip():
            try:
                portfolio = portfolio_manager.create_portfolio_from_text(
                    tickers_text, portfolio_name, portfolio_description
                )
                st.success(f"Портфель '{portfolio_name}' успешно создан с {len(portfolio['assets'])} активами!")

                # Добавьте отладочную печать
                st.write(f"Путь сохранения: {portfolio_manager.storage_dir}")
                st.write(f"Содержимое директории до сохранения: {os.listdir(portfolio_manager.storage_dir)}")

                # Явно сохраняем портфель
                saved_path = portfolio_manager.save_portfolio(portfolio)

                # Еще отладка после сохранения
                st.write(f"Путь сохранения: {saved_path}")
                st.write(f"Содержимое директории после сохранения: {os.listdir(portfolio_manager.storage_dir)}")

                # Показываем итоговую структуру
                st.subheader("Структура созданного портфеля")

                weights_data = {
                    'Тикер': [asset['ticker'] for asset in portfolio['assets']],
                    'Вес': [asset['weight'] for asset in portfolio['assets']]
                }

                fig = px.pie(
                    values=[asset['weight'] for asset in portfolio['assets']],
                    names=[asset['ticker'] for asset in portfolio['assets']],
                    title="Распределение активов"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка создания портфеля: {e}")

    elif input_method == "Пошаговое добавление":
        # Создаем или получаем из состояния список активов
        if 'stepwise_assets' not in st.session_state:
            st.session_state.stepwise_assets = []

        # Форма добавления актива
        with st.form("add_asset_form"):
            st.write("Добавление нового актива")
            ticker = st.text_input("Тикер", key="stepwise_ticker").strip().upper()
            weight = st.slider("Вес (%)", 0, 100, 10, key="stepwise_weight") / 100

            submitted = st.form_submit_button("Добавить актив")

            if submitted and ticker:
                # Проверяем валидность тикера
                valid_tickers, _ = data_fetcher.validate_tickers([ticker])

                if not valid_tickers:
                    st.error(f"Тикер {ticker} не найден. Пожалуйста, проверьте правильность тикера.")
                else:
                    # Добавляем актив в список
                    st.session_state.stepwise_assets.append({
                        'ticker': ticker,
                        'weight': weight
                    })
                    st.success(f"Актив {ticker} успешно добавлен.")

        # Отображаем текущий список активов
        if st.session_state.stepwise_assets:
            st.write("### Текущие активы")

            assets_df = pd.DataFrame({
                'Тикер': [asset['ticker'] for asset in st.session_state.stepwise_assets],
                'Вес': [f"{asset['weight'] * 100:.2f}%" for asset in st.session_state.stepwise_assets]
            })

            st.dataframe(assets_df, use_container_width=True)

            # Визуализация текущего распределения
            total_weight = sum(asset['weight'] for asset in st.session_state.stepwise_assets)

            if total_weight > 0:
                # Нормализуем веса для отображения
                normalized_weights = [asset['weight'] / total_weight for asset in st.session_state.stepwise_assets]

                fig = px.pie(
                    values=normalized_weights,
                    names=[asset['ticker'] for asset in st.session_state.stepwise_assets],
                    title="Распределение активов"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Кнопки для управления списком
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Очистить список"):
                    st.session_state.stepwise_assets = []
                    st.success("Список активов очищен.")

            with col2:
                if st.button("Нормализовать веса"):
                    # Нормализуем веса
                    total_weight = sum(asset['weight'] for asset in st.session_state.stepwise_assets)

                    if total_weight > 0:
                        for asset in st.session_state.stepwise_assets:
                            asset['weight'] = asset['weight'] / total_weight

                        st.success("Веса нормализованы.")

            # Создание портфеля из пошагово добавленных активов
            if st.button("Создать портфель") and portfolio_name and st.session_state.stepwise_assets:
                try:
                    # Создаем текстовое представление для функции создания портфеля
                    tickers_text = "\n".join(
                        [f"{asset['ticker']}:{asset['weight']}" for asset in st.session_state.stepwise_assets])

                    portfolio = portfolio_manager.create_portfolio_from_text(
                        tickers_text, portfolio_name, portfolio_description
                    )

                    st.success(f"Портфель '{portfolio_name}' успешно создан с {len(portfolio['assets'])} активами!")

                    # Очищаем список для создания нового портфеля
                    st.session_state.stepwise_assets = []
                except Exception as e:
                    st.error(f"Ошибка создания портфеля: {e}")

    elif input_method == "Поиск активов":
        # Поисковая форма
        search_query = st.text_input("Поиск активов (введите название компании или тикер)", key="search_query")

        if search_query:
            with st.spinner('Поиск активов...'):
                # Ищем активы
                search_results = data_fetcher.search_tickers(search_query, limit=10)

                if not search_results:
                    st.info(f"По запросу '{search_query}' ничего не найдено.")
                else:
                    st.success(f"Найдено {len(search_results)} активов.")

                    # Отображаем результаты поиска
                    results_df = pd.DataFrame(search_results)

                    # Форматируем таблицу
                    if 'symbol' in results_df.columns and 'name' in results_df.columns:
                        results_df = results_df[['symbol', 'name', 'type', 'region', 'currency']]
                        results_df.columns = ['Тикер', 'Название', 'Тип', 'Регион', 'Валюта']

                    st.dataframe(results_df, use_container_width=True)

                    # Добавление выбранного актива
                    selected_ticker = st.selectbox("Выберите актив для добавления",
                                                   [f"{result['symbol']} - {result['name']}" for result in
                                                    search_results])

                    # Извлекаем тикер из выбранной строки
                    if selected_ticker:
                        ticker = selected_ticker.split(" - ")[0]

                        # Вес для выбранного актива
                        weight = st.slider("Вес (%)", 0, 100, 10, key="search_weight") / 100

                        if st.button("Добавить в портфель"):
                            # Создаем или получаем из состояния список активов
                            if 'search_assets' not in st.session_state:
                                st.session_state.search_assets = []

                            # Добавляем актив в список
                            st.session_state.search_assets.append({
                                'ticker': ticker,
                                'weight': weight
                            })

                            st.success(f"Актив {ticker} успешно добавлен в портфель.")

        # Отображаем текущий список активов из поиска
        if 'search_assets' in st.session_state and st.session_state.search_assets:
            st.write("### Текущие активы")

            assets_df = pd.DataFrame({
                'Тикер': [asset['ticker'] for asset in st.session_state.search_assets],
                'Вес': [f"{asset['weight'] * 100:.2f}%" for asset in st.session_state.search_assets]
            })

            st.dataframe(assets_df, use_container_width=True)

            # Визуализация текущего распределения
            total_weight = sum(asset['weight'] for asset in st.session_state.search_assets)

            if total_weight > 0:
                # Нормализуем веса для отображения
                normalized_weights = [asset['weight'] / total_weight for asset in st.session_state.search_assets]

                fig = px.pie(
                    values=normalized_weights,
                    names=[asset['ticker'] for asset in st.session_state.search_assets],
                    title="Распределение активов"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Кнопки для управления списком
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Очистить список", key="clear_search_list"):
                    st.session_state.search_assets = []
                    st.success("Список активов очищен.")

            with col2:
                if st.button("Нормализовать веса", key="normalize_search_weights"):
                    # Нормализуем веса
                    total_weight = sum(asset['weight'] for asset in st.session_state.search_assets)

                    if total_weight > 0:
                        for asset in st.session_state.search_assets:
                            asset['weight'] = asset['weight'] / total_weight

                        st.success("Веса нормализованы.")

            # Создание портфеля из найденных активов
            if st.button("Создать портфель",
                         key="create_search_portfolio") and portfolio_name and st.session_state.search_assets:
                try:
                    # Создаем текстовое представление для функции создания портфеля
                    tickers_text = "\n".join(
                        [f"{asset['ticker']}:{asset['weight']}" for asset in st.session_state.search_assets])

                    portfolio = portfolio_manager.create_portfolio_from_text(
                        tickers_text, portfolio_name, portfolio_description
                    )

                    st.success(f"Портфель '{portfolio_name}' успешно создан с {len(portfolio['assets'])} активами!")

                    # Очищаем список для создания нового портфеля
                    st.session_state.search_assets = []
                except Exception as e:
                    st.error(f"Ошибка создания портфеля: {e}")


def import_portfolio_from_csv(data_fetcher, portfolio_manager):
    """
    Функция для импорта портфеля из CSV-файла

    Args:
        data_fetcher: Экземпляр DataFetcher для загрузки данных
        portfolio_manager: Экземпляр PortfolioDataManager для работы с портфелями
    """
    st.header("Импорт портфеля из CSV")

    # Базовая информация о портфеле
    portfolio_name = st.text_input("Название портфеля", key="csv_portfolio_name")

    # Инструкция по формату CSV
    with st.expander("Формат CSV-файла"):
        st.write("""
        CSV-файл должен содержать как минимум столбец 'ticker' с тикерами активов.

        Дополнительные столбцы, которые могут быть включены:
        - 'weight': вес актива в портфеле (если не указан, будут использованы равные веса)
        - 'quantity': количество единиц актива
        - 'purchase_price': цена покупки
        - 'purchase_date': дата покупки в формате YYYY-MM-DD
        - 'sector': сектор
        - 'asset_class': класс актива
        - 'region': регион
        - 'currency': валюта

        Пример:
        ```
        ticker,weight,quantity,purchase_price
        AAPL,0.4,10,150.5
        MSFT,0.3,5,250.75
        GOOGL,0.3,2,2500.0
        ```
        """)

    # Загрузка CSV-файла
    uploaded_file = st.file_uploader("Выберите CSV-файл", type="csv")

    if uploaded_file is not None:
        try:
            # Предварительный просмотр CSV
            df = pd.read_csv(uploaded_file)

            st.write("### Предварительный просмотр данных")
            st.dataframe(df.head(10), use_container_width=True)

            # Проверка наличия обязательного столбца
            if 'ticker' not in df.columns:
                st.error("CSV-файл должен содержать столбец 'ticker' с тикерами активов.")
            else:
                # Импорт портфеля
                if st.button("Импортировать портфель") and portfolio_name:
                    with st.spinner('Импорт портфеля...'):
                        # Сохраняем файл временно
                        temp_file = "./data/temp_upload.csv"
                        with open(temp_file, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Импортируем портфель
                        portfolio = portfolio_manager.import_from_csv(temp_file, portfolio_name)

                        # Сохраняем портфель
                        portfolio_manager.save_portfolio(portfolio)

                        st.success(
                            f"Портфель '{portfolio_name}' успешно импортирован с {len(portfolio['assets'])} активами!")

                        # Визуализация импортированного портфеля
                        st.subheader("Структура импортированного портфеля")

                        # Визуализация весов
                        fig = px.pie(
                            values=[asset['weight'] for asset in portfolio['assets']],
                            names=[asset['ticker'] for asset in portfolio['assets']],
                            title="Распределение активов"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка при обработке CSV-файла: {e}")


def create_portfolio_from_template(data_fetcher, portfolio_manager):
    """
    Функция для создания портфеля на основе шаблонов

    Args:
        data_fetcher: Экземпляр DataFetcher для загрузки данных
        portfolio_manager: Экземпляр PortfolioDataManager для работы с портфелями
    """
    st.header("Создание портфеля из шаблона")

    # Базовая информация о портфеле
    portfolio_name = st.text_input("Название портфеля", key="template_portfolio_name")
    portfolio_description = st.text_area("Описание (опционально)", key="template_description")

    # Список шаблонов
    templates = {
        "S&P 500 Top 10": {
            "description": "10 крупнейших компаний индекса S&P 500",
            "assets": [
                {"ticker": "AAPL", "weight": 0.20, "name": "Apple Inc.", "sector": "Technology"},
                {"ticker": "MSFT", "weight": 0.18, "name": "Microsoft Corporation", "sector": "Technology"},
                {"ticker": "AMZN", "weight": 0.12, "name": "Amazon.com Inc.", "sector": "Consumer Discretionary"},
                {"ticker": "NVDA", "weight": 0.10, "name": "NVIDIA Corporation", "sector": "Technology"},
                {"ticker": "GOOGL", "weight": 0.09, "name": "Alphabet Inc. Class A",
                 "sector": "Communication Services"},
                {"ticker": "META", "weight": 0.08, "name": "Meta Platforms Inc.", "sector": "Communication Services"},
                {"ticker": "GOOG", "weight": 0.07, "name": "Alphabet Inc. Class C", "sector": "Communication Services"},
                {"ticker": "BRK.B", "weight": 0.06, "name": "Berkshire Hathaway Inc. Class B", "sector": "Financials"},
                {"ticker": "TSLA", "weight": 0.05, "name": "Tesla, Inc.", "sector": "Consumer Discretionary"},
                {"ticker": "UNH", "weight": 0.05, "name": "UnitedHealth Group Incorporated", "sector": "Healthcare"}
            ]
        },
        "Классический 60/40": {
            "description": "Классическое распределение: 60% акции, 40% облигации",
            "assets": [
                {"ticker": "VOO", "weight": 0.40, "name": "Vanguard S&P 500 ETF", "sector": "Equities",
                 "asset_class": "ETF"},
                {"ticker": "VEA", "weight": 0.20, "name": "Vanguard FTSE Developed Markets ETF", "sector": "Equities",
                 "asset_class": "ETF"},
                {"ticker": "BND", "weight": 0.30, "name": "Vanguard Total Bond Market ETF", "sector": "Fixed Income",
                 "asset_class": "ETF"},
                {"ticker": "BNDX", "weight": 0.10, "name": "Vanguard Total International Bond ETF",
                 "sector": "Fixed Income", "asset_class": "ETF"}
            ]
        },
        "Портфель постоянных весов": {
            "description": "Равномерное распределение между классами активов",
            "assets": [
                {"ticker": "VTI", "weight": 0.25, "name": "Vanguard Total Stock Market ETF", "sector": "Equities",
                 "asset_class": "ETF"},
                {"ticker": "TLT", "weight": 0.25, "name": "iShares 20+ Year Treasury Bond ETF",
                 "sector": "Fixed Income", "asset_class": "ETF"},
                {"ticker": "GLD", "weight": 0.25, "name": "SPDR Gold Shares", "sector": "Commodities",
                 "asset_class": "ETF"},
                {"ticker": "IEF", "weight": 0.25, "name": "iShares 7-10 Year Treasury Bond ETF",
                 "sector": "Fixed Income", "asset_class": "ETF"}
            ]
        },
        "Дивидендный портфель": {
            "description": "Портфель, ориентированный на стабильные дивиденды",
            "assets": [
                {"ticker": "VYM", "weight": 0.20, "name": "Vanguard High Dividend Yield ETF", "sector": "Equities",
                 "asset_class": "ETF"},
                {"ticker": "SCHD", "weight": 0.20, "name": "Schwab US Dividend Equity ETF", "sector": "Equities",
                 "asset_class": "ETF"},
                {"ticker": "PG", "weight": 0.10, "name": "Procter & Gamble Co", "sector": "Consumer Staples"},
                {"ticker": "JNJ", "weight": 0.10, "name": "Johnson & Johnson", "sector": "Healthcare"},
                {"ticker": "KO", "weight": 0.10, "name": "Coca-Cola Co", "sector": "Consumer Staples"},
                {"ticker": "PEP", "weight": 0.10, "name": "PepsiCo, Inc.", "sector": "Consumer Staples"},
                {"ticker": "MCD", "weight": 0.10, "name": "McDonald's Corp", "sector": "Consumer Discretionary"},
                {"ticker": "MMM", "weight": 0.10, "name": "3M Co", "sector": "Industrials"}
            ]
        },
        "Технологический портфель": {
            "description": "Портфель, ориентированный на технологический сектор",
            "assets": [
                {"ticker": "AAPL", "weight": 0.15, "name": "Apple Inc.", "sector": "Technology"},
                {"ticker": "MSFT", "weight": 0.15, "name": "Microsoft Corporation", "sector": "Technology"},
                {"ticker": "AMZN", "weight": 0.10, "name": "Amazon.com Inc.", "sector": "Consumer Discretionary"},
                {"ticker": "NVDA", "weight": 0.10, "name": "NVIDIA Corporation", "sector": "Technology"},
                {"ticker": "GOOGL", "weight": 0.10, "name": "Alphabet Inc. Class A",
                 "sector": "Communication Services"},
                {"ticker": "META", "weight": 0.10, "name": "Meta Platforms Inc.", "sector": "Communication Services"},
                {"ticker": "TSLA", "weight": 0.10, "name": "Tesla, Inc.", "sector": "Consumer Discretionary"},
                {"ticker": "ADBE", "weight": 0.07, "name": "Adobe Inc.", "sector": "Technology"},
                {"ticker": "CRM", "weight": 0.07, "name": "Salesforce, Inc.", "sector": "Technology"},
                {"ticker": "PYPL", "weight": 0.06, "name": "PayPal Holdings, Inc.", "sector": "Financials"}
            ]
        }
    }

    # Выбор шаблона
    selected_template = st.selectbox(
        "Выберите шаблон портфеля",
        list(templates.keys()),
        format_func=lambda x: f"{x} - {templates[x]['description']}",
        key="template_selection"
    )

    if selected_template:
        template = templates[selected_template]

        st.subheader(f"Шаблон: {selected_template}")
        st.write(template["description"])

        # Отображаем состав шаблона
        template_df = pd.DataFrame({
            'Тикер': [asset['ticker'] for asset in template['assets']],
            'Название': [asset.get('name', '') for asset in template['assets']],
            'Вес': [f"{asset['weight'] * 100:.2f}%" for asset in template['assets']],
            'Сектор': [asset.get('sector', 'N/A') for asset in template['assets']]
        })

        st.dataframe(template_df, use_container_width=True)

        # Визуализация шаблона
        fig = px.pie(
            values=[asset['weight'] for asset in template['assets']],
            names=[asset['ticker'] for asset in template['assets']],
            title="Распределение активов в шаблоне"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Создание портфеля из шаблона
        if st.button("Создать портфель из шаблона") and portfolio_name:
            with st.spinner('Создание портфеля...'):
                try:
                    # Проверяем валидность тикеров
                    tickers = [asset['ticker'] for asset in template['assets']]
                    valid_tickers, invalid_tickers = data_fetcher.validate_tickers(tickers)

                    if invalid_tickers:
                        st.warning(
                            f"Следующие тикеры не найдены: {', '.join(invalid_tickers)}. Они будут исключены из портфеля.")

                    # Создаем структуру портфеля
                    assets = []
                    for asset in template['assets']:
                        if asset['ticker'] in valid_tickers:
                            assets.append(asset.copy())

                    portfolio_data = {
                        'name': portfolio_name,
                        'description': portfolio_description or template['description'],
                        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'assets': assets
                    }

                    # Нормализуем веса, если были исключены невалидные тикеры
                    if invalid_tickers:
                        total_weight = sum(asset['weight'] for asset in portfolio_data['assets'])
                        for asset in portfolio_data['assets']:
                            asset['weight'] = asset['weight'] / total_weight

                    # Обогащаем данные портфеля дополнительной информацией
                    portfolio_manager._enrich_portfolio_data(portfolio_data)

                    # Сохраняем портфель
                    portfolio_manager.save_portfolio(portfolio_data)

                    st.success(
                        f"Портфель '{portfolio_name}' успешно создан с {len(portfolio_data['assets'])} активами!")
                except Exception as e:
                    st.error(f"Ошибка при создании портфеля: {e}")


def manage_portfolios(data_fetcher, portfolio_manager):
    """
    Функция для управления существующими портфелями

    Args:
        data_fetcher: Экземпляр DataFetcher для загрузки данных
        portfolio_manager: Экземпляр PortfolioDataManager для работы с портфелями
    """
    st.header("Управление портфелями")

    # Получаем список портфелей
    portfolios = portfolio_manager.list_portfolios()

    if not portfolios:
        st.info("Портфели не найдены. Создайте портфель в одном из разделов.")
        return

    # Отображаем список портфелей
    st.subheader("Список портфелей")

    portfolios_df = pd.DataFrame({
        'Название': [p['name'] for p in portfolios],
        'Активов': [p['asset_count'] for p in portfolios],
        'Последнее обновление': [p['last_updated'] for p in portfolios]
    })

    st.dataframe(portfolios_df, use_container_width=True)

    # Выбор портфеля для действий
    selected_portfolio = st.selectbox(
        "Выберите портфель для действий",
        [p['name'] for p in portfolios],
        key="manage_portfolio_selection"
    )

    if selected_portfolio:
        # Доступные действия
        action = st.radio(
            "Выберите действие",
            ["Просмотр", "Экспорт в CSV", "Дублирование", "Удаление"],
            key="portfolio_action"
        )

        if action == "Просмотр":
            # Загружаем данные портфеля
            portfolio_data = portfolio_manager.load_portfolio(selected_portfolio)

            st.subheader(f"Портфель: {portfolio_data['name']}")

            if 'description' in portfolio_data and portfolio_data['description']:
                st.write(portfolio_data['description'])

            # Отображаем список активов
            assets_data = []
            for asset in portfolio_data['assets']:
                asset_row = {
                    'Тикер': asset['ticker'],
                    'Вес (%)': f"{asset['weight'] * 100:.2f}%"
                }

                # Добавляем доступную информацию
                for field in ['name', 'sector', 'industry', 'asset_class', 'currency']:
                    if field in asset:
                        asset_row[field.capitalize()] = asset[field]

                if 'current_price' in asset:
                    asset_row['Текущая цена'] = asset['current_price']

                if 'price_change_pct' in asset:
                    asset_row['Изменение (%)'] = f"{asset['price_change_pct']:.2f}%"

                assets_data.append(asset_row)

            st.dataframe(pd.DataFrame(assets_data), use_container_width=True)

            # Визуализация распределения активов
            fig = px.pie(
                values=[asset['weight'] for asset in portfolio_data['assets']],
                names=[asset['ticker'] for asset in portfolio_data['assets']],
                title="Распределение активов"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Если есть секторное распределение, отображаем его
            sectors = {}
            for asset in portfolio_data['assets']:
                if 'sector' in asset and asset['sector'] != 'N/A':
                    sector = asset['sector']
                    if sector in sectors:
                        sectors[sector] += asset['weight']
                    else:
                        sectors[sector] = asset['weight']

            if sectors:
                fig_sectors = px.pie(
                    values=list(sectors.values()),
                    names=list(sectors.keys()),
                    title="Распределение по секторам"
                )
                st.plotly_chart(fig_sectors, use_container_width=True)

        elif action == "Экспорт в CSV":
            # Загружаем данные портфеля
            portfolio_data = portfolio_manager.load_portfolio(selected_portfolio)

            # Экспортируем в CSV
            try:
                csv_path = portfolio_manager.export_to_csv(portfolio_data)

                st.success(f"Портфель '{selected_portfolio}' успешно экспортирован в CSV: {csv_path}")

                # Создаем кнопку для скачивания
                with open(csv_path, 'r') as f:
                    csv_content = f.read()

                st.download_button(
                    label="Скачать CSV-файл",
                    data=csv_content,
                    file_name=f"{selected_portfolio}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Ошибка при экспорте портфеля: {e}")

        elif action == "Дублирование":
            # Новое имя для дубликата
            new_name = st.text_input("Введите имя для нового портфеля",
                                     value=f"{selected_portfolio} (копия)",
                                     key="duplicate_name")

            if st.button("Дублировать портфель") and new_name:
                try:
                    # Загружаем данные оригинального портфеля
                    portfolio_data = portfolio_manager.load_portfolio(selected_portfolio)

                    # Создаем копию с новым именем
                    portfolio_data['name'] = new_name
                    portfolio_data['created'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    portfolio_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # Сохраняем новый портфель
                    portfolio_manager.save_portfolio(portfolio_data)

                    st.success(f"Портфель '{selected_portfolio}' успешно дублирован как '{new_name}'!")
                except Exception as e:
                    st.error(f"Ошибка при дублировании портфеля: {e}")

        elif action == "Удаление":
            st.warning(f"Вы собираетесь удалить портфель '{selected_portfolio}'. Это действие нельзя отменить.")

            # Подтверждение удаления
            if st.button("Подтвердить удаление"):
                try:
                    # Получаем имя файла портфеля
                    portfolio_file = next((p['filename'] for p in portfolios if p['name'] == selected_portfolio), None)

                    if portfolio_file:
                        # Удаляем портфель
                        portfolio_manager.delete_portfolio(portfolio_file)

                        st.success(f"Портфель '{selected_portfolio}' успешно удален!")
                    else:
                        st.error("Не удалось найти файл портфеля.")
                except Exception as e:
                    st.error(f"Ошибка при удалении портфеля: {e}")