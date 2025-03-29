# src/pages/portfolio_optimization.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Добавляем корень проекта в путь Python
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Используем абсолютные импорты
from src.utils.optimization import PortfolioOptimizer
from src.utils.calculations import PortfolioAnalytics
from src.utils.risk_management import RiskManagement
from src.utils.visualization import PortfolioVisualization
import src.config as config


def run(data_fetcher, portfolio_manager):
    """
    Функция для отображения страницы оптимизации портфеля

    Args:
        data_fetcher: Экземпляр DataFetcher для загрузки данных
        portfolio_manager: Экземпляр PortfolioDataManager для работы с портфелями
    """
    st.title("Оптимизация портфеля")

    # Создаем вкладки для разных типов оптимизации
    tabs = st.tabs([
        "Существующий портфель",
        "Новый портфель",
        "Тактическое распределение",
        "Монте-Карло симуляция"
    ])

    # Вкладка "Существующий портфель"
    with tabs[0]:
        optimize_existing_portfolio(data_fetcher, portfolio_manager)

    # Вкладка "Новый портфель"
    with tabs[1]:
        optimize_new_portfolio(data_fetcher, portfolio_manager)

    # Вкладка "Тактическое распределение"
    with tabs[2]:
        tactical_allocation(data_fetcher, portfolio_manager)

    # Вкладка "Монте-Карло симуляция"
    with tabs[3]:
        monte_carlo_simulation(data_fetcher, portfolio_manager)


def optimize_existing_portfolio(data_fetcher, portfolio_manager):
    """
    Функция для оптимизации существующего портфеля

    Args:
        data_fetcher: Экземпляр DataFetcher для загрузки данных
        portfolio_manager: Экземпляр PortfolioDataManager для работы с портфелями
    """
    st.header("Оптимизация существующего портфеля")

    # Получаем список портфелей
    portfolios = portfolio_manager.list_portfolios()

    if not portfolios:
        st.info("Портфели не найдены. Создайте портфель в разделе 'Создание портфеля'.")
        return

    # Выбор портфеля для оптимизации
    portfolio_names = [p['name'] for p in portfolios]
    selected_portfolio = st.selectbox("Выберите портфель для оптимизации", portfolio_names)

    if not selected_portfolio:
        return

    # Загружаем данные портфеля
    portfolio_data = portfolio_manager.load_portfolio(selected_portfolio)

    # Отображаем основную информацию о портфеле
    st.subheader(f"Портфель: {portfolio_data['name']}")

    if 'description' in portfolio_data and portfolio_data['description']:
        st.write(portfolio_data['description'])

    # Параметры оптимизации
    col1, col2, col3 = st.columns(3)

    with col1:
        method = st.selectbox(
            "Метод оптимизации",
            [
                "maximum_sharpe",
                "minimum_variance",
                "risk_parity",
                "markowitz",
                "equal_weight"
            ],
            format_func=lambda x: {
                "maximum_sharpe": "Максимальный коэффициент Шарпа",
                "minimum_variance": "Минимальная дисперсия",
                "risk_parity": "Равный риск (Risk Parity)",
                "markowitz": "Марковиц (эффективная граница)",
                "equal_weight": "Равные веса"
            }.get(x, x)
        )

    with col2:
        start_date = st.date_input(
            "Начальная дата",
            datetime.now() - timedelta(days=365)
        )

    with col3:
        end_date = st.date_input(
            "Конечная дата",
            datetime.now()
        )

    # Дополнительные параметры в зависимости от выбранного метода
    if method == "markowitz":
        col1, col2 = st.columns(2)

        with col1:
            target_return = st.slider(
                "Целевая годовая доходность (%)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5
            ) / 100

            st.write(f"**Целевая доходность:** {target_return * 100:.1f}%")

        with col2:
            # Риск-фри ставка
            risk_free_rate = st.slider(
                "Безрисковая ставка (%)",
                min_value=0.0,
                max_value=10.0,
                value=config.RISK_FREE_RATE * 100,
                step=0.25
            ) / 100

            st.write(f"**Безрисковая ставка:** {risk_free_rate * 100:.2f}%")
    else:
        # Для других методов только риск-фри ставка
        risk_free_rate = st.slider(
            "Безрисковая ставка (%)",
            min_value=0.0,
            max_value=10.0,
            value=config.RISK_FREE_RATE * 100,
            step=0.25
        ) / 100

        # Устанавливаем target_return в None для других методов
        target_return = None

    # Ограничения весов
    col1, col2 = st.columns(2)

    with col1:
        min_weight = st.slider(
            "Минимальный вес актива (%)",
            min_value=0.0,
            max_value=50.0,
            value=1.0,
            step=0.5
        ) / 100

    with col2:
        max_weight = st.slider(
            "Максимальный вес актива (%)",
            min_value=10.0,
            max_value=100.0,
            value=30.0,
            step=5.0
        ) / 100

    # Кнопка для запуска оптимизации
    if st.button("Оптимизировать портфель"):
        # Загрузка данных
        with st.spinner('Загрузка исторических данных...'):
            # Получаем тикеры из портфеля
            tickers = [asset['ticker'] for asset in portfolio_data['assets']]

            # Конвертируем даты в строки
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')

            # Загружаем исторические цены для всех активов
            prices_data = data_fetcher.get_batch_data(tickers, start_date_str, end_date_str)

            # Проверяем, что данные загружены успешно
            if not prices_data or all(df.empty for df in prices_data.values()):
                st.error("Не удалось загрузить исторические данные. Пожалуйста, проверьте тикеры или измените период.")
                return

            # Создаем DataFrame с ценами закрытия
            close_prices = pd.DataFrame()

            for ticker, df in prices_data.items():
                if not df.empty:
                    if 'Adj Close' in df.columns:
                        close_prices[ticker] = df['Adj Close']
                    elif 'Close' in df.columns:
                        close_prices[ticker] = df['Close']

            # Рассчитываем доходности
            returns = PortfolioAnalytics.calculate_returns(close_prices)

        # Оптимизация портфеля
        with st.spinner('Оптимизация портфеля...'):
            # Получаем текущие веса из портфеля
            current_weights = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

            # Выполняем оптимизацию
            optimization_args = {
                'risk_free_rate': risk_free_rate,
                'min_weight': min_weight,
                'max_weight': max_weight
            }

            # Добавляем target_return для метода Марковица
            if method == "markowitz" and target_return is not None:
                optimization_args['target_return'] = target_return

            # Запускаем оптимизацию
            optimization_result = PortfolioOptimizer.optimize_portfolio(
                returns, method=method, **optimization_args
            )

            # Проверяем результат
            if 'error' in optimization_result:
                st.error(f"Ошибка оптимизации: {optimization_result['error']}")
                return

        # Отображение результатов оптимизации
        st.subheader("Результаты оптимизации")

        # Метрики оптимизированного портфеля
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Ожидаемая годовая доходность",
                f"{optimization_result['expected_return'] * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Ожидаемая волатильность",
                f"{optimization_result['expected_risk'] * 100:.2f}%"
            )

        with col3:
            if 'sharpe_ratio' in optimization_result:
                st.metric(
                    "Коэффициент Шарпа",
                    f"{optimization_result['sharpe_ratio']:.2f}"
                )

        # Сравнение текущих и оптимальных весов
        st.subheader("Сравнение весов")

        # Создаем DataFrame для сравнения
        weights_comparison = pd.DataFrame({
            'Тикер': list(current_weights.keys()),
            'Текущий вес (%)': [current_weights[ticker] * 100 for ticker in current_weights],
            'Оптимальный вес (%)': [optimization_result['optimal_weights'].get(ticker, 0) * 100 for ticker in
                                    current_weights]
        })

        # Добавляем разницу
        weights_comparison['Изменение (%)'] = weights_comparison['Оптимальный вес (%)'] - weights_comparison[
            'Текущий вес (%)']

        # Рассчитываем абсолютное изменение (для цветовой шкалы)
        weights_comparison['abs_change'] = abs(weights_comparison['Изменение (%)'])

        # Сортируем по абсолютному изменению
        weights_comparison = weights_comparison.sort_values('abs_change', ascending=False)

        # Стилизованная таблица
        st.dataframe(
            weights_comparison[['Тикер', 'Текущий вес (%)', 'Оптимальный вес (%)', 'Изменение (%)']],
            use_container_width=True
        )

        # Визуализация изменения весов
        fig_weights = go.Figure()

        fig_weights.add_trace(go.Bar(
            x=weights_comparison['Тикер'],
            y=weights_comparison['Текущий вес (%)'],
            name='Текущий вес (%)',
            marker_color='lightgrey'
        ))

        fig_weights.add_trace(go.Bar(
            x=weights_comparison['Тикер'],
            y=weights_comparison['Оптимальный вес (%)'],
            name='Оптимальный вес (%)',
            marker_color='royalblue'
        ))

        fig_weights.update_layout(
            title='Сравнение текущих и оптимальных весов',
            barmode='group',
            xaxis_title='Актив',
            yaxis_title='Вес (%)',
            legend_title='',
            hovermode='x unified'
        )

        st.plotly_chart(fig_weights, use_container_width=True)

        # Эффективная граница для метода Марковица
        if method == "markowitz" and 'efficient_frontier' in optimization_result:
            st.subheader("Эффективная граница")

            # Создаем DataFrame для эффективной границы
            ef_df = pd.DataFrame(optimization_result['efficient_frontier'])

            # Рассчитываем текущую доходность и риск
            portfolio_return = np.sum([current_weights[ticker] * returns[ticker].mean() * 252
                                       for ticker in current_weights if ticker in returns.columns])

            filtered_returns = returns[[ticker for ticker in current_weights if ticker in returns.columns]]
            cov_matrix = filtered_returns.cov() * 252
            weight_array = np.array([current_weights[ticker] for ticker in filtered_returns.columns])
            portfolio_risk = np.sqrt(np.dot(weight_array.T, np.dot(cov_matrix, weight_array)))

            # Визуализация эффективной границы
            fig_ef = go.Figure()

            # Эффективная граница
            fig_ef.add_trace(go.Scatter(
                x=ef_df['risk'] * 100,
                y=ef_df['return'] * 100,
                mode='lines',
                name='Эффективная граница',
                line=dict(color='blue', width=2)
            ))

            # Текущий портфель
            fig_ef.add_trace(go.Scatter(
                x=[portfolio_risk * 100],
                y=[portfolio_return * 100],
                mode='markers',
                name='Текущий портфель',
                marker=dict(color='red', size=12, symbol='circle')
            ))

            # Оптимальный портфель
            fig_ef.add_trace(go.Scatter(
                x=[optimization_result['expected_risk'] * 100],
                y=[optimization_result['expected_return'] * 100],
                mode='markers',
                name='Оптимальный портфель',
                marker=dict(color='green', size=12, symbol='star')
            ))

            # Добавляем другие примечательные точки
            # Минимальная дисперсия
            min_var_idx = ef_df['risk'].idxmin()

            fig_ef.add_trace(go.Scatter(
                x=[ef_df.iloc[min_var_idx]['risk'] * 100],
                y=[ef_df.iloc[min_var_idx]['return'] * 100],
                mode='markers',
                name='Минимальная дисперсия',
                marker=dict(color='purple', size=12, symbol='triangle-up')
            ))

            # Максимальный коэффициент Шарпа
            if 'sharpe' in ef_df.columns:
                max_sharpe_idx = ef_df['sharpe'].idxmax()

                fig_ef.add_trace(go.Scatter(
                    x=[ef_df.iloc[max_sharpe_idx]['risk'] * 100],
                    y=[ef_df.iloc[max_sharpe_idx]['return'] * 100],
                    mode='markers',
                    name='Максимальный Шарп',
                    marker=dict(color='gold', size=12, symbol='diamond')
                ))

            fig_ef.update_layout(
                title='Эффективная граница',
                xaxis_title='Ожидаемый риск (%)',
                yaxis_title='Ожидаемая доходность (%)',
                legend_title='',
                hovermode='closest'
            )

            st.plotly_chart(fig_ef, use_container_width=True)

        # Специфическая визуализация для метода Risk Parity
        if method == "risk_parity" and 'risk_contribution' in optimization_result:
            st.subheader("Вклад в риск")

            # Создаем DataFrame для вклада в риск
            risk_contrib_df = pd.DataFrame({
                'Актив': list(optimization_result['risk_contribution'].keys()),
                'Вклад в риск (%)': [v * 100 for v in optimization_result['risk_contribution'].values()]
            })

            # Визуализация вклада в риск
            fig_rc = px.bar(
                risk_contrib_df,
                x='Актив',
                y='Вклад в риск (%)',
                title='Вклад в риск после оптимизации',
                color='Вклад в риск (%)',
                color_continuous_scale='Viridis'
            )

            st.plotly_chart(fig_rc, use_container_width=True)

        # Кнопка применения оптимизации
        if st.button("Применить оптимизацию к портфелю"):
            # Обновляем веса в портфеле
            for asset in portfolio_data['assets']:
                ticker = asset['ticker']
                if ticker in optimization_result['optimal_weights']:
                    asset['weight'] = optimization_result['optimal_weights'][ticker]

            # Сохраняем обновленный портфель
            portfolio_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            portfolio_manager.save_portfolio(portfolio_data)

            st.success(f"Портфель '{selected_portfolio}' успешно оптимизирован!")


def optimize_new_portfolio(data_fetcher, portfolio_manager):
    """
    Функция для создания и оптимизации нового портфеля

    Args:
        data_fetcher: Экземпляр DataFetcher для загрузки данных
        portfolio_manager: Экземпляр PortfolioDataManager для работы с портфелями
    """
    st.header("Создание и оптимизация нового портфеля")

    # Ввод имени для нового портфеля
    portfolio_name = st.text_input("Название нового портфеля")
    portfolio_description = st.text_area("Описание (опционально)")

    # Ввод тикеров
    st.subheader("Добавление активов")
    st.write("Введите тикеры активов через запятую (например, AAPL, MSFT, GOOGL)")
    tickers_input = st.text_input("Тикеры")

    # Проверка и поиск тикеров
    if tickers_input:
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]

        with st.spinner('Проверка тикеров...'):
            # Проверяем валидность тикеров
            valid_tickers, invalid_tickers = data_fetcher.validate_tickers(tickers)

            if invalid_tickers:
                st.warning(f"Следующие тикеры не найдены: {', '.join(invalid_tickers)}")

            if not valid_tickers:
                st.error("Ни один из введенных тикеров не найден. Пожалуйста, проверьте правильность тикеров.")
                return

            st.success(f"Найдено {len(valid_tickers)} действительных тикеров.")

        # Параметры оптимизации
        st.subheader("Параметры оптимизации")

        col1, col2, col3 = st.columns(3)

        with col1:
            method = st.selectbox(
                "Метод оптимизации",
                [
                    "maximum_sharpe",
                    "minimum_variance",
                    "risk_parity",
                    "markowitz",
                    "equal_weight"
                ],
                format_func=lambda x: {
                    "maximum_sharpe": "Максимальный коэффициент Шарпа",
                    "minimum_variance": "Минимальная дисперсия",
                    "risk_parity": "Равный риск (Risk Parity)",
                    "markowitz": "Марковиц (эффективная граница)",
                    "equal_weight": "Равные веса"
                }.get(x, x),
                key="new_portfolio_method"
            )

        with col2:
            start_date = st.date_input(
                "Начальная дата",
                datetime.now() - timedelta(days=365 * 2),
                key="new_portfolio_start_date"
            )

        with col3:
            end_date = st.date_input(
                "Конечная дата",
                datetime.now(),
                key="new_portfolio_end_date"
            )

        # Дополнительные параметры в зависимости от выбранного метода
        if method == "markowitz":
            col1, col2 = st.columns(2)

            with col1:
                target_return = st.slider(
                    "Целевая годовая доходность (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=10.0,
                    step=0.5,
                    key="new_portfolio_target_return"
                ) / 100

                st.write(f"**Целевая доходность:** {target_return * 100:.1f}%")

            with col2:
                # Риск-фри ставка
                risk_free_rate = st.slider(
                    "Безрисковая ставка (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=config.RISK_FREE_RATE * 100,
                    step=0.25,
                    key="new_portfolio_risk_free_rate"
                ) / 100

                st.write(f"**Безрисковая ставка:** {risk_free_rate * 100:.2f}%")
        else:
            # Для других методов только риск-фри ставка
            risk_free_rate = st.slider(
                "Безрисковая ставка (%)",
                min_value=0.0,
                max_value=10.0,
                value=config.RISK_FREE_RATE * 100,
                step=0.25,
                key="new_portfolio_risk_free_rate"
            ) / 100

            # Устанавливаем target_return в None для других методов
            target_return = None

        # Ограничения весов
        col1, col2 = st.columns(2)

        with col1:
            min_weight = st.slider(
                "Минимальный вес актива (%)",
                min_value=0.0,
                max_value=50.0,
                value=1.0,
                step=0.5,
                key="new_portfolio_min_weight"
            ) / 100

        with col2:
            max_weight = st.slider(
                "Максимальный вес актива (%)",
                min_value=10.0,
                max_value=100.0,
                value=30.0,
                step=5.0,
                key="new_portfolio_max_weight"
            ) / 100

        # Кнопка для создания и оптимизации портфеля
        if st.button("Создать и оптимизировать портфель") and portfolio_name:
            # Загрузка данных
            with st.spinner('Загрузка исторических данных...'):
                # Конвертируем даты в строки
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')

                # Загружаем исторические цены для всех активов
                prices_data = data_fetcher.get_batch_data(valid_tickers, start_date_str, end_date_str)

                # Проверяем, что данные загружены успешно
                if not prices_data or all(df.empty for df in prices_data.values()):
                    st.error(
                        "Не удалось загрузить исторические данные. Пожалуйста, проверьте тикеры или измените период.")
                    return

                # Создаем DataFrame с ценами закрытия
                close_prices = pd.DataFrame()

                for ticker, df in prices_data.items():
                    if not df.empty:
                        if 'Adj Close' in df.columns:
                            close_prices[ticker] = df['Adj Close']
                        elif 'Close' in df.columns:
                            close_prices[ticker] = df['Close']

                # Рассчитываем доходности
                returns = PortfolioAnalytics.calculate_returns(close_prices)

            # Оптимизация портфеля
            with st.spinner('Оптимизация портфеля...'):
                # Выполняем оптимизацию
                optimization_args = {
                    'risk_free_rate': risk_free_rate,
                    'min_weight': min_weight,
                    'max_weight': max_weight
                }

                # Добавляем target_return для метода Марковица
                if method == "markowitz" and target_return is not None:
                    optimization_args['target_return'] = target_return

                # Запускаем оптимизацию
                optimization_result = PortfolioOptimizer.optimize_portfolio(
                    returns, method=method, **optimization_args
                )

                # Проверяем результат
                if 'error' in optimization_result:
                    st.error(f"Ошибка оптимизации: {optimization_result['error']}")
                    return

            # Создаем новый портфель
            assets = []
            for ticker, weight in optimization_result['optimal_weights'].items():
                asset = {
                    'ticker': ticker,
                    'weight': weight
                }
                assets.append(asset)

            # Создаем структуру портфеля
            portfolio_data = {
                'name': portfolio_name,
                'description': portfolio_description,
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'assets': assets
            }

            # Обогащаем данные портфеля дополнительной информацией
            portfolio_manager._enrich_portfolio_data(portfolio_data)

            # Сохраняем портфель
            portfolio_manager.save_portfolio(portfolio_data)

            st.success(f"Портфель '{portfolio_name}' успешно создан и оптимизирован!")

            # Отображение результатов оптимизации
            st.subheader("Результаты оптимизации")

            # Метрики оптимизированного портфеля
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Ожидаемая годовая доходность",
                    f"{optimization_result['expected_return'] * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Ожидаемая волатильность",
                    f"{optimization_result['expected_risk'] * 100:.2f}%"
                )

            with col3:
                if 'sharpe_ratio' in optimization_result:
                    st.metric(
                        "Коэффициент Шарпа",
                        f"{optimization_result['sharpe_ratio']:.2f}"
                    )

            # Визуализация весов портфеля
            weights_df = pd.DataFrame({
                'Актив': list(optimization_result['optimal_weights'].keys()),
                'Вес (%)': [w * 100 for w in optimization_result['optimal_weights'].values()]
            }).sort_values('Вес (%)', ascending=False)

            fig_weights = px.bar(
                weights_df,
                x='Актив',
                y='Вес (%)',
                title='Распределение весов оптимизированного портфеля',
                color='Вес (%)',
                color_continuous_scale='Viridis'
            )

            st.plotly_chart(fig_weights, use_container_width=True)

            # Эффективная граница для метода Марковица
            if method == "markowitz" and 'efficient_frontier' in optimization_result:
                st.subheader("Эффективная граница")

                # Создаем DataFrame для эффективной границы
                ef_df = pd.DataFrame(optimization_result['efficient_frontier'])

                # Визуализация эффективной границы
                fig_ef = go.Figure()

                # Эффективная граница
                fig_ef.add_trace(go.Scatter(
                    x=ef_df['risk'] * 100,
                    y=ef_df['return'] * 100,
                    mode='lines',
                    name='Эффективная граница',
                    line=dict(color='blue', width=2)
                ))

                # Оптимальный портфель
                fig_ef.add_trace(go.Scatter(
                    x=[optimization_result['expected_risk'] * 100],
                    y=[optimization_result['expected_return'] * 100],
                    mode='markers',
                    name='Оптимальный портфель',
                    marker=dict(color='green', size=12, symbol='star')
                ))

                # Добавляем другие примечательные точки
                # Минимальная дисперсия
                min_var_idx = ef_df['risk'].idxmin()

                fig_ef.add_trace(go.Scatter(
                    x=[ef_df.iloc[min_var_idx]['risk'] * 100],
                    y=[ef_df.iloc[min_var_idx]['return'] * 100],
                    mode='markers',
                    name='Минимальная дисперсия',
                    marker=dict(color='purple', size=12, symbol='triangle-up')
                ))

                # Максимальный коэффициент Шарпа
                if 'sharpe' in ef_df.columns:
                    max_sharpe_idx = ef_df['sharpe'].idxmax()

                    fig_ef.add_trace(go.Scatter(
                        x=[ef_df.iloc[max_sharpe_idx]['risk'] * 100],
                        y=[ef_df.iloc[max_sharpe_idx]['return'] * 100],
                        mode='markers',
                        name='Максимальный Шарп',
                        marker=dict(color='gold', size=12, symbol='diamond')
                    ))

                fig_ef.update_layout(
                    title='Эффективная граница',
                    xaxis_title='Ожидаемый риск (%)',
                    yaxis_title='Ожидаемая доходность (%)',
                    legend_title='',
                    hovermode='closest'
                )

                st.plotly_chart(fig_ef, use_container_width=True)

            # Специфическая визуализация для метода Risk Parity
            if method == "risk_parity" and 'risk_contribution' in optimization_result:
                st.subheader("Вклад в риск")

                # Создаем DataFrame для вклада в риск
                risk_contrib_df = pd.DataFrame({
                    'Актив': list(optimization_result['risk_contribution'].keys()),
                    'Вклад в риск (%)': [v * 100 for v in optimization_result['risk_contribution'].values()]
                })

                # Визуализация вклада в риск
                fig_rc = px.bar(
                    risk_contrib_df,
                    x='Актив',
                    y='Вклад в риск (%)',
                    title='Вклад в риск оптимизированного портфеля',
                    color='Вклад в риск (%)',
                    color_continuous_scale='Viridis'
                )

                st.plotly_chart(fig_rc, use_container_width=True)


def tactical_allocation(data_fetcher, portfolio_manager):
    """
    Функция для тактического распределения активов

    Args:
        data_fetcher: Экземпляр DataFetcher для загрузки данных
        portfolio_manager: Экземпляр PortfolioDataManager для работы с портфелями
    """
    st.header("Тактическое распределение активов")

    # Получаем список портфелей
    portfolios = portfolio_manager.list_portfolios()

    if not portfolios:
        st.info("Портфели не найдены. Создайте портфель в разделе 'Создание портфеля'.")
        return

    # Выбор портфеля для тактического распределения
    portfolio_names = [p['name'] for p in portfolios]
    selected_portfolio = st.selectbox("Выберите портфель для тактического распределения", portfolio_names)

    if not selected_portfolio:
        return

    # Загружаем данные портфеля
    portfolio_data = portfolio_manager.load_portfolio(selected_portfolio)

    # Отображаем основную информацию о портфеле
    st.subheader(f"Портфель: {portfolio_data['name']}")
    st.write(f"Количество активов: {len(portfolio_data['assets'])}")

    # Параметры тактического распределения
    col1, col2 = st.columns(2)

    with col1:
        tactic_method = st.selectbox(
            "Метод тактического распределения",
            [
                "market_momentum",
                "sector_rotation",
                "volatility_targeting",
                "equal_weight"
            ],
            format_func=lambda x: {
                "market_momentum": "Рыночный момент",
                "sector_rotation": "Секторная ротация",
                "volatility_targeting": "Таргетирование волатильности",
                "equal_weight": "Равные веса"
            }.get(x, x)
        )

    with col2:
        if tactic_method != "equal_weight":
            adjustment_strength = st.slider(
                "Сила корректировки (%)",
                min_value=0,
                max_value=100,
                value=50,
                step=10
            ) / 100

    # Загрузка исторических данных
    st.subheader("Параметры анализа")

    col1, col2 = st.columns(2)

    with col1:
        lookback_period = st.selectbox(
            "Период анализа",
            ["1M", "3M", "6M", "1Y", "2Y"],
            index=2,
            format_func=lambda x: {
                "1M": "1 месяц",
                "3M": "3 месяца",
                "6M": "6 месяцев",
                "1Y": "1 год",
                "2Y": "2 года"
            }.get(x, x)
        )

    with col2:
        benchmark = st.selectbox(
            "Бенчмарк",
            ["SPY", "QQQ", "IWM", "VTI", "None"],
            index=0,
            format_func=lambda x: "Нет" if x == "None" else x
        )
        if benchmark == "None":
            benchmark = None

    # Кнопка для запуска тактического распределения
    if st.button("Рассчитать тактическое распределение"):
        with st.spinner("Выполняется тактическое распределение активов..."):
            # Получаем тикеры активов
            tickers = [asset['ticker'] for asset in portfolio_data['assets']]

            # Определяем временной период
            end_date = datetime.now().strftime('%Y-%m-%d')

            if lookback_period == "1M":
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            elif lookback_period == "3M":
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            elif lookback_period == "6M":
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            elif lookback_period == "1Y":
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            else:  # 2Y
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

            # Загружаем исторические цены
            if benchmark:
                tickers_with_benchmark = tickers + [benchmark] if benchmark not in tickers else tickers
                prices_data = data_fetcher.get_batch_data(tickers_with_benchmark, start_date, end_date)
            else:
                prices_data = data_fetcher.get_batch_data(tickers, start_date, end_date)

            # Проверяем наличие данных
            if not prices_data or all(df.empty for df in prices_data.values()):
                st.error("Не удалось загрузить исторические данные. Пожалуйста, проверьте тикеры или измените период.")
                return

            # Обработка данных цен
            close_prices = pd.DataFrame()
            for ticker, df in prices_data.items():
                if not df.empty:
                    if 'Adj Close' in df.columns:
                        close_prices[ticker] = df['Adj Close']
                    elif 'Close' in df.columns:
                        close_prices[ticker] = df['Close']

            # Рассчитываем доходности
            returns = close_prices.pct_change().dropna()

            # Получаем текущие веса
            current_weights = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

            # Применяем выбранный метод тактического распределения
            if tactic_method == "market_momentum":
                # Расчет моментума (средняя доходность за последние 3 месяца)
                momentum_window = min(len(returns), 60)  # Примерно 3 месяца торговых дней
                momentum_scores = returns.iloc[-momentum_window:].mean() * 100  # среднедневная доходность в %

                # Нормализуем моментум-скоры
                momentum_norm = (momentum_scores - momentum_scores.min()) / (
                            momentum_scores.max() - momentum_scores.min() + 1e-10)

                # Корректируем веса на основе моментума
                new_weights = {}
                for ticker in current_weights:
                    if ticker in momentum_norm:
                        # Применяем корректировку с учетом силы
                        adjust_factor = 1 + (momentum_norm[ticker] - 0.5) * adjustment_strength
                        new_weights[ticker] = current_weights[ticker] * adjust_factor

                # Нормализуем веса, чтобы они суммировались к 1
                total_weight = sum(new_weights.values())
                for ticker in new_weights:
                    new_weights[ticker] /= total_weight

            elif tactic_method == "sector_rotation":
                # Получаем информацию о секторах для каждого актива
                sector_data = {}
                for asset in portfolio_data['assets']:
                    ticker = asset['ticker']
                    if 'sector' in asset and asset['sector'] != 'N/A':
                        sector = asset['sector']
                    else:
                        # Если сектор не указан, пытаемся получить его
                        info = data_fetcher.get_company_info(ticker)
                        sector = info.get('sector', 'Unknown')

                    if sector not in sector_data:
                        sector_data[sector] = []
                    sector_data[sector].append(ticker)

                # Получаем производительность секторов
                sector_performance = data_fetcher.get_sector_performance()

                # Определяем веса секторов на основе их производительности
                sector_weights = {}
                if not sector_performance.empty and '3M' in sector_performance.columns:
                    # Нормализуем доходность секторов
                    perf_3m = sector_performance.set_index('Sector')['3M']
                    perf_norm = (perf_3m - perf_3m.min()) / (perf_3m.max() - perf_3m.min() + 1e-10)

                    # Рассчитываем новые веса секторов
                    for sector in sector_data:
                        if sector in perf_norm.index:
                            # Больший вес для секторов с лучшей производительностью
                            sector_weights[sector] = 0.5 + perf_norm[sector] * adjustment_strength
                        else:
                            sector_weights[sector] = 0.5  # нейтральный вес для неизвестных секторов
                else:
                    # Если нет данных о производительности, используем равные веса
                    for sector in sector_data:
                        sector_weights[sector] = 1.0

                # Нормализуем веса секторов
                total_sector_weight = sum(sector_weights.values())
                for sector in sector_weights:
                    sector_weights[sector] /= total_sector_weight

                # Распределяем веса активов внутри секторов
                new_weights = {}
                for sector, tickers in sector_data.items():
                    sector_weight = sector_weights.get(sector, 0)
                    ticker_weight = sector_weight / len(tickers)
                    for ticker in tickers:
                        new_weights[ticker] = ticker_weight

            elif tactic_method == "volatility_targeting":
                # Расчет волатильности каждого актива
                volatility_window = min(len(returns), 60)  # Примерно 3 месяца торговых дней
                volatilities = returns.iloc[-volatility_window:].std() * np.sqrt(252)  # Годовая волатильность

                # Инвертируем волатильность (меньшая волатильность = больший вес)
                inv_vol = 1 / (volatilities + 1e-10)

                # Рассчитываем базовые веса обратно пропорциональные волатильности
                vol_weights = inv_vol / inv_vol.sum()

                # Комбинируем текущие веса и веса на основе волатильности
                new_weights = {}
                for ticker in current_weights:
                    if ticker in vol_weights:
                        # Применяем корректировку с учетом силы
                        new_weights[ticker] = (1 - adjustment_strength) * current_weights[
                            ticker] + adjustment_strength * vol_weights[ticker]

                # Нормализуем веса
                total_weight = sum(new_weights.values())
                for ticker in new_weights:
                    new_weights[ticker] /= total_weight

            else:  # equal_weight
                # Простое равное распределение
                equal_weight = 1.0 / len(current_weights)
                new_weights = {ticker: equal_weight for ticker in current_weights}

            # Отображение результатов
            st.subheader("Результаты тактического распределения")

            # Создаем таблицу с текущими и новыми весами
            weights_df = pd.DataFrame({
                'Актив': list(current_weights.keys()),
                'Текущий вес (%)': [current_weights[t] * 100 for t in current_weights],
                'Новый вес (%)': [new_weights.get(t, 0) * 100 for t in current_weights],
                'Изменение (%)': [(new_weights.get(t, 0) - current_weights[t]) * 100 for t in current_weights]
            })

            # Сортируем по абсолютному изменению
            weights_df['Абс. изменение'] = weights_df['Изменение (%)'].abs()
            weights_df = weights_df.sort_values('Абс. изменение', ascending=False)
            weights_df = weights_df.drop('Абс. изменение', axis=1)

            st.dataframe(weights_df, use_container_width=True)

            # Визуализация изменения весов
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=weights_df['Актив'],
                y=weights_df['Текущий вес (%)'],
                name='Текущий вес (%)',
                marker_color='lightgrey'
            ))

            fig.add_trace(go.Bar(
                x=weights_df['Актив'],
                y=weights_df['Новый вес (%)'],
                name='Новый вес (%)',
                marker_color='royalblue'
            ))

            fig.update_layout(
                title='Сравнение текущих и новых весов',
                barmode='group',
                xaxis_title='Актив',
                yaxis_title='Вес (%)',
                legend_title='',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Кнопка для применения нового распределения
            if st.button("Применить новое распределение к портфелю"):
                # Обновляем веса в портфеле
                for asset in portfolio_data['assets']:
                    ticker = asset['ticker']
                    if ticker in new_weights:
                        asset['weight'] = new_weights[ticker]

                # Сохраняем обновленный портфель
                portfolio_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                portfolio_manager.save_portfolio(portfolio_data)

                st.success(f"Портфель '{selected_portfolio}' успешно обновлен с новым тактическим распределением!")

    # Отображение текущего распределения
    st.subheader("Текущее распределение активов")
    weights_df = pd.DataFrame({
        'Актив': [asset['ticker'] for asset in portfolio_data['assets']],
        'Вес (%)': [asset['weight'] * 100 for asset in portfolio_data['assets']]
    }).sort_values('Вес (%)', ascending=False)

    st.dataframe(weights_df, use_container_width=True)

    # Визуализация текущего распределения
    fig = px.pie(
        weights_df,
        values='Вес (%)',
        names='Актив',
        title='Текущее распределение активов'
    )
    st.plotly_chart(fig, use_container_width=True)


def monte_carlo_simulation(data_fetcher, portfolio_manager):
    """
    Функция для Монте-Карло симуляции портфеля

    Args:
        data_fetcher: Экземпляр DataFetcher для загрузки данных
        portfolio_manager: Экземпляр PortfolioDataManager для работы с портфелями
    """
    st.header("Монте-Карло симуляция")

    # Получаем список портфелей
    portfolios = portfolio_manager.list_portfolios()

    if not portfolios:
        st.info("Портфели не найдены. Создайте портфель в разделе 'Создание портфеля'.")
        return

    # Выбор портфеля для симуляции
    portfolio_names = [p['name'] for p in portfolios]
    selected_portfolio = st.selectbox("Выберите портфель для симуляции", portfolio_names)

    if not selected_portfolio:
        return

    # Загружаем данные портфеля
    portfolio_data = portfolio_manager.load_portfolio(selected_portfolio)

    # Отображаем основную информацию о портфеле
    st.subheader(f"Портфель: {portfolio_data['name']}")

    # Параметры симуляции
    col1, col2, col3 = st.columns(3)

    with col1:
        years = st.slider(
            "Горизонт прогноза (лет)",
            min_value=1,
            max_value=30,
            value=10,
            step=1
        )

    with col2:
        simulations = st.slider(
            "Количество симуляций",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )

    with col3:
        initial_investment = st.number_input(
            "Начальная инвестиция ($)",
            min_value=1000,
            max_value=10000000,
            value=10000,
            step=1000
        )

    # Дополнительные параметры
    col1, col2 = st.columns(2)

    with col1:
        annual_contribution = st.number_input(
            "Ежегодный взнос ($)",
            min_value=0,
            max_value=1000000,
            value=0,
            step=1000
        )

    with col2:
        rebalance_frequency = st.selectbox(
            "Частота ребалансировки",
            ["Нет", "Ежегодно", "Ежеквартально", "Ежемесячно"],
            index=1
        )

    # Расширенные параметры
    with st.expander("Расширенные параметры"):
        col1, col2 = st.columns(2)

        with col1:
            confidence_level = st.slider(
                "Уровень доверия для расчета VaR",
                min_value=80,
                max_value=99,
                value=95,
                step=1,
                format="%d%%"
            ) / 100

        with col2:
            return_method = st.selectbox(
                "Метод расчета доходности",
                ["Исторический", "Параметрический"],
                index=0
            )

        st.info(
            "Параметрический метод предполагает нормальное распределение доходностей, исторический использует эмпирическое распределение.")

    # Кнопка для запуска симуляции
    if st.button("Запустить Монте-Карло симуляцию"):
        with st.spinner('Выполнение симуляции... Это может занять некоторое время.'):
            # Получаем тикеры и веса
            tickers = [asset['ticker'] for asset in portfolio_data['assets']]
            weights = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

            # Загружаем исторические данные
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')  # 5 лет истории

            prices_data = data_fetcher.get_batch_data(tickers, start_date, end_date)

            # Проверяем наличие данных
            if not prices_data or all(df.empty for df in prices_data.values()):
                st.error("Не удалось загрузить исторические данные. Пожалуйста, проверьте тикеры или измените период.")
                return

            # Создаем DataFrame с ценами закрытия
            close_prices = pd.DataFrame()
            for ticker, df in prices_data.items():
                if not df.empty:
                    if 'Adj Close' in df.columns:
                        close_prices[ticker] = df['Adj Close']
                    elif 'Close' in df.columns:
                        close_prices[ticker] = df['Close']

            # Рассчитываем доходности
            returns = close_prices.pct_change().dropna()

            # Рассчитываем доходность портфеля
            portfolio_returns = PortfolioAnalytics.calculate_portfolio_return(returns, weights)

            # Выполняем Монте-Карло симуляцию
            mc_results = RiskManagement.perform_monte_carlo_simulation(
                portfolio_returns,
                initial_value=initial_investment,
                years=years,
                simulations=simulations,
                annual_contribution=annual_contribution
            )

            # Извлекаем результаты симуляции
            simulation_data = mc_results['simulation_data']
            percentiles = mc_results['percentiles']

            # Рассчитываем ключевые метрики
            median_value = percentiles['median']
            p10_value = percentiles['p10']
            p90_value = percentiles['p90']

            # Отображение результатов
            st.subheader("Результаты симуляции")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Медианная стоимость",
                    f"${median_value:,.2f}",
                    f"{(median_value / initial_investment - 1) * 100:.1f}%"
                )

            with col2:
                st.metric(
                    f"Оптимистичный сценарий ({int(90)}%)",
                    f"${p90_value:,.2f}",
                    f"{(p90_value / initial_investment - 1) * 100:.1f}%"
                )

            with col3:
                st.metric(
                    f"Пессимистичный сценарий ({int(10)}%)",
                    f"${p10_value:,.2f}",
                    f"{(p10_value / initial_investment - 1) * 100:.1f}%"
                )

            # Рассчитываем дополнительные метрики
            probability_double = mc_results['probabilities']['prob_reaching_double']
            var = percentiles['p10'] - initial_investment
            var_percent = -var / initial_investment if var < 0 else 0

            st.write(f"**Вероятность удвоения инвестиций:** {probability_double * 100:.1f}%")
            st.write(
                f"**Value at Risk (VaR) при {confidence_level * 100:.0f}% уровне доверия:** ${abs(var):,.2f} ({var_percent * 100:.1f}%)")

            # Визуализация результатов
            st.subheader("Визуализация симуляции")

            # Создаем DataFrame для графика
            years_arr = np.linspace(0, years, simulation_data.shape[1])

            # Рассчитываем персентили
            p10_line = np.percentile(simulation_data, 10, axis=0)
            p25_line = np.percentile(simulation_data, 25, axis=0)
            p50_line = np.percentile(simulation_data, 50, axis=0)
            p75_line = np.percentile(simulation_data, 75, axis=0)
            p90_line = np.percentile(simulation_data, 90, axis=0)

            # Создаем график
            fig = go.Figure()

            # Добавляем области для разных персентилей
            fig.add_trace(go.Scatter(
                x=years_arr,
                y=p90_line,
                mode='lines',
                name='90-й персентиль',
                line=dict(color='lightgreen', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=years_arr,
                y=p75_line,
                mode='lines',
                name='75-й персентиль',
                line=dict(color='rgba(0, 176, 246, 0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(144, 238, 144, 0.3)'
            ))

            fig.add_trace(go.Scatter(
                x=years_arr,
                y=p50_line,
                mode='lines',
                name='Медиана',
                line=dict(color='blue', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=years_arr,
                y=p25_line,
                mode='lines',
                name='25-й персентиль',
                line=dict(color='rgba(255, 165, 0, 0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(173, 216, 230, 0.3)'
            ))

            fig.add_trace(go.Scatter(
                x=years_arr,
                y=p10_line,
                mode='lines',
                name='10-й персентиль',
                line=dict(color='salmon', width=2),
                fill='tonexty',
                fillcolor='rgba(255, 165, 0, 0.3)'
            ))

            # Добавляем начальную инвестицию
            fig.add_trace(go.Scatter(
                x=[0, years],
                y=[initial_investment, initial_investment],
                mode='lines',
                name='Начальная инвестиция',
                line=dict(color='red', width=2, dash='dash')
            ))

            # Добавляем траектории нескольких случайных симуляций
            np.random.seed(42)  # Для воспроизводимости
            sample_indices = np.random.choice(simulation_data.shape[0], min(20, simulation_data.shape[0]),
                                              replace=False)

            for idx in sample_indices:
                fig.add_trace(go.Scatter(
                    x=years_arr,
                    y=simulation_data[idx, :],
                    mode='lines',
                    name=f'Симуляция {idx}',
                    line=dict(color='rgba(128, 128, 128, 0.2)', width=1),
                    showlegend=False
                ))

            fig.update_layout(
                title='Прогноз стоимости портфеля',
                xaxis_title='Годы',
                yaxis_title='Стоимость портфеля ($)',
                legend_title='Персентили',
                hovermode='x',
                yaxis=dict(type='log' if st.checkbox("Логарифмическая шкала", value=False) else 'linear')
            )

            st.plotly_chart(fig, use_container_width=True)

            # Гистограмма финальных значений
            st.subheader("Распределение финальной стоимости портфеля")

            final_values = simulation_data[:, -1]

            fig_hist = go.Figure()

            fig_hist.add_trace(go.Histogram(
                x=final_values,
                nbinsx=50,
                marker_color='rgba(0, 0, 255, 0.5)'
            ))

            # Добавляем вертикальные линии для персентилей
            fig_hist.add_vline(x=p10_line[-1], line_dash="dash", line_color="red", annotation_text="10%")
            fig_hist.add_vline(x=p50_line[-1], line_dash="dash", line_color="green", annotation_text="50%")
            fig_hist.add_vline(x=p90_line[-1], line_dash="dash", line_color="blue", annotation_text="90%")

            # Добавляем начальную инвестицию
            fig_hist.add_vline(x=initial_investment, line_dash="solid", line_color="black",
                               annotation_text="Начальная инвестиция")

            fig_hist.update_layout(
                title='Распределение финальной стоимости портфеля через {0} лет'.format(years),
                xaxis_title='Стоимость портфеля ($)',
                yaxis_title='Частота',
                showlegend=False
            )

            st.plotly_chart(fig_hist, use_container_width=True)

            # Таблица с детальными результатами
            st.subheader("Детальные результаты по годам")

            # Равномерно выбираем значения из массивов персентилей для каждого года
            years_range = range(years + 1)
            indices = np.linspace(0, len(p10_line) - 1, years + 1, dtype=int)

            yearly_results = pd.DataFrame({
                'Год': years_range,
                'Начальная инвестиция': [initial_investment] * (years + 1),
                '10й персентиль': p10_line[indices],
                '25й персентиль': p25_line[indices],
                'Медиана': p50_line[indices],
                '75й персентиль': p75_line[indices],
                '90й персентиль': p90_line[indices]
            })

            # Форматируем данные для отображения
            for col in yearly_results.columns:
                if col != 'Год':
                    yearly_results[col] = yearly_results[col].map('${:,.2f}'.format)

            st.dataframe(yearly_results, use_container_width=True)