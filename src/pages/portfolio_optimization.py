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

# Добавляем корневую директорию проекта в путь для импорта
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils.calculations import PortfolioAnalytics
from utils.risk_management import RiskManagement
from utils.visualization import PortfolioVisualization
import config


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