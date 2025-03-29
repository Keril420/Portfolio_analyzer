# src/pages/portfolio_analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Добавляем корень проекта в путь Python
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Используем абсолютные импорты
from src.utils.calculations import PortfolioAnalytics
from src.utils.risk_management import RiskManagement
from src.utils.visualization import PortfolioVisualization
import src.config as config

def run(data_fetcher, portfolio_manager):
    """
    Функция для отображения страницы анализа портфеля

    Args:
        data_fetcher: Экземпляр DataFetcher для загрузки данных
        portfolio_manager: Экземпляр PortfolioDataManager для работы с портфелями
    """
    st.title("Анализ портфеля")

    # Получаем список портфелей
    portfolios = portfolio_manager.list_portfolios()

    if not portfolios:
        st.info("Портфели не найдены. Создайте портфель в разделе 'Создание портфеля'.")
        return

    # Выбор портфеля для анализа
    portfolio_names = [p['name'] for p in portfolios]
    selected_portfolio = st.selectbox("Выберите портфель для анализа", portfolio_names)

    if not selected_portfolio:
        return

    # Загружаем данные портфеля
    portfolio_data = portfolio_manager.load_portfolio(selected_portfolio)

    # Отображаем основную информацию о портфеле
    st.header(f"Портфель: {portfolio_data['name']}")

    if 'description' in portfolio_data and portfolio_data['description']:
        st.write(portfolio_data['description'])

    st.write(f"Последнее обновление: {portfolio_data.get('last_updated', 'Неизвестно')}")

    # Параметры анализа
    col1, col2, col3 = st.columns(3)
    with col1:
        benchmark = st.selectbox(
            "Бенчмарк",
            ["SPY", "QQQ", "DIA", "IWM", "VTI"],
            index=0
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

    # Обновление цен портфеля
    update_col, refresh_col = st.columns([3, 1])
    with update_col:
        if st.button("Обновить цены активов"):
            portfolio_data = portfolio_manager.update_portfolio_prices(portfolio_data)
            portfolio_manager.save_portfolio(portfolio_data)
            st.success("Цены активов обновлены!")

    with refresh_col:
        if st.button("Загрузить исторические данные", help="Загрузить исторические данные для выбранного периода"):
            st.info("Загрузка исторических данных...")

    # Вкладки для различных типов анализа
    tabs = st.tabs([
        "Обзор портфеля",
        "Доходность",
        "Риск",
        "Активы",
        "Корреляции",
        "Стресс-тестирование"
    ])

    # Загружаем исторические данные для всех активов
    tickers = [asset['ticker'] for asset in portfolio_data['assets']]

    # Добавляем бенчмарк к списку тикеров для загрузки
    if benchmark not in tickers:
        tickers.append(benchmark)

    # Конвертируем даты в строки
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    with st.spinner('Загрузка исторических данных...'):
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

        # Рассчитываем веса из портфеля
        weights = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

        # Рассчитываем доходность портфеля
        portfolio_returns = PortfolioAnalytics.calculate_portfolio_return(returns, weights)

        # Рассчитываем бенчмарк доходность
        if benchmark in returns.columns:
            benchmark_returns = returns[benchmark]
        else:
            benchmark_returns = None

    # Вкладка "Обзор портфеля"
    with tabs[0]:
        st.subheader("Обзор портфеля")

        # Рассчитываем метрики портфеля
        with st.spinner('Расчет метрик портфеля...'):
            portfolio_metrics = PortfolioAnalytics.calculate_portfolio_metrics(
                portfolio_returns,
                benchmark_returns,
                risk_free_rate=config.RISK_FREE_RATE
            )

        # Отображаем основные метрики портфеля
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Общая доходность",
                f"{portfolio_metrics.get('total_return', 0) * 100:.2f}%",
                f"{(portfolio_metrics.get('total_return', 0) - portfolio_metrics.get('benchmark_return', 0)) * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Годовая доходность",
                f"{portfolio_metrics.get('annualized_return', 0) * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Волатильность",
                f"{portfolio_metrics.get('volatility', 0) * 100:.2f}%"
            )

        with col4:
            st.metric(
                "Коэффициент Шарпа",
                f"{portfolio_metrics.get('sharpe_ratio', 0):.2f}"
            )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Макс. просадка",
                f"{portfolio_metrics.get('max_drawdown', 0) * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Коэффициент Сортино",
                f"{portfolio_metrics.get('sortino_ratio', 0):.2f}"
            )

        with col3:
            st.metric(
                "Бета",
                f"{portfolio_metrics.get('beta', 0):.2f}"
            )

        with col4:
            st.metric(
                "Альфа",
                f"{portfolio_metrics.get('alpha', 0) * 100:.2f}%",
                f"{portfolio_metrics.get('alpha', 0) * 100:.2f}%"
            )

        # Отображаем структуру портфеля
        st.subheader("Структура портфеля")

        # Распределение активов
        fig_weights = px.pie(
            values=[asset['weight'] for asset in portfolio_data['assets']],
            names=[asset['ticker'] for asset in portfolio_data['assets']],
            title="Распределение активов"
        )
        st.plotly_chart(fig_weights, use_container_width=True)

        # Отображаем распределение по секторам, если доступно
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

    # Вкладка "Доходность"
    with tabs[1]:
        st.subheader("Анализ доходности")

        # Кумулятивная доходность
        cumulative_portfolio_returns = (1 + portfolio_returns).cumprod() - 1

        if benchmark_returns is not None:
            cumulative_benchmark_returns = (1 + benchmark_returns).cumprod() - 1

            # Создаем график с помощью plotly
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=cumulative_portfolio_returns.index,
                y=cumulative_portfolio_returns.values * 100,
                mode='lines',
                name='Портфель'
            ))

            fig.add_trace(go.Scatter(
                x=cumulative_benchmark_returns.index,
                y=cumulative_benchmark_returns.values * 100,
                mode='lines',
                name=benchmark
            ))

            fig.update_layout(
                title='Кумулятивная доходность',
                xaxis_title='Дата',
                yaxis_title='Доходность (%)',
                legend_title='',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            # Если бенчмарк не доступен, показываем только портфель
            fig = px.line(
                y=cumulative_portfolio_returns.values * 100,
                x=cumulative_portfolio_returns.index,
                labels={'x': 'Дата', 'y': 'Доходность (%)'},
                title='Кумулятивная доходность'
            )
            st.plotly_chart(fig, use_container_width=True)
        # Проверяем тип индекса перед ресемплингом
        if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
            st.warning("Не удалось провести месячный ресемплинг: индекс не является DatetimeIndex")
            monthly_returns = pd.Series()  # Пустая серия в случае ошибки
        else:
            monthly_returns = portfolio_returns.resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )

        fig_monthly = px.bar(
            y=monthly_returns.values * 100,
            x=monthly_returns.index.strftime('%Y-%m'),
            labels={'y': 'Доходность (%)', 'x': 'Месяц'},
            title='Месячная доходность',
            color=monthly_returns.values,
            color_continuous_scale=['red', 'green'],
            range_color=[-max(abs(monthly_returns.values)) * 100, max(abs(monthly_returns.values)) * 100]
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Распределение дневной доходности
        fig_dist = px.histogram(
            portfolio_returns * 100,
            title='Распределение дневной доходности',
            labels={'value': 'Дневная доходность (%)', 'count': 'Частота'},
            marginal='box',
            nbins=50
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Вкладка "Риск"
    with tabs[2]:
        st.subheader("Анализ риска")

        # Просадки
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns / peak - 1) * 100

        fig_drawdown = px.area(
            drawdown,
            title='Просадки портфеля',
            labels={'value': 'Просадка (%)', 'index': 'Дата'},
            color_discrete_sequence=['red']
        )
        st.plotly_chart(fig_drawdown, use_container_width=True)

        # Value at Risk (VaR) и Conditional VaR (CVaR)
        col1, col2 = st.columns(2)

        with col1:
            confidence_level = st.slider(
                "Уровень доверия для VaR/CVaR",
                min_value=90,
                max_value=99,
                value=95,
                step=1,
                format="%d%%"
            ) / 100

            var_hist = RiskManagement.calculate_var_historical(portfolio_returns, confidence_level=confidence_level)
            var_param = RiskManagement.calculate_var_parametric(portfolio_returns, confidence_level=confidence_level)
            var_mc = RiskManagement.calculate_var_monte_carlo(portfolio_returns, confidence_level=confidence_level)
            cvar = RiskManagement.calculate_cvar(portfolio_returns, confidence_level=confidence_level)

            st.write(f"**Value at Risk (VaR) при {confidence_level * 100:.0f}% уровне доверия:**")
            st.write(f"Исторический метод: {var_hist * 100:.2f}%")
            st.write(f"Параметрический метод: {var_param * 100:.2f}%")
            st.write(f"Метод Монте-Карло: {var_mc * 100:.2f}%")
            st.write(f"Conditional VaR (CVaR): {cvar * 100:.2f}%")

        with col2:
            # Визуализация VaR
            var_df = pd.DataFrame({
                'Метод': ['Исторический', 'Параметрический', 'Монте-Карло', 'CVaR'],
                'Значение (%)': [var_hist * 100, var_param * 100, var_mc * 100, cvar * 100]
            })

            fig_var = px.bar(
                var_df,
                x='Метод',
                y='Значение (%)',
                title=f'Value at Risk при {confidence_level * 100:.0f}% уровне доверия',
                color='Метод',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#1A535C', '#FF9F1C']
            )
            st.plotly_chart(fig_var, use_container_width=True)

        # Вклад в риск
        risk_contribution = RiskManagement.calculate_risk_contribution(returns, weights)

        if risk_contribution:
            risk_contrib_df = pd.DataFrame({
                'Актив': list(risk_contribution.keys()),
                'Вклад в риск (%)': [v * 100 for v in risk_contribution.values()]
            })

            fig_risk_contrib = px.bar(
                risk_contrib_df,
                x='Актив',
                y='Вклад в риск (%)',
                title='Вклад активов в общий риск портфеля',
                color='Вклад в риск (%)',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_risk_contrib, use_container_width=True)

    # Вкладка "Активы"
    with tabs[3]:
        st.subheader("Анализ активов")

        # Отображение списка активов
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

        assets_df = pd.DataFrame(assets_data)
        st.dataframe(assets_df, use_container_width=True)

        # График изменения цен активов
        st.subheader("Изменение цен активов")

        # Нормализуем цены для сравнения
        normalized_prices = close_prices.copy()

        for ticker in tickers:
            if ticker in normalized_prices.columns:
                normalized_prices[ticker] = normalized_prices[ticker] / normalized_prices[ticker].iloc[0]

        # Строим график
        fig_prices = px.line(
            normalized_prices,
            title='Нормализованные цены (от начальной даты)',
            labels={'value': 'Нормализованная цена', 'index': 'Дата'}
        )
        st.plotly_chart(fig_prices, use_container_width=True)

        # Анализ отдельных активов
        st.subheader("Анализ отдельного актива")
        selected_asset = st.selectbox("Выберите актив для детального анализа", tickers)

        if selected_asset in close_prices.columns:
            asset_price = close_prices[selected_asset]
            asset_return = returns[selected_asset]

            # Статистика актива
            st.write("### Статистика актива")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Общая доходность",
                    f"{((asset_price.iloc[-1] / asset_price.iloc[0]) - 1) * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Волатильность",
                    f"{asset_return.std() * np.sqrt(252) * 100:.2f}%"
                )

            with col3:
                st.metric(
                    "Максимальная просадка",
                    f"{PortfolioAnalytics.calculate_max_drawdown(asset_return) * 100:.2f}%"
                )

            # График цены и объема
            if selected_asset in prices_data and 'Volume' in prices_data[selected_asset].columns:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=asset_price.index,
                    y=asset_price.values,
                    mode='lines',
                    name='Цена'
                ))

                fig.add_trace(go.Bar(
                    x=prices_data[selected_asset].index,
                    y=prices_data[selected_asset]['Volume'],
                    name='Объем',
                    yaxis='y2',
                    opacity=0.3
                ))

                fig.update_layout(
                    title=f'{selected_asset} - Цена и объем',
                    yaxis=dict(title='Цена'),
                    yaxis2=dict(
                        title='Объем',
                        overlaying='y',
                        side='right'
                    ),
                    xaxis=dict(title='Дата'),
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                # Если данные об объеме недоступны, просто показываем цену
                fig = px.line(
                    asset_price,
                    title=f'{selected_asset} - Цена',
                    labels={'value': 'Цена', 'index': 'Дата'}
                )
                st.plotly_chart(fig, use_container_width=True)

    # Вкладка "Корреляции"
    with tabs[4]:
        st.subheader("Анализ корреляций")

        # Корреляционная матрица
        correlation_matrix = returns.corr()

        fig_corr = px.imshow(
            correlation_matrix,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title='Корреляционная матрица'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Корреляция с бенчмарком
        if benchmark in returns.columns:
            correlations_with_benchmark = returns.corr()[benchmark].drop(benchmark)

            fig_bench_corr = px.bar(
                x=correlations_with_benchmark.index,
                y=correlations_with_benchmark.values,
                title=f'Корреляция активов с {benchmark}',
                labels={'x': 'Актив', 'y': 'Корреляция'},
                color=correlations_with_benchmark.values,
                color_continuous_scale='RdBu_r',
                range_color=[-1, 1]
            )
            st.plotly_chart(fig_bench_corr, use_container_width=True)

    # Вкладка "Стресс-тестирование"
    with tabs[5]:
        st.subheader("Стресс-тестирование")

        # Выбор сценария стресс-тестирования
        scenarios = [
            "financial_crisis_2008",
            "covid_2020",
            "tech_bubble_2000",
            "black_monday_1987",
            "inflation_shock",
            "rate_hike",
            "moderate_recession",
            "severe_recession"
        ]

        scenario_names = {
            "financial_crisis_2008": "Финансовый кризис 2008",
            "covid_2020": "Пандемия COVID-19 (2020)",
            "tech_bubble_2000": "Крах доткомов (2000)",
            "black_monday_1987": "Черный понедельник (1987)",
            "inflation_shock": "Инфляционный шок",
            "rate_hike": "Резкое повышение ставок",
            "moderate_recession": "Умеренная рецессия",
            "severe_recession": "Тяжелая рецессия"
        }

        # Выбор сценария и портфельной стоимости
        col1, col2 = st.columns(2)

        with col1:
            selected_scenario = st.selectbox(
                "Выберите сценарий стресс-тестирования",
                options=scenarios,
                format_func=lambda x: scenario_names.get(x, x)
            )

        with col2:
            portfolio_value = st.number_input(
                "Стоимость портфеля ($)",
                min_value=1000,
                value=10000,
                step=1000
            )

        # Проводим стресс-тестирование
        stress_test_result = RiskManagement.perform_stress_test(
            portfolio_returns, selected_scenario, portfolio_value
        )

        if 'error' not in stress_test_result:
            st.subheader(f"Результаты стресс-теста: {scenario_names.get(selected_scenario, selected_scenario)}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Изменение стоимости",
                    f"${stress_test_result['portfolio_loss']:.2f}",
                    f"{stress_test_result['shock_percentage'] * 100:.1f}%",
                    delta_color="inverse"
                )

            with col2:
                st.metric(
                    "Стоимость после шока",
                    f"${stress_test_result['portfolio_after_shock']:.2f}"
                )

            with col3:
                st.metric(
                    "Ожидаемое время восстановления",
                    f"{stress_test_result['recovery_months']:.1f} мес."
                )

            # Визуализация результатов стресс-теста
            st.write("### Визуализация стресс-теста")

            fig_stress = go.Figure()

            # Создаем временную шкалу (в месяцах)
            months = list(range(-1, int(stress_test_result['recovery_months']) + 2))
            values = []

            # Добавляем начальное значение
            values.append(portfolio_value)

            # Добавляем значение после шока
            values.append(stress_test_result['portfolio_after_shock'])

            # Добавляем значения восстановления (линейное приближение)
            recovery_rate = (portfolio_value - stress_test_result['portfolio_after_shock']) / stress_test_result[
                'recovery_months']

            for i in range(1, len(months) - 1):
                values.append(stress_test_result['portfolio_after_shock'] + recovery_rate * i)

            fig_stress.add_trace(go.Scatter(
                x=months,
                y=values,
                mode='lines+markers',
                name='Стоимость портфеля'
            ))

            fig_stress.add_shape(
                type="line",
                x0=months[0],
                y0=portfolio_value,
                x1=months[-1],
                y1=portfolio_value,
                line=dict(color="green", width=2, dash="dot"),
                name="Исходная стоимость"
            )

            fig_stress.update_layout(
                title=f"Стресс-тест: {scenario_names.get(selected_scenario, selected_scenario)}",
                xaxis_title="Месяцы",
                yaxis_title="Стоимость портфеля ($)",
                hovermode="x"
            )

            st.plotly_chart(fig_stress, use_container_width=True)

            # Пользовательский стресс-тест
            st.subheader("Пользовательский стресс-тест")

            st.write("""
            Создайте собственный сценарий стресс-тестирования, указав шоковые изменения 
            для отдельных активов или классов активов.
            """)

            # Создаем пользовательские шоки для каждого актива
            custom_shocks = {}

            for asset in portfolio_data['assets']:
                ticker = asset['ticker']
                default_shock = -0.2  # По умолчанию -20%
                custom_shocks[ticker] = st.slider(
                    f"Шок для {ticker} (%)",
                    min_value=-90,
                    max_value=50,
                    value=int(default_shock * 100),
                    step=5
                ) / 100

            # Кнопка для запуска пользовательского стресс-теста
            if st.button("Запустить пользовательский стресс-тест"):
                # Проводим пользовательский стресс-тест
                custom_test_result = RiskManagement.perform_custom_stress_test(
                    returns, weights, custom_shocks, portfolio_value
                )

                if 'error' not in custom_test_result:
                    st.subheader("Результаты пользовательского стресс-теста")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Изменение стоимости",
                            f"${custom_test_result['portfolio_loss']:.2f}",
                            f"{custom_test_result['loss_percentage'] * 100:.1f}%",
                            delta_color="inverse"
                        )

                    with col2:
                        st.metric(
                            "Стоимость после шока",
                            f"${custom_test_result['portfolio_after_shock']:.2f}"
                        )

                    # Визуализация потерь по активам
                    if 'position_losses' in custom_test_result:
                        pos_loss_df = pd.DataFrame({
                            'Актив': list(custom_test_result['position_losses'].keys()),
                            'Потери ($)': [abs(val) for val in custom_test_result['position_losses'].values()]
                        })

                        # Сортируем по абсолютной величине потерь
                        pos_loss_df = pos_loss_df.sort_values('Потери ($)', ascending=False)

                        fig_pos_loss = px.bar(
                            pos_loss_df,
                            x='Актив',
                            y='Потери ($)',
                            title='Потери по активам',
                            color='Потери ($)',
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig_pos_loss, use_container_width=True)