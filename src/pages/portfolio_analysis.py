import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from datetime import datetime, timedelta
import calendar
import logging
logger = logging.getLogger(__name__)
import sys
import os

# Добавляем корень проекта в путь Python
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Используем абсолютные импорты
from src.utils.calculations import PortfolioAnalytics
from src.utils.risk_management import RiskManagement
from src.utils.visualization import PortfolioVisualization
import src.config as config

# Добавляем нашу новую функцию здесь
def load_portfolio_data(data_fetcher, tickers, start_date_str, end_date_str, benchmark=None):
    """
    Загружает исторические данные для списка тикеров и бенчмарка

    Args:
        data_fetcher: Экземпляр DataFetcher
        tickers: Список тикеров
        start_date_str: Начальная дата в формате строки
        end_date_str: Конечная дата в формате строки
        benchmark: Тикер бенчмарка (опционально)

    Returns:
        Tuple с (close_prices, returns, valid_tickers, benchmark_returns, prices_data)
    """
    # Добавляем бенчмарк к списку тикеров для загрузки
    if benchmark and benchmark not in tickers:
        tickers_to_load = tickers + [benchmark]
    else:
        tickers_to_load = tickers

    # Загружаем исторические цены для всех активов
    prices_data = data_fetcher.get_batch_data(tickers_to_load, start_date_str, end_date_str)

    # Проверяем, что данные загружены успешно
    if not prices_data or all(df.empty for df in prices_data.values()):
        raise ValueError("Не удалось загрузить исторические данные. Пожалуйста, проверьте тикеры или измените период.")

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

    # Фильтруем тикеры, которые есть в данных
    valid_tickers = [t for t in tickers if t in returns.columns]

    # Если не все тикеры найдены, выдаем предупреждение
    if len(valid_tickers) < len(tickers):
        missing_tickers = set(tickers) - set(valid_tickers)
        logger.warning(f"Не найдены данные для следующих тикеров: {missing_tickers}")

    # Рассчитываем бенчмарк доходность
    benchmark_returns = None
    if benchmark and benchmark in returns.columns:
        benchmark_returns = returns[benchmark]

    return close_prices, returns, valid_tickers, benchmark_returns, prices_data

# Основная функция...
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

    tabs = st.tabs([
        "Обзор портфеля",
        "Доходность",
        "Риск",
        "Активы",
        "Корреляции",
        "Стресс-тестирование",
        "Скользящие метрики",
        "Расширенный анализ"
    ])

    # Получаем тикеры из портфеля
    tickers = [asset['ticker'] for asset in portfolio_data['assets']]

    # Конвертируем даты в строки
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    with st.spinner('Загрузка исторических данных...'):
        try:
            # Используем новую функцию для загрузки данных
            close_prices, returns, valid_tickers, benchmark_returns, prices_data = load_portfolio_data(
                data_fetcher, tickers, start_date_str, end_date_str, benchmark
            )

            # Если не все тикеры найдены, выводим предупреждение
            if len(valid_tickers) < len(tickers):
                missing_tickers = set(tickers) - set(valid_tickers)
                st.warning(f"Не найдены данные для следующих тикеров: {', '.join(missing_tickers)}")

            # Рассчитываем веса из портфеля
            weights = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

            # Рассчитываем доходность портфеля
            portfolio_returns = PortfolioAnalytics.calculate_portfolio_return(returns, weights)

        except ValueError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Произошла ошибка при загрузке данных: {e}")
            return

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

        # Создаем несколько строк метрик
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Общая доходность",
                f"{portfolio_metrics.get('total_return', 0) * 100:.2f}%",
                f"{(portfolio_metrics.get('total_return', 0) - portfolio_metrics.get('benchmark_return', 0)) * 100:.2f}%" if 'benchmark_return' in portfolio_metrics else None
            )

        with col2:
            st.metric(
                "Годовая доходность",
                f"{portfolio_metrics.get('annualized_return', 0) * 100:.2f}%",
                f"{(portfolio_metrics.get('annualized_return', 0) - portfolio_metrics.get('benchmark_annualized_return', 0)) * 100:.2f}%" if 'benchmark_annualized_return' in portfolio_metrics else None
            )

        with col3:
            st.metric(
                "Волатильность",
                f"{portfolio_metrics.get('volatility', 0) * 100:.2f}%",
                f"{(portfolio_metrics.get('benchmark_volatility', 0) - portfolio_metrics.get('volatility', 0)) * 100:.2f}%" if 'benchmark_volatility' in portfolio_metrics else None,
                delta_color="inverse"
            )

        with col4:
            st.metric(
                "Коэффициент Шарпа",
                f"{portfolio_metrics.get('sharpe_ratio', 0):.2f}",
                f"{portfolio_metrics.get('sharpe_ratio', 0) - portfolio_metrics.get('benchmark_sharpe_ratio', 0):.2f}" if 'benchmark_sharpe_ratio' in portfolio_metrics else None
            )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Макс. просадка",
                f"{portfolio_metrics.get('max_drawdown', 0) * 100:.2f}%",
                f"{(portfolio_metrics.get('benchmark_max_drawdown', 0) - portfolio_metrics.get('max_drawdown', 0)) * 100:.2f}%" if 'benchmark_max_drawdown' in portfolio_metrics else None,
                delta_color="inverse"
            )

        with col2:
            st.metric(
                "Коэффициент Сортино",
                f"{portfolio_metrics.get('sortino_ratio', 0):.2f}",
                f"{portfolio_metrics.get('sortino_ratio', 0) - portfolio_metrics.get('benchmark_sortino_ratio', 0):.2f}" if 'benchmark_sortino_ratio' in portfolio_metrics else None
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
                f"{portfolio_metrics.get('alpha', 0) * 100:.2f}%" if portfolio_metrics.get('alpha', 0) != 0 else None
            )

        # Комбинированный график производительности
        st.subheader("Динамика портфеля")

        # Создаем график с подграфиками
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=("Кумулятивная доходность", "Просадки", "Дневная доходность")
        )

        # 1. Кумулятивная доходность (верхний график)
        cumulative_portfolio_returns = (1 + portfolio_returns).cumprod() - 1
        fig.add_trace(
            go.Scatter(
                x=cumulative_portfolio_returns.index,
                y=cumulative_portfolio_returns.values * 100,
                mode='lines',
                line=dict(color='#1f77b4', width=2),
                name='Портфель'
            ),
            row=1, col=1
        )

        # Добавляем бенчмарк, если есть
        if benchmark_returns is not None:
            cumulative_benchmark_returns = (1 + benchmark_returns).cumprod() - 1
            fig.add_trace(
                go.Scatter(
                    x=cumulative_benchmark_returns.index,
                    y=cumulative_benchmark_returns.values * 100,
                    mode='lines',
                    line=dict(color='#ff7f0e', width=1.5, dash='dash'),
                    name=benchmark
                ),
                row=1, col=1
            )

        # 2. Просадки (средний график)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / peak - 1) * 100

        drawdowns = np.clip(drawdowns, -100, 0)  # Ограничение просадок

        fig.add_trace(
            go.Scatter(
                x=drawdowns.index,
                y=drawdowns.values,
                mode='lines',
                line=dict(color='#d62728', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(214, 39, 40, 0.2)',
                name='Просадки'
            ),
            row=2, col=1
        )

        # Добавляем бенчмарк, если есть
        if benchmark_returns is not None:
            cumulative_benchmark = (1 + benchmark_returns).cumprod()
            peak_benchmark = cumulative_benchmark.cummax()
            benchmark_drawdowns = (cumulative_benchmark / peak_benchmark - 1) * 100

            fig.add_trace(
                go.Scatter(
                    x=benchmark_drawdowns.index,
                    y=benchmark_drawdowns.values,
                    mode='lines',
                    line=dict(color='#ff7f0e', dash='dash'),
                    name=benchmark
                ),
                row=2, col=1
            )

        fig.update_yaxes(title_text="Просадка (%)", row=2, col=1)

        # 3. Дневная доходность (нижний график)
        fig.add_trace(
            go.Bar(
                x=portfolio_returns.index,
                y=portfolio_returns.values * 100,
                marker=dict(color='#2ca02c'),
                name='Дневная доходность'
            ),
            row=3, col=1
        )

        # Обновляем настройки графика
        fig.update_layout(
            height=800,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode="x unified"
        )

        # Заголовки осей Y
        fig.update_yaxes(title_text="Доходность (%)", row=1, col=1)
        fig.update_yaxes(title_text="Просадка (%)", row=2, col=1)
        fig.update_yaxes(title_text="Доходность (%)", row=3, col=1)

        # Отображаем график
        st.plotly_chart(fig, use_container_width=True)

        # Отображаем структуру портфеля
        st.subheader("Структура портфеля")

        col1, col2 = st.columns(2)

        with col1:
            # Улучшенный график распределения активов
            fig_weights = px.pie(
                values=[asset['weight'] for asset in portfolio_data['assets']],
                names=[asset['ticker'] for asset in portfolio_data['assets']],
                title="Распределение по активам",
                hole=0.4,  # Пончиковый график
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_weights.update_traces(textposition='inside', textinfo='percent+label')
            fig_weights.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_weights, use_container_width=True)

        with col2:
            # Распределение по секторам, если доступно
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
                    title="Распределение по секторам",
                    hole=0.4,  # Пончиковый график
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_sectors.update_traces(textposition='inside', textinfo='percent+label')
                fig_sectors.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_sectors, use_container_width=True)

    # Вкладка "Доходность"
    with tabs[1]:
        st.subheader("Анализ доходности")

        # Создаем подвкладки для разных видов анализа доходности
        return_tabs = st.tabs([
            "Кумулятивная доходность",
            "Периодический анализ",
            "Распределение доходности",
            "Доходность по периодам"
        ])

        with return_tabs[0]:
            # Улучшенный график кумулятивной доходности с логарифмической опцией
            st.subheader("Кумулятивная доходность с бенчмарком")

            log_scale = st.checkbox("Логарифмическая шкала", value=False)

            cumulative_portfolio_returns = (1 + portfolio_returns).cumprod() - 1

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=cumulative_portfolio_returns.index,
                y=cumulative_portfolio_returns.values * 100,
                mode='lines',
                name='Портфель',
                line=dict(width=2, color='#1f77b4')
            ))

            if benchmark_returns is not None:
                cumulative_benchmark_returns = (1 + benchmark_returns).cumprod() - 1
                fig.add_trace(go.Scatter(
                    x=cumulative_benchmark_returns.index,
                    y=cumulative_benchmark_returns.values * 100,
                    mode='lines',
                    name=benchmark,
                    line=dict(width=1.5, color='#ff7f0e', dash='dash')
                ))

            fig.update_layout(
                title='Кумулятивная доходность',
                xaxis_title='Дата',
                yaxis_title='Доходность (%)',
                yaxis=dict(type='log' if log_scale else 'linear'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        with return_tabs[1]:
            st.subheader("Периодический анализ доходности")

            # Рассчитываем годовую доходность при наличии DatetimeIndex
            if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                annual_returns = portfolio_returns.resample('Y').apply(
                    lambda x: (1 + x).prod() - 1
                ) * 100

                benchmark_annual_returns = None
                if benchmark_returns is not None:
                    benchmark_annual_returns = benchmark_returns.resample('Y').apply(
                        lambda x: (1 + x).prod() - 1
                    ) * 100

                # График годовой доходности
                fig_annual = go.Figure()

                fig_annual.add_trace(go.Bar(
                    x=annual_returns.index.year,
                    y=annual_returns.values,
                    name='Портфель',
                    marker_color='#1f77b4'
                ))

                if benchmark_annual_returns is not None:
                    fig_annual.add_trace(go.Bar(
                        x=benchmark_annual_returns.index.year,
                        y=benchmark_annual_returns.values,
                        name=benchmark,
                        marker_color='#ff7f0e'
                    ))

                fig_annual.update_layout(
                    title='Годовая доходность',
                    xaxis_title='Год',
                    yaxis_title='Доходность (%)',
                    barmode='group',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode='x unified'
                )

                st.plotly_chart(fig_annual, use_container_width=True)

                # Тепловая карта месячной доходности
                monthly_returns = PortfolioVisualization.create_monthly_returns_heatmap(portfolio_returns)

                if not monthly_returns.empty:
                    # Преобразуем в проценты для отображения
                    monthly_returns_pct = monthly_returns * 100

                    fig_heatmap = px.imshow(
                        monthly_returns_pct,
                        labels=dict(x="Месяц", y="Год", color="Доходность (%)"),
                        x=monthly_returns_pct.columns,
                        y=monthly_returns_pct.index,
                        color_continuous_scale='RdYlGn',
                        aspect="auto",
                        text_auto='.1f'
                    )

                    fig_heatmap.update_layout(
                        title='Календарь месячных доходностей (%)',
                        height=400
                    )

                    st.plotly_chart(fig_heatmap, use_container_width=True)

        with return_tabs[2]:
            st.subheader("Распределение доходности")

            # Добавляем прямой импорт здесь
            from scipy import stats as scipy_stats

            col1, col2 = st.columns(2)

            with col1:
                # Улучшенная гистограмма дневной доходности
                fig_daily_dist = px.histogram(
                    portfolio_returns * 100,
                    nbins=40,
                    title="Распределение дневной доходности",
                    labels={'value': 'Доходность (%)', 'count': 'Частота'},
                    histnorm='probability density',
                    marginal='box',
                    color_discrete_sequence=['#1f77b4']
                )

                # Добавляем кривую нормального распределения
                x = np.linspace(min(portfolio_returns * 100), max(portfolio_returns * 100), 100)
                mean = (portfolio_returns * 100).mean()
                std = (portfolio_returns * 100).std()
                y = scipy_stats.norm.pdf(x, mean, std)  # Используем scipy_stats вместо stats

                fig_daily_dist.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='lines',
                        name='Нормальное распределение',
                        line=dict(color='red', dash='dash')
                    )
                )

                st.plotly_chart(fig_daily_dist, use_container_width=True)

            with col2:
                # Гистограмма месячной доходности, если доступна
                if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                    monthly_returns_series = portfolio_returns.resample('M').apply(
                        lambda x: (1 + x).prod() - 1
                    ) * 100

                    fig_monthly_dist = px.histogram(
                        monthly_returns_series,
                        nbins=20,
                        title="Распределение месячной доходности",
                        labels={'value': 'Доходность (%)', 'count': 'Частота'},
                        histnorm='probability density',
                        marginal='box',
                        color_discrete_sequence=['#2ca02c']
                    )

                    # Добавляем кривую нормального распределения для месячной доходности
                    x_monthly = np.linspace(min(monthly_returns_series), max(monthly_returns_series), 100)
                    mean_monthly = monthly_returns_series.mean()
                    std_monthly = monthly_returns_series.std()
                    y_monthly = scipy_stats.norm.pdf(x_monthly, mean_monthly,
                                                     std_monthly)  # Используем scipy_stats вместо stats

                    fig_monthly_dist.add_trace(
                        go.Scatter(
                            x=x_monthly,
                            y=y_monthly,
                            mode='lines',
                            name='Нормальное распределение',
                            line=dict(color='red', dash='dash')
                        )
                    )

                    st.plotly_chart(fig_monthly_dist, use_container_width=True)

        with return_tabs[3]:
            st.subheader("Доходность по периодам")

            # Анализ доходности по различным периодам
            if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                period_returns = PortfolioAnalytics.calculate_period_performance(portfolio_returns)

                if benchmark_returns is not None:
                    benchmark_period_returns = PortfolioAnalytics.calculate_period_performance(benchmark_returns)

                    # Создаем DataFrame для сравнения
                    periods_data = []
                    for period in period_returns:
                        periods_data.append({
                            'Период': period,
                            'Портфель (%)': period_returns[period] * 100,
                            'Бенчмарк (%)': benchmark_period_returns.get(period, 0) * 100,
                            'Разница (%)': (period_returns[period] - benchmark_period_returns.get(period, 0)) * 100
                        })

                    periods_df = pd.DataFrame(periods_data)

                    # Стилизуем DataFrame
                    def highlight_diff(val):
                        if isinstance(val, float):
                            if val > 0:
                                return 'background-color: rgba(75, 192, 192, 0.2); color: green'
                            elif val < 0:
                                return 'background-color: rgba(255, 99, 132, 0.2); color: red'
                        return ''

                    styled_df = periods_df.style.format({
                        'Портфель (%)': '{:.2f}%',
                        'Бенчмарк (%)': '{:.2f}%',
                        'Разница (%)': '{:.2f}%'
                    }).applymap(highlight_diff, subset=['Разница (%)'])

                    st.dataframe(styled_df, use_container_width=True)

                    # Визуализация сравнения
                    fig_periods = px.bar(
                        periods_df,
                        x='Период',
                        y=['Портфель (%)', 'Бенчмарк (%)'],
                        barmode='group',
                        title='Сравнение доходности по периодам',
                        labels={'value': 'Доходность (%)', 'variable': ''}
                    )

                    st.plotly_chart(fig_periods, use_container_width=True)

                    # Лучшие и худшие периоды
                    st.subheader("Лучшие и худшие периоды")

                    worst_periods = PortfolioVisualization.create_worst_periods_table(
                        portfolio_returns, benchmark_returns
                    )

                    if not worst_periods.empty:
                        # Форматируем числовые колонки
                        for col in ['Доходность', 'Бенчмарк', 'Разница']:
                            worst_periods[col] = worst_periods[col].apply(lambda x: f"{x * 100:.2f}%")

                        st.dataframe(worst_periods, use_container_width=True)

    # Вкладка "Риск"
    with tabs[2]:
        st.subheader("Анализ риска")

        # Создаем подвкладки для разных видов анализа риска
        risk_tabs = st.tabs([
            "Основные метрики",
            "Анализ просадок",
            "VaR и CVaR",
            "Вклад в риск"
        ])

        with risk_tabs[0]:
            st.subheader("Основные метрики риска")

            # Сравнение метрик риска с бенчмарком
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Волатильность (годовая)",
                    f"{portfolio_metrics.get('volatility', 0) * 100:.2f}%",
                    f"{(portfolio_metrics.get('benchmark_volatility', 0) - portfolio_metrics.get('volatility', 0)) * 100:.2f}%"
                    if 'benchmark_volatility' in portfolio_metrics else None,
                    delta_color="inverse"
                )

            with col2:
                st.metric(
                    "Максимальная просадка",
                    f"{portfolio_metrics.get('max_drawdown', 0) * 100:.2f}%",
                    f"{(portfolio_metrics.get('benchmark_max_drawdown', 0) - portfolio_metrics.get('max_drawdown', 0)) * 100:.2f}%"
                    if 'benchmark_max_drawdown' in portfolio_metrics else None,
                    delta_color="inverse"
                )

            with col3:
                st.metric(
                    "Коэффициент Сортино",
                    f"{portfolio_metrics.get('sortino_ratio', 0):.2f}",
                    f"{portfolio_metrics.get('sortino_ratio', 0) - portfolio_metrics.get('benchmark_sortino_ratio', 0):.2f}"
                    if 'benchmark_sortino_ratio' in portfolio_metrics else None
                )

            with col4:
                st.metric(
                    "Коэффициент Кальмара",
                    f"{portfolio_metrics.get('calmar_ratio', 0):.2f}",
                    f"{portfolio_metrics.get('calmar_ratio', 0) - portfolio_metrics.get('benchmark_calmar_ratio', 0):.2f}"
                    if 'benchmark_calmar_ratio' in portfolio_metrics else None
                )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "VAR (95%)",
                    f"{portfolio_metrics.get('var_95', 0) * 100:.2f}%",
                    f"{(portfolio_metrics.get('benchmark_var_95', 0) - portfolio_metrics.get('var_95', 0)) * 100:.2f}%"
                    if 'benchmark_var_95' in portfolio_metrics else None,
                    delta_color="inverse"
                )

            with col2:
                st.metric(
                    "CVAR (95%)",
                    f"{portfolio_metrics.get('cvar_95', 0) * 100:.2f}%",
                    f"{(portfolio_metrics.get('benchmark_cvar_95', 0) - portfolio_metrics.get('cvar_95', 0)) * 100:.2f}%"
                    if 'benchmark_cvar_95' in portfolio_metrics else None,
                    delta_color="inverse"
                )

            with col3:
                st.metric(
                    "Восходящий захват",
                    f"{portfolio_metrics.get('up_capture', 0):.2f}",
                    f"{portfolio_metrics.get('up_capture', 0) - 1:.2f}" if 'up_capture' in portfolio_metrics else None
                )

            with col4:
                st.metric(
                    "Нисходящий захват",
                    f"{portfolio_metrics.get('down_capture', 0):.2f}",
                    f"{1 - portfolio_metrics.get('down_capture', 0):.2f}" if 'down_capture' in portfolio_metrics else None,
                    delta_color="inverse"
                )

            # График риск/доходность
            st.subheader("Соотношение риск/доходность")

            fig_risk_return = go.Figure()

            fig_risk_return.add_trace(go.Scatter(
                x=[portfolio_metrics.get('volatility', 0) * 100],
                y=[portfolio_metrics.get('annualized_return', 0) * 100],
                mode='markers',
                marker=dict(size=15, color='blue'),
                name='Портфель'
            ))

            if 'benchmark_volatility' in portfolio_metrics and 'benchmark_annualized_return' in portfolio_metrics:
                fig_risk_return.add_trace(go.Scatter(
                    x=[portfolio_metrics.get('benchmark_volatility', 0) * 100],
                    y=[portfolio_metrics.get('benchmark_annualized_return', 0) * 100],
                    mode='markers',
                    marker=dict(size=15, color='orange'),
                    name=benchmark
                ))

            # Добавляем безрисковую ставку
            risk_free = config.RISK_FREE_RATE * 100

            # Добавляем Capital Market Line
            x_range = np.linspace(0, max(portfolio_metrics.get('volatility', 0) * 100 * 1.5,
                                         portfolio_metrics.get('benchmark_volatility',
                                                               0) * 100 * 1.5 if 'benchmark_volatility' in portfolio_metrics else 0),
                                  100)
            if 'sharpe_ratio' in portfolio_metrics:
                slope = portfolio_metrics.get('sharpe_ratio', 0)
                y_range = risk_free + slope * x_range

                fig_risk_return.add_trace(go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    line=dict(color='green', dash='dash'),
                    name='CML (Capital Market Line)'
                ))

            fig_risk_return.update_layout(
                title='Риск/Доходность',
                xaxis_title='Риск (Волатильность, %)',
                yaxis_title='Доходность (%)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500
            )

            st.plotly_chart(fig_risk_return, use_container_width=True)

        with risk_tabs[1]:
            st.subheader("Анализ просадок")

            # Улучшенный график просадок
            cumulative_returns = (1 + portfolio_returns).cumprod()
            peak = cumulative_returns.cummax()
            drawdowns = (cumulative_returns / peak - 1) * 100

            fig_drawdown = go.Figure()

            fig_drawdown.add_trace(go.Scatter(
                x=drawdowns.index,
                y=drawdowns.values,
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(214, 39, 40, 0.2)',
                line=dict(color='#d62728'),
                name='Портфель'
            ))

            if benchmark_returns is not None:
                cumulative_benchmark = (1 + benchmark_returns).cumprod()
                peak_benchmark = cumulative_benchmark.cummax()
                benchmark_drawdowns = (cumulative_benchmark / peak_benchmark - 1) * 100

                fig_drawdown.add_trace(go.Scatter(
                    x=benchmark_drawdowns.index,
                    y=benchmark_drawdowns.values,
                    mode='lines',
                    line=dict(color='#ff7f0e', dash='dash'),
                    name=benchmark
                ))

            fig_drawdown.update_layout(
                title='Просадки',
                xaxis_title='Дата',
                yaxis_title='Просадка (%)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500
            )

            st.plotly_chart(fig_drawdown, use_container_width=True)

            # Детальный анализ просадок
            st.subheader("Топ-5 просадок")

            drawdown_analysis = RiskManagement.analyze_drawdowns(portfolio_returns)

            if not drawdown_analysis.empty:
                # Сортируем по глубине (сначала самые большие просадки) и берем топ-5
                top_drawdowns = drawdown_analysis.sort_values('depth', ascending=True).head(5)

                # Форматируем для отображения
                display_drawdowns = pd.DataFrame({
                    'Начало': top_drawdowns['start_date'].dt.strftime('%Y-%m-%d'),
                    'Дно': top_drawdowns['valley_date'].dt.strftime('%Y-%m-%d'),
                    'Восстановление': top_drawdowns['recovery_date'].fillna('В процессе').apply(
                        lambda x: x.strftime('%Y-%m-%d') if x != 'В процессе' else x
                    ),
                    'Глубина (%)': top_drawdowns['depth'] * 100,
                    'Длительность (дней)': top_drawdowns['length'],
                    'Восстановление (дней)': top_drawdowns['recovery']
                })

                st.dataframe(display_drawdowns.style.format({
                    'Глубина (%)': '{:.2f}%',
                    'Длительность (дней)': '{:.0f}',
                    'Восстановление (дней)': '{:.0f}'
                }), use_container_width=True)

            with risk_tabs[2]:
                st.subheader("Value at Risk (VaR) и Conditional VaR (CVaR)")

                col1, col2 = st.columns(2)

                with col1:
                    confidence_level = st.slider(
                        "Уровень доверия",
                        min_value=90,
                        max_value=99,
                        value=95,
                        step=1,
                        format="%d%%"
                    ) / 100

                    # Рассчитываем VaR разными методами
                    var_hist = RiskManagement.calculate_var_historical(portfolio_returns,
                                                                       confidence_level=confidence_level)
                    var_param = RiskManagement.calculate_var_parametric(portfolio_returns,
                                                                        confidence_level=confidence_level)
                    var_mc = RiskManagement.calculate_var_monte_carlo(portfolio_returns,
                                                                      confidence_level=confidence_level)
                    cvar = RiskManagement.calculate_cvar(portfolio_returns, confidence_level=confidence_level)

                    # Создаем таблицу с VaR
                    var_data = [
                        {"Метод": "Исторический", "Значение (%)": var_hist * 100},
                        {"Метод": "Параметрический", "Значение (%)": var_param * 100},
                        {"Метод": "Монте-Карло", "Значение (%)": var_mc * 100},
                        {"Метод": "CVaR", "Значение (%)": cvar * 100}
                    ]

                    var_df = pd.DataFrame(var_data)

                    st.dataframe(var_df.style.format({"Значение (%)": "{:.2f}%"}), use_container_width=True)

                with col2:
                    # Визуализация VaR на распределении доходностей
                    fig_var = px.histogram(
                        portfolio_returns * 100,
                        nbins=50,
                        title=f"Value at Risk при {confidence_level * 100:.0f}% уровне доверия",
                        labels={'value': 'Дневная доходность (%)', 'count': 'Частота'},
                        color_discrete_sequence=['lightskyblue']
                    )

                    # Добавляем линию VaR
                    fig_var.add_vline(
                        x=-var_hist * 100,
                        line_width=2,
                        line_color='red',
                        line_dash='dash',
                        annotation_text=f'VaR {confidence_level * 100:.0f}%: {var_hist * 100:.2f}%',
                        annotation_position="top right"
                    )

                    # Добавляем линию CVaR
                    fig_var.add_vline(
                        x=-cvar * 100,
                        line_width=2,
                        line_color='darkred',
                        line_dash='dash',
                        annotation_text=f'CVaR {confidence_level * 100:.0f}%: {cvar * 100:.2f}%',
                        annotation_position="bottom right"
                    )

                    fig_var.update_layout(
                        xaxis_title='Дневная доходность (%)',
                        yaxis_title='Частота',
                        showlegend=False
                    )

                    st.plotly_chart(fig_var, use_container_width=True)

            with risk_tabs[3]:
                st.subheader("Вклад в риск")

                # Расчет вклада в риск
                risk_contribution = RiskManagement.calculate_risk_contribution(returns, weights)

                if risk_contribution:
                    # Сортируем по вкладу в риск (по убыванию)
                    risk_contrib_sorted = {k: v for k, v in
                                           sorted(risk_contribution.items(), key=lambda item: item[1], reverse=True)}

                    # Создаем DataFrame
                    risk_contrib_df = pd.DataFrame({
                        'Актив': list(risk_contrib_sorted.keys()),
                        'Вклад в риск (%)': [v * 100 for v in risk_contrib_sorted.values()]
                    })

                    # Визуализация вклада в риск
                    fig_risk_contrib = px.bar(
                        risk_contrib_df,
                        x='Актив',
                        y='Вклад в риск (%)',
                        title='Вклад активов в общий риск портфеля',
                        color='Вклад в риск (%)',
                        color_continuous_scale='viridis'
                    )

                    st.plotly_chart(fig_risk_contrib, use_container_width=True)

                    # Сравнение вклада в риск с весами
                    compare_df = pd.DataFrame({
                        'Актив': list(risk_contrib_sorted.keys()),
                        'Вклад в риск (%)': [risk_contrib_sorted[t] * 100 for t in risk_contrib_sorted],
                        'Вес (%)': [weights.get(t, 0) * 100 for t in risk_contrib_sorted]
                    })

                    fig_risk_weight = go.Figure()

                    fig_risk_weight.add_trace(go.Bar(
                        x=compare_df['Актив'],
                        y=compare_df['Вклад в риск (%)'],
                        name='Вклад в риск (%)',
                        marker_color='#1f77b4'
                    ))

                    fig_risk_weight.add_trace(go.Bar(
                        x=compare_df['Актив'],
                        y=compare_df['Вес (%)'],
                        name='Вес (%)',
                        marker_color='#ff7f0e'
                    ))

                    fig_risk_weight.update_layout(
                        title='Сравнение вклада в риск и веса активов',
                        xaxis_title='Актив',
                        yaxis_title='Процент (%)',
                        barmode='group',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )

                    st.plotly_chart(fig_risk_weight, use_container_width=True)

                    # Диверсификация портфеля
                    st.subheader("Оценка диверсификации")

                    # Рассчитываем коэффициент диверсификации
                    portfolio_vol = portfolio_metrics.get('volatility', 0)
                    weighted_vol = sum([weights.get(ticker, 0) * returns[ticker].std() * np.sqrt(252)
                                        for ticker in weights if ticker in returns.columns])

                    diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 0

                    st.metric(
                        "Коэффициент диверсификации",
                        f"{diversification_ratio:.2f}",
                        "Выше - лучше"
                    )

                    st.info("""
                        **Коэффициент диверсификации** показывает отношение взвешенной суммы волатильностей 
                        отдельных активов к общей волатильности портфеля. Значение выше 1 указывает на 
                        положительный эффект диверсификации.
                        """)

    with tabs[3]:
        st.subheader("Анализ активов")

        # Создаем подвкладки для разных видов анализа активов
        assets_tabs = st.tabs([
            "Обзор активов",
            "Вклад активов",
            "Динамика цен",
            "Анализ актива"
        ])

        with assets_tabs[0]:
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

            if len(portfolio_data['assets']) > 10:
                st.warning(
                    f"Портфель содержит {len(portfolio_data['assets'])} активов. Для лучшей диверсификации рекомендуется не более 20-25 активов.")

        with assets_tabs[1]:
            st.subheader("Вклад активов в доходность и риск")

            # Рассчитываем вклад каждого актива в общую доходность
            cumulative_returns = {}
            weights_dict = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

            for ticker in returns.columns:
                if ticker in weights_dict:
                    cumulative_returns[ticker] = (1 + returns[ticker]).cumprod().iloc[-1] - 1

            # Создаем DataFrame
            contrib_df = pd.DataFrame({
                'Актив': list(cumulative_returns.keys()),
                'Доходность (%)': [cumulative_returns[ticker] * 100 for ticker in cumulative_returns],
                'Взвешенная доходность (%)': [cumulative_returns[ticker] * weights_dict[ticker] * 100 for ticker in
                                              cumulative_returns],
                'Вес (%)': [weights_dict[ticker] * 100 for ticker in cumulative_returns]
            })

            # Сортируем по взвешенной доходности
            contrib_df = contrib_df.sort_values('Взвешенная доходность (%)', ascending=False)

            # Визуализация вклада в доходность
            fig_contrib = px.bar(
                contrib_df,
                x='Актив',
                y='Взвешенная доходность (%)',
                title='Вклад активов в общую доходность',
                color='Взвешенная доходность (%)',
                color_continuous_scale='RdYlGn',
                text='Взвешенная доходность (%)'
            )

            fig_contrib.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_contrib.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

            st.plotly_chart(fig_contrib, use_container_width=True)

            # Сравнение вклада в доходность и риск
            if risk_contribution:
                compare_contrib_df = pd.DataFrame({
                    'Актив': list(cumulative_returns.keys()),
                    'Вклад в доходность (%)': [cumulative_returns[ticker] * weights_dict[ticker] * 100 /
                                               sum([cumulative_returns[t] * weights_dict[t] for t in
                                                    cumulative_returns])
                                               for ticker in cumulative_returns],
                    'Вклад в риск (%)': [risk_contribution.get(ticker, 0) * 100 for ticker in cumulative_returns]
                })

                # Сортируем по вкладу в доходность
                compare_contrib_df = compare_contrib_df.sort_values('Вклад в доходность (%)', ascending=False)

                # Визуализация сравнения
                fig_compare = go.Figure()

                fig_compare.add_trace(go.Bar(
                    x=compare_contrib_df['Актив'],
                    y=compare_contrib_df['Вклад в доходность (%)'],
                    name='Вклад в доходность (%)',
                    marker_color='#2ca02c'
                ))

                fig_compare.add_trace(go.Bar(
                    x=compare_contrib_df['Актив'],
                    y=compare_contrib_df['Вклад в риск (%)'],
                    name='Вклад в риск (%)',
                    marker_color='#d62728'
                ))

                fig_compare.update_layout(
                    title='Сравнение вклада в доходность и риск',
                    xaxis_title='Актив',
                    yaxis_title='Процент (%)',
                    barmode='group',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig_compare, use_container_width=True)

        with assets_tabs[2]:
            st.subheader("Динамика цен активов")

            # Нормализация цен для сравнения
            normalized_prices = close_prices.copy()
            for ticker in close_prices.columns:
                if ticker in weights:
                    first_price = normalized_prices[ticker].iloc[0]
                    if first_price > 0:
                        normalized_prices[ticker] = normalized_prices[ticker] / first_price

            # Выбор активов для отображения
            selected_assets = st.multiselect(
                "Выберите активы для отображения",
                options=list(weights.keys()),
                default=list(weights.keys())[:5] if len(weights) > 5 else list(weights.keys())
            )

            # Добавляем переключатель для показа в процентах
            show_percentage = st.checkbox("Показывать изменение в процентах", value=True)

            # Построение графика
            if selected_assets:
                # Создаем копию, чтобы не изменять оригинальные данные
                display_prices = close_prices.copy()

                # Обрабатываем данные в зависимости от выбранного режима отображения
                for ticker in display_prices.columns:
                    first_price = display_prices[ticker].iloc[0]
                    if first_price > 0:
                        if show_percentage:
                            # Процентное изменение относительно начальной даты
                            display_prices[ticker] = (display_prices[ticker] / first_price - 1) * 100
                        else:
                            # Нормализованное значение относительно начальной даты
                            display_prices[ticker] = display_prices[ticker] / first_price

                fig_norm_prices = go.Figure()

                # Цвета для активов - используем встроенную палитру Plotly
                colors = px.colors.qualitative.Plotly

                # Добавляем линии для каждого актива
                for i, ticker in enumerate(selected_assets):
                    if ticker in display_prices.columns:
                        fig_norm_prices.add_trace(go.Scatter(
                            x=display_prices.index,
                            y=display_prices[ticker],
                            mode='lines',
                            name=ticker,
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))

                # Добавляем бенчмарк, если есть
                if benchmark in display_prices.columns:
                    fig_norm_prices.add_trace(go.Scatter(
                        x=display_prices.index,
                        y=display_prices[benchmark],
                        mode='lines',
                        line=dict(color='#FFFFFF', width=2, dash='dash'),
                        name=f"{benchmark} (бенчмарк)"
                    ))

                # Настраиваем заголовок в зависимости от режима
                if show_percentage:
                    title = 'Изменение цен активов (% от начальной даты)'
                    y_axis_title = 'Изменение (%)'
                else:
                    title = 'Нормализованные цены активов (от начальной даты)'
                    y_axis_title = 'Нормализованная цена'

                fig_norm_prices.update_layout(
                    title=title,
                    xaxis_title='Дата',
                    yaxis_title=y_axis_title,
                    legend_title='Активы',
                    hovermode='x unified',
                    # Добавляем темную тему
                    template='plotly_dark',
                    # Улучшаем внешний вид
                    height=500,
                    margin=dict(l=40, r=40, t=50, b=40)
                )

                st.plotly_chart(fig_norm_prices, use_container_width=True)

        with assets_tabs[3]:
            st.subheader("Детальный анализ отдельного актива")

            # Выбор актива для анализа
            selected_asset = st.selectbox(
                "Выберите актив для детального анализа",
                options=list(weights.keys())
            )

            if selected_asset in close_prices.columns:
                asset_price = close_prices[selected_asset]
                asset_return = returns[selected_asset]

                # Статистика актива
                st.write("### Статистика актива")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    total_return = (asset_price.iloc[-1] / asset_price.iloc[0]) - 1
                    st.metric(
                        "Общая доходность",
                        f"{total_return * 100:.2f}%"
                    )

                with col2:
                    annualized_return = (1 + total_return) ** (252 / len(asset_return)) - 1
                    st.metric(
                        "Годовая доходность",
                        f"{annualized_return * 100:.2f}%"
                    )

                with col3:
                    volatility = asset_return.std() * np.sqrt(252)
                    st.metric(
                        "Волатильность",
                        f"{volatility * 100:.2f}%"
                    )

                with col4:
                    max_drawdown = PortfolioAnalytics.calculate_max_drawdown(asset_return)
                    st.metric(
                        "Макс. просадка",
                        f"{max_drawdown * 100:.2f}%"
                    )

                # Получаем данные объема для выбранного актива
                with st.spinner('Загрузка данных объема...'):
                    # Загружаем оригинальные данные только для выбранного актива, чтобы получить объем
                    asset_data = data_fetcher.get_historical_prices(
                        selected_asset,
                        start_date_str,
                        end_date_str
                    )

                    has_volume = not asset_data.empty and 'Volume' in asset_data.columns

                # График цены и объема
                if has_volume:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])

                    fig.add_trace(
                        go.Scatter(
                            x=asset_price.index,
                            y=asset_price.values,
                            mode='lines',
                            name='Цена',
                            line=dict(color='#1f77b4')
                        ),
                        secondary_y=False
                    )

                    fig.add_trace(
                        go.Bar(
                            x=asset_data.index,
                            y=asset_data['Volume'],
                            name='Объем',
                            marker=dict(color='#2ca02c', opacity=0.3)
                        ),
                        secondary_y=True
                    )


                    fig.update_layout(
                        title=f'{selected_asset} - Цена и объем',
                        xaxis_title='Дата',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode='x unified'
                    )

                    fig.update_yaxes(title_text="Цена", secondary_y=False)
                    fig.update_yaxes(title_text="Объем", secondary_y=True)

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Если данные об объеме недоступны, просто показываем цену
                    fig = px.line(
                        asset_price,
                        title=f'{selected_asset} - Цена',
                        labels={'value': 'Цена', 'index': 'Дата'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Сравнение с портфелем и бенчмарком
                st.subheader("Сравнение доходности")

                cumulative_asset = (1 + asset_return).cumprod() - 1
                cumulative_portfolio = (1 + portfolio_returns).cumprod() - 1

                fig_compare = go.Figure()

                fig_compare.add_trace(go.Scatter(
                    x=cumulative_asset.index,
                    y=cumulative_asset.values * 100,
                    mode='lines',
                    name=selected_asset,
                    line=dict(color='#1f77b4')
                ))

                fig_compare.add_trace(go.Scatter(
                    x=cumulative_portfolio.index,
                    y=cumulative_portfolio.values * 100,
                    mode='lines',
                    name='Портфель',
                    line=dict(color='#2ca02c')
                ))

                if benchmark_returns is not None:
                    cumulative_benchmark = (1 + benchmark_returns).cumprod() - 1
                    fig_compare.add_trace(go.Scatter(
                        x=cumulative_benchmark.index,
                        y=cumulative_benchmark.values * 100,
                        mode='lines',
                        name=benchmark,
                        line=dict(color='#ff7f0e', dash='dash')
                    ))

                fig_compare.update_layout(
                    title=f'Сравнение доходности {selected_asset} с портфелем и бенчмарком',
                    xaxis_title='Дата',
                    yaxis_title='Кумулятивная доходность (%)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode='x unified'
                )

                st.plotly_chart(fig_compare, use_container_width=True)

                # Корреляции актива
                st.subheader("Корреляции")

                correlations = returns.corr()[selected_asset].drop(selected_asset).sort_values(ascending=False)

                fig_corr = px.bar(
                    x=correlations.index,
                    y=correlations.values,
                    title=f'Корреляция {selected_asset} с другими активами',
                    labels={'x': 'Актив', 'y': 'Корреляция'},
                    color=correlations.values,
                    color_continuous_scale='RdBu_r',
                    range_color=[-1, 1]
                )

                st.plotly_chart(fig_corr, use_container_width=True)


    # Вкладка "Корреляции"
    with tabs[4]:
        st.subheader("Анализ корреляций")

        # Создаем подвкладки для разных видов анализа корреляций
        corr_tabs = st.tabs([
            "Корреляционная матрица",
            "Корреляция с бенчмарком",
            "Кластерный анализ",
            "Динамика корреляций"
        ])

        with corr_tabs[0]:
            # Корреляционная матрица
            correlation_matrix = returns.corr()

            # Создаем тепловую карту корреляций
            fig_corr = px.imshow(
                correlation_matrix,
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1,
                title='Корреляционная матрица'
            )

            fig_corr.update_layout(
                height=600,
                xaxis_title='Актив',
                yaxis_title='Актив'
            )

            st.plotly_chart(fig_corr, use_container_width=True)

            # Статистика корреляций
            st.subheader("Статистика корреляций")

            # Создаем маску для верхнего треугольника матрицы (без диагонали)
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

            # Получаем все корреляции из верхнего треугольника
            all_correlations = correlation_matrix.values[mask]

            # Отображаем статистику
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Средняя корреляция",
                    f"{np.mean(all_correlations):.2f}"
                )

            with col2:
                st.metric(
                    "Медианная корреляция",
                    f"{np.median(all_correlations):.2f}"
                )

            with col3:
                st.metric(
                    "Мин. корреляция",
                    f"{np.min(all_correlations):.2f}"
                )

            with col4:
                st.metric(
                    "Макс. корреляция",
                    f"{np.max(all_correlations):.2f}"
                )

        with corr_tabs[1]:
            # Корреляция с бенчмарком
            if benchmark in returns.columns:
                correlations_with_benchmark = returns.corr()[benchmark].drop(benchmark).sort_values(ascending=False)

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

                # Отображаем таблицу с корреляциями
                corr_df = pd.DataFrame({
                    'Актив': correlations_with_benchmark.index,
                    'Корреляция с бенчмарком': correlations_with_benchmark.values,
                    'Бета': [PortfolioAnalytics.calculate_beta(returns[ticker], returns[benchmark])
                             for ticker in correlations_with_benchmark.index]
                })

                def color_correlation(val):
                    if val > 0.8:
                        return 'background-color: rgba(255, 0, 0, 0.2)'
                    elif val > 0.5:
                        return 'background-color: rgba(255, 165, 0, 0.2)'
                    elif val < -0.5:
                        return 'background-color: rgba(0, 128, 0, 0.2)'
                    return ''

                styled_corr_df = corr_df.style.format({
                    'Корреляция с бенчмарком': '{:.2f}',
                    'Бета': '{:.2f}'
                }).applymap(color_correlation, subset=['Корреляция с бенчмарком'])

                st.dataframe(styled_corr_df, use_container_width=True)

        with corr_tabs[2]:
            st.subheader("Кластерный анализ корреляций")

            try:
                from scipy.cluster import hierarchy
                from scipy.spatial import distance

                # Преобразуем корреляционную матрицу в матрицу расстояний
                dist = distance.squareform(1 - np.abs(correlation_matrix))

                # Выполняем иерархическую кластеризацию
                linkage = hierarchy.linkage(dist, method='average')

                # Получаем результат дендрограммы, но без отображения
                dendrogram_result = hierarchy.dendrogram(linkage, no_plot=True)

                # Получаем порядок кластеризации
                order = dendrogram_result['leaves']

                # Переупорядочиваем корреляционную матрицу
                reordered_corr = correlation_matrix.iloc[order, order]

                # Визуализируем кластеризованную тепловую карту
                fig_cluster = px.imshow(
                    reordered_corr,
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    title='Кластеризованная корреляционная матрица'
                )

                fig_cluster.update_layout(
                    height=600,
                    xaxis_title='Актив',
                    yaxis_title='Актив'
                )

                st.plotly_chart(fig_cluster, use_container_width=True)

                # Визуализируем дендрограмму
                fig_dendro = go.Figure()

                # Создаем дендрограмму с метками
                dendro = hierarchy.dendrogram(linkage, labels=correlation_matrix.index, orientation='bottom',
                                              no_plot=True)

                # Преобразуем дендрограмму в формат Plotly
                for i, d in enumerate(dendro['dcoord']):
                    fig_dendro.add_trace(go.Scatter(
                        x=dendro['icoord'][i],
                        y=dendro['dcoord'][i],
                        mode='lines',
                        line=dict(color='blue', width=1),
                        hoverinfo='none',
                        showlegend=False
                    ))

                # Вычисляем позиции меток вручную, если нет ivl_positions
                if 'ivl_positions' not in dendro:
                    # Вычисляем позиции самостоятельно
                    leaf_count = len(dendro['ivl'])
                    leaf_positions = []
                    for i in range(leaf_count):
                        # Найдем минимальные x-координаты для каждого листа
                        leaf_x = []
                        for j, (xx, yy) in enumerate(zip(dendro['icoord'], dendro['dcoord'])):
                            if yy[0] == 0.0 and yy[-1] == 0.0:  # Только линии, идущие к листам
                                leaf_x.append(xx[0])
                        leaf_positions = sorted(leaf_x)
                        if len(leaf_positions) == leaf_count:
                            break

                    # Если не удалось вычислить позиции, используем равномерное распределение
                    if len(leaf_positions) != leaf_count:
                        leaf_positions = [10 * i for i in range(leaf_count)]
                else:
                    leaf_positions = dendro['ivl_positions']

                fig_dendro.update_layout(
                    title='Дендрограмма кластеризации активов',
                    xaxis=dict(
                        tickmode='array',
                        tickvals=leaf_positions,
                        ticktext=dendro['ivl'],
                        tickangle=45
                    ),
                    yaxis=dict(title='Расстояние'),
                    height=500
                )

                st.plotly_chart(fig_dendro, use_container_width=True)

            except ImportError:
                st.error("Для кластерного анализа требуется установленный пакет scipy.")
                st.info("Установите scipy с помощью команды: pip install scipy")
            except Exception as e:
                st.error(f"Не удалось выполнить кластерный анализ: {str(e)}")
                st.info("Для кластерного анализа требуется установленный пакет scipy.")

        with corr_tabs[3]:
            st.subheader("Динамика корреляций")

            if isinstance(returns.index, pd.DatetimeIndex) and benchmark in returns.columns:
                # Выбор активов для анализа динамики
                selected_corr_assets = st.multiselect(
                    "Выберите активы для анализа динамики корреляций",
                    options=[t for t in weights.keys() if t != benchmark],
                    default=list(weights.keys())[:3] if len(weights) > 3 else list(weights.keys())
                )

                window_size = st.slider(
                    "Размер окна (дней) для расчета скользящей корреляции",
                    min_value=30,
                    max_value=252,
                    value=60,
                    step=10
                )

                if selected_corr_assets:
                    # Создаем DataFrame для скользящих корреляций
                    rolling_corrs = pd.DataFrame(index=returns.index)

                    for ticker in selected_corr_assets:
                        if ticker in returns.columns:
                            # Рассчитываем скользящую корреляцию с бенчмарком
                            rolling_corrs[ticker] = returns[ticker].rolling(window=window_size).corr(returns[benchmark])

                    # Удаляем начальные NaN
                    rolling_corrs = rolling_corrs.dropna()

                    if not rolling_corrs.empty:
                        # Визуализируем скользящие корреляции
                        fig_rolling_corrs = go.Figure()

                        for ticker in rolling_corrs.columns:
                            fig_rolling_corrs.add_trace(go.Scatter(
                                x=rolling_corrs.index,
                                y=rolling_corrs[ticker],
                                mode='lines',
                                name=f"{ticker} с {benchmark}"
                            ))

                        fig_rolling_corrs.update_layout(
                            title=f'Скользящая корреляция активов с {benchmark} (окно {window_size} дней)',
                            xaxis_title='Дата',
                            yaxis_title='Корреляция',
                            yaxis=dict(range=[-1, 1]),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            hovermode='x unified'
                        )

                        # Добавляем линии для уровней корреляции
                        fig_rolling_corrs.add_hline(y=0.8, line_dash="dash", line_color="red",
                                                    annotation_text="Сильная корреляция (0.8)")
                        fig_rolling_corrs.add_hline(y=0.5, line_dash="dash", line_color="orange",
                                                    annotation_text="Средняя корреляция (0.5)")
                        fig_rolling_corrs.add_hline(y=0, line_dash="dash", line_color="green",
                                                    annotation_text="Нет корреляции (0)")
                        fig_rolling_corrs.add_hline(y=-0.5, line_dash="dash", line_color="blue",
                                                    annotation_text="Средняя отрицательная (-0.5)")

                        st.plotly_chart(fig_rolling_corrs, use_container_width=True)

                    # Рассчитываем среднюю корреляцию портфеля с бенчмарком
                    weighted_corr = pd.Series(index=returns.index)

                    # Создаем маску для валидных дат
                    valid_dates = returns.index

                    # Для каждой даты рассчитываем взвешенную корреляцию
                    window_indices = range(len(valid_dates) - window_size + 1)

                    for i in window_indices:
                        window_start = i
                        window_end = i + window_size
                        window_dates = valid_dates[window_start:window_end]

                        # Рассчитываем корреляции для текущего окна
                        window_corrs = {}
                        for ticker in weights:
                            if ticker in returns.columns and ticker != benchmark:
                                ticker_returns = returns.loc[window_dates, ticker]
                                bench_returns = returns.loc[window_dates, benchmark]
                                if not ticker_returns.empty and not bench_returns.empty:
                                    window_corrs[ticker] = ticker_returns.corr(bench_returns)

                        # Рассчитываем взвешенную корреляцию
                        if window_corrs:
                            total_weight = sum(weights[t] for t in window_corrs)
                            if total_weight > 0:
                                avg_corr = sum(window_corrs[t] * weights[t] for t in window_corrs) / total_weight
                                if i + window_size <= len(valid_dates):
                                    weighted_corr[valid_dates[i + window_size - 1]] = avg_corr

                    # Удаляем NaN
                    weighted_corr = weighted_corr.dropna()

                    if not weighted_corr.empty:
                        # Визуализируем среднюю корреляцию портфеля
                        fig_avg_corr = go.Figure()

                        fig_avg_corr.add_trace(go.Scatter(
                            x=weighted_corr.index,
                            y=weighted_corr.values,
                            mode='lines',
                            line=dict(color='purple', width=2),
                            name='Средняя корреляция портфеля'
                        ))

                        fig_avg_corr.update_layout(
                            title=f'Средняя взвешенная корреляция портфеля с {benchmark} (окно {window_size} дней)',
                            xaxis_title='Дата',
                            yaxis_title='Корреляция',
                            yaxis=dict(range=[-1, 1]),
                            hovermode='x unified'
                        )

                        # Добавляем линии для уровней корреляции
                        fig_avg_corr.add_hline(y=0.8, line_dash="dash", line_color="red",
                                               annotation_text="Сильная корреляция (0.8)")
                        fig_avg_corr.add_hline(y=0.5, line_dash="dash", line_color="orange",
                                               annotation_text="Средняя корреляция (0.5)")
                        fig_avg_corr.add_hline(y=0, line_dash="dash", line_color="green",
                                               annotation_text="Нет корреляции (0)")

                        st.plotly_chart(fig_avg_corr, use_container_width=True)

    # Обновленный код для вкладки "Стресс-тестирование" в portfolio_analysis.py

    with tabs[5]:
        st.subheader("Стресс-тестирование")

        # Создаем подвкладки для разных видов стресс-тестов
        stress_tabs = st.tabs([
            "Исторические сценарии",
            "Пользовательские сценарии",
            "Анализ чувствительности",
            "Экстремальные сценарии"
        ])

        with stress_tabs[0]:
            st.subheader("Исторические сценарии стресс-тестирования")

            # Выбор сценария стресс-тестирования
            scenarios = [
                "financial_crisis_2008",
                "covid_2020",
                "tech_bubble_2000",
                "black_monday_1987",
                "inflation_shock",
                "rate_hike_2018",
                "moderate_recession",
                "severe_recession"
            ]

            scenario_names = {
                "financial_crisis_2008": "Финансовый кризис 2008",
                "covid_2020": "Пандемия COVID-19 (2020)",
                "tech_bubble_2000": "Крах доткомов (2000-2002)",
                "black_monday_1987": "Черный понедельник (1987)",
                "inflation_shock": "Инфляционный шок (2021-2022)",
                "rate_hike_2018": "Повышение ставок ФРС (2018)",
                "moderate_recession": "Умеренная рецессия",
                "severe_recession": "Тяжелая рецессия"
            }

            # Выбор сценария и стоимости портфеля
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

            # Проводим улучшенное стресс-тестирование с историческими данными
            if st.button("Провести исторический стресс-тест"):
                with st.spinner("Выполнение исторического стресс-теста..."):
                    # Получаем тикеры из портфеля
                    tickers = [asset['ticker'] for asset in portfolio_data['assets']]

                    # Получаем веса из портфеля
                    weights = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

                    # Проводим улучшенное стресс-тестирование
                    stress_test_result = RiskManagement.perform_historical_stress_test(
                        data_fetcher, tickers, weights, selected_scenario, portfolio_value
                    )

                if 'error' in stress_test_result:
                    st.error(f"Ошибка при выполнении стресс-теста: {stress_test_result['error']}")
                else:
                    st.subheader(f"Результаты стресс-теста: {stress_test_result['scenario_name']}")
                    st.write(f"**Период:** {stress_test_result['period']}")
                    st.write(f"**Описание:** {stress_test_result['scenario_description']}")

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

                    # Создаем DataFrame для воздействия на отдельные позиции
                    position_data = []
                    for ticker, impact in stress_test_result['position_impacts'].items():
                        position_data.append({
                            'Тикер': ticker,
                            'Вес (%)': impact['weight'] * 100,
                            'Изменение (%)': impact['price_change'] * 100,
                            'Стоимость позиции ($)': impact['position_value'],
                            'Потери ($)': impact['position_loss']
                        })

                    position_df = pd.DataFrame(position_data)

                    # Сортируем по потерям
                    position_df = position_df.sort_values('Потери ($)', ascending=True)

                    st.subheader("Влияние на отдельные активы")
                    st.dataframe(position_df.style.format({
                        'Вес (%)': '{:.2f}%',
                        'Изменение (%)': '{:.2f}%',
                        'Стоимость позиции ($)': '${:.2f}',
                        'Потери ($)': '${:.2f}'
                    }), use_container_width=True)

                    # Визуализация потерь по активам
                    fig_pos_loss = px.bar(
                        position_df,
                        x='Тикер',
                        y='Потери ($)',
                        title='Потери по активам',
                        color='Изменение (%)',
                        color_continuous_scale='RdYlGn'
                    )

                    st.plotly_chart(fig_pos_loss, use_container_width=True)

                    # Визуализация результатов стресс-теста
                    st.subheader("Визуализация стресс-теста")

                    fig_stress = go.Figure()

                    # Создаем временную шкалу (в месяцах)
                    months = list(range(-1, int(stress_test_result['recovery_months']) + 2))
                    values = []

                    # Добавляем начальное значение
                    values.append(portfolio_value)

                    # Добавляем значение после шока
                    values.append(stress_test_result['portfolio_after_shock'])

                    # Добавляем значения восстановления (линейное приближение)
                    recovery_rate = (portfolio_value - stress_test_result['portfolio_after_shock']) / \
                                    stress_test_result[
                                        'recovery_months'] if stress_test_result['recovery_months'] > 0 else 0

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
                        title=f"Стресс-тест: {stress_test_result['scenario_name']}",
                        xaxis_title="Месяцы",
                        yaxis_title="Стоимость портфеля ($)",
                        hovermode="x"
                    )

                    st.plotly_chart(fig_stress, use_container_width=True)

                    # Сравниваем воздействие на уровне классов активов или секторов, если данные доступны
                    sectors = {}
                    for asset in portfolio_data['assets']:
                        if 'sector' in asset and asset['sector'] != 'N/A':
                            sector = asset['sector']
                            ticker = asset['ticker']
                            if sector not in sectors:
                                sectors[sector] = {
                                    'weight': 0,
                                    'impact': 0,
                                    'tickers': []
                                }

                            # Добавляем информацию о тикере
                            sectors[sector]['tickers'].append(ticker)
                            sectors[sector]['weight'] += weights[ticker]

                            # Добавляем вклад в общее воздействие
                            if ticker in stress_test_result['position_impacts']:
                                impact = stress_test_result['position_impacts'][ticker]['price_change'] * weights[
                                    ticker]
                                sectors[sector]['impact'] += impact

                    if sectors:
                        # Создаем DataFrame для секторного анализа
                        sector_data = []
                        for sector, data in sectors.items():
                            sector_data.append({
                                'Сектор': sector,
                                'Вес (%)': data['weight'] * 100,
                                'Изменение (%)': (data['impact'] / data['weight']) * 100 if data['weight'] > 0 else 0,
                                'Общий вклад (%)': data['impact'] * 100,
                                'Количество активов': len(data['tickers'])
                            })

                        sector_df = pd.DataFrame(sector_data)
                        sector_df = sector_df.sort_values('Общий вклад (%)', ascending=True)

                        st.subheader("Влияние на уровне секторов")
                        st.dataframe(sector_df.style.format({
                            'Вес (%)': '{:.2f}%',
                            'Изменение (%)': '{:.2f}%',
                            'Общий вклад (%)': '{:.2f}%'
                        }), use_container_width=True)

                        # Визуализация влияния по секторам
                        fig_sector = px.bar(
                            sector_df,
                            x='Сектор',
                            y='Общий вклад (%)',
                            title='Влияние стресс-сценария по секторам',
                            color='Изменение (%)',
                            color_continuous_scale='RdYlGn'
                        )

                        st.plotly_chart(fig_sector, use_container_width=True)

        # Остальные вкладки стресс-тестирования остаются без изменений
        # Сохраняем существующий функционал для других вкладок

        with stress_tabs[1]:
            st.subheader("Пользовательский стресс-тест")

            st.write("""
            Создайте собственный сценарий стресс-тестирования, указав шоковые изменения 
            для отдельных активов или классов активов. Учитываются исторические корреляции и беты активов.
            """)

            # Ввод стоимости портфеля
            custom_portfolio_value = st.number_input(
                "Стоимость портфеля ($)",
                min_value=1000,
                value=10000,
                step=1000,
                key="custom_portfolio_value"
            )

            # Выбор типа шока
            shock_type = st.radio(
                "Тип пользовательского сценария",
                ["По отдельным активам", "По секторам", "Комбинированный шок"]
            )

            # Опции для расширенного анализа
            with st.expander("Настройки расширенного анализа"):
                col1, col2 = st.columns(2)

                with col1:
                    use_correlations = st.checkbox("Учитывать корреляции между активами", value=True,
                                                   help="Учитывает, как шок одного актива влияет на другие активы на основе исторических корреляций")

                with col2:
                    use_beta = st.checkbox("Учитывать бету активов", value=True,
                                           help="Учитывает чувствительность каждого актива к рыночным движениям")

            custom_shocks = {}
            asset_sectors = {}

            if shock_type == "По отдельным активам":
                # Создаем ползунки для каждого актива
                st.subheader("Задайте шоки для отдельных активов")

                # Добавляем общий рыночный шок
                market_shock = st.slider(
                    "Общий рыночный шок (%)",
                    min_value=-90,
                    max_value=50,
                    value=0,
                    step=5,
                    key="market_shock"
                ) / 100

                custom_shocks['market'] = market_shock
                custom_shocks['assets'] = {}

                # Создаем два столбца для более компактного отображения
                col1, col2 = st.columns(2)

                # Распределяем активы по двум столбцам
                assets_list = list(portfolio_data['assets'])
                mid_point = len(assets_list) // 2

                with col1:
                    for i, asset in enumerate(assets_list[:mid_point]):
                        ticker = asset['ticker']

                        # Извлекаем информацию о секторе, если доступна
                        sector = asset.get('sector', 'N/A')
                        if sector != 'N/A':
                            asset_sectors[ticker] = sector

                        # Создаем слайдер для каждого актива
                        asset_shock = st.slider(
                            f"{ticker} - {asset.get('name', '')} (%)",
                            min_value=-90,
                            max_value=50,
                            value=0,
                            step=5,
                            key=f"shock_asset_{ticker}"
                        ) / 100

                        if asset_shock != 0:
                            custom_shocks['assets'][ticker] = asset_shock

                with col2:
                    for i, asset in enumerate(assets_list[mid_point:]):
                        ticker = asset['ticker']

                        # Извлекаем информацию о секторе, если доступна
                        sector = asset.get('sector', 'N/A')
                        if sector != 'N/A':
                            asset_sectors[ticker] = sector

                        # Создаем слайдер для каждого актива
                        asset_shock = st.slider(
                            f"{ticker} - {asset.get('name', '')} (%)",
                            min_value=-90,
                            max_value=50,
                            value=0,
                            step=5,
                            key=f"shock_asset_{ticker}_col2"
                        ) / 100

                        if asset_shock != 0:
                            custom_shocks['assets'][ticker] = asset_shock

            elif shock_type == "По секторам":
                # Получаем уникальные секторы
                sectors = {}
                for asset in portfolio_data['assets']:
                    if 'sector' in asset and asset['sector'] != 'N/A':
                        sector = asset['sector']
                        ticker = asset['ticker']
                        if sector not in sectors:
                            sectors[sector] = []
                        sectors[sector].append(ticker)

                        # Сохраняем сектор для каждого актива
                        asset_sectors[ticker] = sector

                # Создаем ползунки для каждого сектора
                st.subheader("Задайте шоки для секторов")

                # Добавляем общий рыночный шок
                market_shock = st.slider(
                    "Общий рыночный шок (%)",
                    min_value=-90,
                    max_value=50,
                    value=0,
                    step=5,
                    key="market_shock_sectors"
                ) / 100

                custom_shocks['market'] = market_shock

                # Создаем ползунки для каждого сектора
                if sectors:
                    for sector, tickers in sectors.items():
                        sector_shock = st.slider(
                            f"Шок для сектора {sector} ({len(tickers)} активов) (%)",
                            min_value=-90,
                            max_value=50,
                            value=0,
                            step=5,
                            key=f"shock_sector_{sector}"
                        ) / 100

                        if sector_shock != 0:
                            custom_shocks[sector] = sector_shock
                else:
                    st.warning("Информация о секторах для активов недоступна. Используйте шок по отдельным активам.")

            else:  # Комбинированный шок
                st.subheader("Комбинированный шок для рынка, секторов и активов")

                # Общий рыночный шок
                market_shock = st.slider(
                    "Общий рыночный шок (%)",
                    min_value=-90,
                    max_value=50,
                    value=-25,
                    step=5,
                    key="market_shock_combined"
                ) / 100

                custom_shocks['market'] = market_shock

                # Получаем секторы
                sectors = {}
                for asset in portfolio_data['assets']:
                    if 'sector' in asset and asset['sector'] != 'N/A':
                        sector = asset['sector']
                        ticker = asset['ticker']
                        if sector not in sectors:
                            sectors[sector] = []
                        sectors[sector].append(ticker)

                        # Сохраняем сектор для каждого актива
                        asset_sectors[ticker] = sector

                # Секторные шоки (если есть секторы)
                if sectors:
                    st.subheader("Секторные шоки:")
                    # Выбираем несколько ключевых секторов для указания шоков
                    key_sectors = list(sectors.keys())[:min(5, len(sectors))]

                    for sector in key_sectors:
                        sector_shock = st.slider(
                            f"Дополнительный шок для сектора {sector} (%)",
                            min_value=-50,
                            max_value=30,
                            value=0,
                            step=5,
                            key=f"shock_sector_combined_{sector}"
                        ) / 100

                        if sector_shock != 0:
                            custom_shocks[sector] = sector_shock

                # Шоки для отдельных активов
                st.subheader("Шоки для отдельных активов (опционально):")

                # Выбираем активы для задания индивидуальных шоков
                custom_assets = st.multiselect(
                    "Выберите активы для задания индивидуальных шоков",
                    options=[asset['ticker'] for asset in portfolio_data['assets']],
                    default=[]
                )

                custom_shocks['assets'] = {}

                if custom_assets:
                    for ticker in custom_assets:
                        asset_name = next((asset.get('name', ticker) for asset in portfolio_data['assets']
                                           if asset['ticker'] == ticker), ticker)

                        asset_shock = st.slider(
                            f"Дополнительный шок для {ticker} - {asset_name} (%)",
                            min_value=-50,
                            max_value=30,
                            value=0,
                            step=5,
                            key=f"shock_asset_combined_{ticker}"
                        ) / 100

                        if asset_shock != 0:
                            custom_shocks['assets'][ticker] = asset_shock

            # Кнопка для запуска пользовательского стресс-теста
            if st.button("Запустить пользовательский стресс-тест"):
                with st.spinner("Выполнение пользовательского стресс-теста..."):
                    # Загружаем данные для анализа
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                    end_date = datetime.now().strftime('%Y-%m-%d')

                    # Получаем список тикеров и весов
                    tickers = [asset['ticker'] for asset in portfolio_data['assets']]
                    weights = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

                    # Добавляем SPY для расчета беты, если его нет в списке
                    if 'SPY' not in tickers:
                        tickers.append('SPY')

                    # Загружаем исторические данные
                    price_data = data_fetcher.get_batch_data(tickers, start_date, end_date)

                    # Проверяем, что данные загружены успешно
                    if not price_data or all(df.empty for df in price_data.values()):
                        st.error(
                            "Не удалось загрузить исторические данные. Пожалуйста, проверьте тикеры или измените период.")
                    else:
                        # Создаем DataFrame с ценами закрытия
                        close_prices = pd.DataFrame()

                        for ticker, df in price_data.items():
                            if not df.empty:
                                if 'Adj Close' in df.columns:
                                    close_prices[ticker] = df['Adj Close']
                                elif 'Close' in df.columns:
                                    close_prices[ticker] = df['Close']

                        # Рассчитываем доходности
                        returns = PortfolioAnalytics.calculate_returns(close_prices)

                        # Проводим расширенный стресс-тест
                        stress_test_result = RiskManagement.perform_advanced_custom_stress_test(
                            returns=returns,
                            weights=weights,
                            custom_shocks=custom_shocks,
                            asset_sectors=asset_sectors,
                            portfolio_value=custom_portfolio_value,
                            correlation_adjusted=use_correlations,
                            use_beta=use_beta
                        )

                        if 'error' in stress_test_result:
                            st.error(f"Ошибка при выполнении стресс-теста: {stress_test_result['error']}")
                        else:
                            st.subheader("Результаты пользовательского стресс-теста")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    "Изменение стоимости",
                                    f"${stress_test_result['portfolio_loss']:.2f}",
                                    f"{stress_test_result['loss_percentage'] * 100:.1f}%",
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

                            # Создаем DataFrame для детального анализа активов
                            detailed_data = []
                            for ticker, impact in stress_test_result['detailed_impacts'].items():
                                detailed_data.append({
                                    'Тикер': ticker,
                                    'Вес (%)': impact['weight'] * 100,
                                    'Бета': impact['beta'],
                                    'Изменение (%)': impact['shock_percentage'] * 100,
                                    'Стоимость позиции ($)': impact['position_value'],
                                    'Потери ($)': impact['position_loss']
                                })

                            detailed_df = pd.DataFrame(detailed_data)

                            # Сортируем по потерям
                            detailed_df = detailed_df.sort_values('Потери ($)', ascending=True)

                            st.subheader("Детальный анализ влияния на активы")
                            st.dataframe(detailed_df.style.format({
                                'Вес (%)': '{:.2f}%',
                                'Бета': '{:.2f}',
                                'Изменение (%)': '{:.2f}%',
                                'Стоимость позиции ($)': '${:.2f}',
                                'Потери ($)': '${:.2f}'
                            }), use_container_width=True)

                            # Визуализация потерь по активам
                            fig_pos_loss = px.bar(
                                detailed_df,
                                x='Тикер',
                                y='Потери ($)',
                                title='Потери по активам',
                                color='Изменение (%)',
                                color_continuous_scale='RdYlGn',
                                hover_data=['Вес (%)', 'Бета']
                            )

                            st.plotly_chart(fig_pos_loss, use_container_width=True)

                            # Секторный анализ (если доступны данные о секторах)
                            if asset_sectors:
                                sector_impacts = {}
                                for ticker, impact in stress_test_result['detailed_impacts'].items():
                                    if ticker in asset_sectors:
                                        sector = asset_sectors[ticker]
                                        if sector not in sector_impacts:
                                            sector_impacts[sector] = {
                                                'weight': 0,
                                                'loss': 0,
                                                'value': 0,
                                                'tickers_count': 0
                                            }

                                        sector_impacts[sector]['weight'] += impact['weight']
                                        sector_impacts[sector]['loss'] += impact['position_loss']
                                        sector_impacts[sector]['value'] += impact['position_value']
                                        sector_impacts[sector]['tickers_count'] += 1

                                # Создаем DataFrame для секторного анализа
                                sector_data = []
                                for sector, data in sector_impacts.items():
                                    sector_data.append({
                                        'Сектор': sector,
                                        'Вес (%)': data['weight'] * 100,
                                        'Количество активов': data['tickers_count'],
                                        'Стоимость ($)': data['value'],
                                        'Потери ($)': data['loss'],
                                        'Изменение (%)': (data['loss'] / data['value'] * 100) if data[
                                                                                                     'value'] > 0 else 0
                                    })

                                sector_df = pd.DataFrame(sector_data)
                                sector_df = sector_df.sort_values('Потери ($)', ascending=True)

                                st.subheader("Секторный анализ")
                                st.dataframe(sector_df.style.format({
                                    'Вес (%)': '{:.2f}%',
                                    'Стоимость ($)': '${:.2f}',
                                    'Потери ($)': '${:.2f}',
                                    'Изменение (%)': '{:.2f}%'
                                }), use_container_width=True)

                                # Визуализация потерь по секторам
                                fig_sector_loss = px.bar(
                                    sector_df,
                                    x='Сектор',
                                    y='Потери ($)',
                                    title='Потери по секторам',
                                    color='Изменение (%)',
                                    color_continuous_scale='RdYlGn',
                                    hover_data=['Вес (%)', 'Количество активов']
                                )

                                st.plotly_chart(fig_sector_loss, use_container_width=True)

                            # Визуализация восстановления
                            st.subheader("Прогноз восстановления")

                            # Создаем временную шкалу (в месяцах)
                            months = list(range(-1, int(stress_test_result['recovery_months']) + 2))
                            values = []

                            # Добавляем начальное значение
                            values.append(custom_portfolio_value)

                            # Добавляем значение после шока
                            values.append(stress_test_result['portfolio_after_shock'])

                            # Добавляем значения восстановления (линейное приближение)
                            recovery_rate = (custom_portfolio_value - stress_test_result['portfolio_after_shock']) / \
                                            stress_test_result[
                                                'recovery_months'] if stress_test_result['recovery_months'] > 0 else 0

                            for i in range(1, len(months) - 1):
                                values.append(stress_test_result['portfolio_after_shock'] + recovery_rate * i)

                            fig_recovery = go.Figure()

                            fig_recovery.add_trace(go.Scatter(
                                x=months,
                                y=values,
                                mode='lines+markers',
                                name='Прогноз стоимости'
                            ))

                            fig_recovery.add_shape(
                                type="line",
                                x0=months[0],
                                y0=custom_portfolio_value,
                                x1=months[-1],
                                y1=custom_portfolio_value,
                                line=dict(color="green", width=2, dash="dot"),
                                name="Исходная стоимость"
                            )

                            fig_recovery.update_layout(
                                title="Прогноз восстановления портфеля",
                                xaxis_title="Месяцы",
                                yaxis_title="Стоимость портфеля ($)",
                                hovermode="x unified"
                            )

                            st.plotly_chart(fig_recovery, use_container_width=True)

                            # Добавляем информацию о заданных шоках
                            st.subheader("Параметры стресс-теста")

                            # Создаем DataFrame с заданными шоками для наглядности
                            shock_summary = []

                            # Добавляем рыночный шок
                            if 'market' in custom_shocks:
                                shock_summary.append({
                                    'Тип': 'Рыночный шок',
                                    'Объект': 'Весь рынок',
                                    'Заданный шок (%)': custom_shocks['market'] * 100
                                })

                            # Добавляем секторные шоки
                            for key, value in custom_shocks.items():
                                if key != 'market' and key != 'assets':
                                    shock_summary.append({
                                        'Тип': 'Секторный шок',
                                        'Объект': key,
                                        'Заданный шок (%)': value * 100
                                    })

                            # Добавляем шоки для отдельных активов
                            if 'assets' in custom_shocks:
                                for ticker, shock in custom_shocks['assets'].items():
                                    shock_summary.append({
                                        'Тип': 'Актив',
                                        'Объект': ticker,
                                        'Заданный шок (%)': shock * 100
                                    })

                            if shock_summary:
                                shock_df = pd.DataFrame(shock_summary)
                                st.dataframe(shock_df.style.format({
                                    'Заданный шок (%)': '{:.1f}%'
                                }), use_container_width=True)

                                st.info("""
                                **Как интерпретировать результаты:**

                                1. **Корреляции и беты**: В расчете учитываются исторические корреляции между активами и их беты относительно рынка, 
                                   что дает более реалистичную картину, чем просто суммирование индивидуальных шоков.

                                2. **Время восстановления**: Оценка основана на средней годовой доходности рынка 7%. Фактическое время восстановления 
                                   может отличаться в зависимости от рыночных условий.

                                3. **Влияние на активы**: Активы с высокой бетой и/или с сильной корреляцией с негативно шокированными секторами 
                                   будут испытывать более значительное влияние.
                                """)
                            else:
                                st.warning("Не заданы шоковые значения. Результаты могут быть неинформативными.")

        with stress_tabs[2]:
            st.subheader("Анализ чувствительности")

            st.write("""
            Анализ чувствительности показывает, как изменится стоимость портфеля при 
            изменении ключевых факторов риска.
            """)

            # Выбор факторов для анализа
            factors = st.multiselect(
                "Выберите факторы для анализа чувствительности",
                options=["Процентные ставки", "Инфляция", "Цены на нефть", "Курс доллара", "Рецессия"],
                default=["Процентные ставки", "Инфляция"]
            )

            if factors:
                # Создаем словарь факторов и их влияния на разные классы активов
                factor_impacts = {
                    "Процентные ставки": {
                        "Акции": -0.05,  # -5% при повышении ставок на 1%
                        "Облигации": -0.1,  # -10% при повышении ставок на 1%
                        "Недвижимость": -0.08,  # -8% при повышении ставок на 1%
                        "Золото": -0.03,  # -3% при повышении ставок на 1%
                        "Денежный рынок": 0.01  # +1% при повышении ставок на 1%
                    },
                    "Инфляция": {
                        "Акции": -0.02,  # -2% при росте инфляции на 1%
                        "Облигации": -0.05,  # -5% при росте инфляции на 1%
                        "Недвижимость": 0.02,  # +2% при росте инфляции на 1%
                        "Золото": 0.05,  # +5% при росте инфляции на 1%
                        "Денежный рынок": -0.01  # -1% при росте инфляции на 1%
                    },
                    "Цены на нефть": {
                        "Акции": 0.01,  # +1% при росте цен на нефть на 10%
                        "Облигации": -0.01,  # -1% при росте цен на нефть на 10%
                        "Недвижимость": 0.01,  # +1% при росте цен на нефть на 10%
                        "Золото": 0.02,  # +2% при росте цен на нефть на 10%
                        "Денежный рынок": 0  # 0% при росте цен на нефть на 10%
                    },
                    "Курс доллара": {
                        "Акции": -0.02,  # -2% при укреплении доллара на 5%
                        "Облигации": -0.01,  # -1% при укреплении доллара на 5%
                        "Недвижимость": -0.02,  # -2% при укреплении доллара на 5%
                        "Золото": -0.03,  # -3% при укреплении доллара на 5%
                        "Денежный рынок": 0.01  # +1% при укреплении доллара на 5%
                    },
                    "Рецессия": {
                        "Акции": -0.3,  # -30% при рецессии
                        "Облигации": -0.05,  # -5% при рецессии
                        "Недвижимость": -0.2,  # -20% при рецессии
                        "Золото": 0.1,  # +10% при рецессии
                        "Денежный рынок": 0.01  # +1% при рецессии
                    }
                }

                # Определяем класс актива для каждого тикера
                asset_classes = {}
                for asset in portfolio_data['assets']:
                    ticker = asset['ticker']
                    if 'asset_class' in asset and asset['asset_class'] != 'N/A':
                        class_name = asset['asset_class']
                    elif 'sector' in asset and asset['sector'] in ['Financial Services', 'Financial', 'Financials']:
                        class_name = "Акции"  # Если сектор - финансы, то это скорее всего акции
                    else:
                        # По умолчанию считаем, что это акции
                        class_name = "Акции"

                    asset_classes[ticker] = class_name

                # Создаем таблицу результатов чувствительности
                sensitivity_results = []

                for factor in factors:
                    # Для каждого фактора определяем диапазон изменений
                    if factor == "Процентные ставки":
                        changes = [0.25, 0.5, 1.0, 1.5, 2.0]  # Изменения в процентных пунктах
                        change_label = "п.п."
                    elif factor == "Инфляция":
                        changes = [1.0, 2.0, 3.0, 4.0, 5.0]  # Изменения в процентных пунктах
                        change_label = "п.п."
                    elif factor == "Цены на нефть":
                        changes = [10, 20, 30, 40, 50]  # Изменения в процентах
                        change_label = "%"
                    elif factor == "Курс доллара":
                        changes = [5, 10, 15, 20, 25]  # Изменения в процентах
                        change_label = "%"
                    else:  # Рецессия - бинарное событие
                        changes = [1]  # Просто наступление рецессии
                        change_label = ""

                    # Рассчитываем влияние каждого изменения фактора
                    for change in changes:
                        portfolio_impact = 0

                        for ticker, weight in weights.items():
                            asset_class = asset_classes.get(ticker, "Акции")

                            # Преобразуем класс актива в ключ из factor_impacts
                            if asset_class in ["Equity", "Stock", "Stocks"]:
                                asset_class = "Акции"
                            elif asset_class in ["Bond", "Bonds", "Fixed Income"]:
                                asset_class = "Облигации"
                            elif asset_class in ["Real Estate", "REIT"]:
                                asset_class = "Недвижимость"
                            elif asset_class in ["Gold", "Precious Metals"]:
                                asset_class = "Золото"
                            elif asset_class in ["Cash", "Money Market"]:
                                asset_class = "Денежный рынок"

                            # Находим соответствующий коэффициент влияния
                            impact_coef = factor_impacts[factor].get(asset_class, 0)

                            # Рассчитываем влияние на актив
                            asset_impact = impact_coef * change

                            # Добавляем взвешенное влияние к общему влиянию на портфель
                            portfolio_impact += asset_impact * weight

                        sensitivity_results.append({
                            'Фактор': factor,
                            'Изменение': f"+{change} {change_label}" if change_label else "Наступление",
                            'Влияние на портфель (%)': portfolio_impact * 100
                        })

                # Создаем DataFrame и сортируем по влиянию
                sensitivity_df = pd.DataFrame(sensitivity_results)
                sensitivity_df = sensitivity_df.sort_values('Влияние на портфель (%)', ascending=True)

                # Визуализация результатов чувствительности
                fig_sensitivity = px.bar(
                    sensitivity_df,
                    x='Влияние на портфель (%)',
                    y='Фактор',
                    color='Влияние на портфель (%)',
                    color_continuous_scale='RdYlGn',
                    text='Изменение',
                    orientation='h',
                    title='Чувствительность портфеля к факторам риска'
                )

                fig_sensitivity.update_layout(
                    xaxis_title='Влияние на доходность портфеля (%)',
                    yaxis_title='Фактор риска',
                    height=500
                )

                st.plotly_chart(fig_sensitivity, use_container_width=True)

                # Отображаем таблицу с результатами
                st.dataframe(sensitivity_df.style.format({
                    'Влияние на портфель (%)': '{:.2f}%'
                }), use_container_width=True)

        with stress_tabs[3]:
            st.subheader("Экстремальные сценарии")

            st.write("""
            Анализ экстремальных сценариев оценивает влияние маловероятных, но возможных 
            событий на стоимость портфеля, используя статистическое моделирование.
            """)

            # Ввод стоимости портфеля
            extreme_portfolio_value = st.number_input(
                "Стоимость портфеля ($)",
                min_value=1000,
                value=10000,
                step=1000,
                key="extreme_portfolio_value"
            )

            # Определяем экстремальные сценарии с более подробным описанием
            extreme_scenarios = {
                "market_crash_50": {
                    "name": "Рыночный крах (-50%)",
                    "description": "Крупномасштабный крах рынка, аналогичный кризису 2008 года, ведущий к падению основных индексов на 50%.",
                    "impact": -0.5,
                    "details": "Аналогично периоду сентябрь 2008 - март 2009, когда S&P 500 потерял более 50%. События такой серьезности случаются примерно раз в 50-80 лет."
                },
                "severe_recession_35": {
                    "name": "Сильная рецессия (-35%)",
                    "description": "Тяжелая экономическая рецессия с длительным сокращением ВВП и высокой безработицей.",
                    "impact": -0.35,
                    "details": "Похоже на начало пандемии COVID-19 (февраль-март 2020), когда рынки упали примерно на 35%. Вероятность таких событий - примерно раз в 10-15 лет."
                },
                "inflation_shock_25": {
                    "name": "Инфляционный шок (+8%)",
                    "description": "Резкий скачок инфляции выше 8%, вынуждающий центральные банки агрессивно повышать процентные ставки.",
                    "impact": -0.25,
                    "details": "Исторические примеры: инфляционный шок 1970-х годов и период 2021-2022 гг. Особенно негативно влияет на долгосрочные облигации и акции роста."
                },
                "geopolitical_crisis_20": {
                    "name": "Геополитический кризис",
                    "description": "Серьезный международный конфликт или кризис, влияющий на глобальную торговлю и энергетические рынки.",
                    "impact": -0.20,
                    "details": "Например, нефтяной кризис 1973 года или напряженность между крупными державами. Часто приводит к росту цен на сырьевые товары и золото при падении большинства акций."
                },
                "tech_bubble_burst_40": {
                    "name": "Лопнувший технологический пузырь",
                    "description": "Резкая коррекция переоцененных технологических акций, подобная краху доткомов 2000 года.",
                    "impact": -0.40,
                    "details": "Технологические и акции роста могут потерять 60-80% стоимости, в то время как стоимостные акции и защитные секторы будут чувствовать себя лучше. Вероятность - примерно раз в 20-25 лет."
                },
                "currency_crisis_15": {
                    "name": "Валютный кризис",
                    "description": "Сильная девальвация одной или нескольких мировых валют, вызывающая цепную реакцию на рынках.",
                    "impact": -0.15,
                    "details": "Примеры включают Азиатский финансовый кризис 1997 года или Европейский валютный кризис 1992 года. Обычно имеет региональный характер, но может распространиться глобально."
                }
            }

            # Выбор сценариев для анализа
            selected_extreme_scenarios = st.multiselect(
                "Выберите экстремальные сценарии для анализа",
                options=list(extreme_scenarios.keys()),
                default=list(extreme_scenarios.keys())[:3],
                format_func=lambda x: extreme_scenarios[x]["name"]
            )

            # Опциональная настройка параметров моделирования
            with st.expander("Параметры моделирования"):
                col1, col2 = st.columns(2)

                with col1:
                    confidence_level = st.slider(
                        "Уровень доверия моделирования (%)",
                        min_value=80,
                        max_value=99,
                        value=95,
                        format="%d%%"
                    )

                    monte_carlo_sims = st.slider(
                        "Количество симуляций Монте-Карло",
                        min_value=100,
                        max_value=5000,
                        value=1000,
                        step=100
                    )

                with col2:
                    recovery_annual_return = st.slider(
                        "Ожидаемая годовая доходность для восстановления (%)",
                        min_value=3,
                        max_value=12,
                        value=7,
                        step=1
                    ) / 100

                    fat_tail_factor = st.slider(
                        "Коэффициент тяжелых хвостов",
                        min_value=1.0,
                        max_value=3.0,
                        value=1.5,
                        step=0.1,
                        help="Увеличивает вероятность экстремальных событий. Значение 1.0 соответствует нормальному распределению."
                    )

            if selected_extreme_scenarios:
                # Загрузка исторических данных для более реалистичного моделирования
                if st.button("Провести анализ экстремальных сценариев"):
                    with st.spinner("Выполнение анализа экстремальных сценариев..."):
                        # Загружаем исторические данные
                        start_date = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')  # 5 лет истории
                        end_date = datetime.now().strftime('%Y-%m-%d')

                        # Получаем список тикеров и весов
                        tickers = [asset['ticker'] for asset in portfolio_data['assets']]
                        weights = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

                        # Добавляем SPY для использования как рыночный индикатор
                        if 'SPY' not in tickers:
                            tickers.append('SPY')

                        # Загружаем исторические данные
                        prices_data = data_fetcher.get_batch_data(tickers, start_date, end_date)

                        # Проверяем, что данные загружены успешно
                        if not prices_data or all(df.empty for df in prices_data.values()):
                            st.error(
                                "Не удалось загрузить исторические данные. Пожалуйста, проверьте тикеры или измените период.")
                        else:
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
                            portfolio_returns = PortfolioAnalytics.calculate_portfolio_return(returns, weights)

                            # Создаем DataFrame для отображения результатов
                            extreme_results = []

                            for scenario_key in selected_extreme_scenarios:
                                scenario = extreme_scenarios[scenario_key]

                                # Основные данные о сценарии
                                impact = scenario["impact"]
                                portfolio_loss = extreme_portfolio_value * impact
                                portfolio_after_shock = extreme_portfolio_value + portfolio_loss

                                # Расчет VaR и CVaR на основе исторических данных для более реалистичной оценки
                                if not portfolio_returns.empty:
                                    # Моделируем хвосты распределения с учетом коэффициента тяжелых хвостов
                                    # и используем исторические данные для более реалистичного результата
                                    hist_std = portfolio_returns.std()

                                    # Генерируем случайные возвраты с тяжелыми хвостами
                                    np.random.seed(42)  # Для воспроизводимости
                                    sim_returns = []

                                    # Генерируем симуляции с более тяжелыми хвостами для реалистичности
                                    for _ in range(monte_carlo_sims):
                                        # Используем t-распределение для моделирования тяжелых хвостов
                                        # Меньшее число степеней свободы дает более тяжелые хвосты
                                        degrees_of_freedom = 4  # Чем меньше, тем тяжелее хвосты
                                        t_random = np.random.standard_t(degrees_of_freedom)

                                        # Масштабируем t-распределение к нужной волатильности
                                        shock = t_random * hist_std * fat_tail_factor * (
                                                    -impact / 0.5)  # Нормализуем к impact
                                        sim_returns.append(shock)

                                    # Рассчитываем VaR как квантиль симулированных возвратов
                                    var_level = np.percentile(sim_returns, 100 - confidence_level)

                                    # Рассчитываем CVaR как среднее значений хуже VaR
                                    cvar_values = [r for r in sim_returns if r <= var_level]
                                    cvar = np.mean(cvar_values) if cvar_values else var_level

                                    # Преобразуем в денежные значения
                                    var_amount = extreme_portfolio_value * var_level
                                    cvar_amount = extreme_portfolio_value * cvar
                                else:
                                    # Если нет исторических данных, используем приближения
                                    var_amount = portfolio_loss * 1.2  # Примерно на 20% хуже среднего сценария
                                    cvar_amount = portfolio_loss * 1.4  # Примерно на 40% хуже среднего сценария

                                # Рассчитываем восстановление (примерное)
                                daily_return = (1 + recovery_annual_return) ** (1 / 252) - 1

                                if impact < 0:
                                    # Рассчитываем количество дней для восстановления
                                    recovery_days = -np.log(1 + impact) / np.log(1 + daily_return)
                                    recovery_months = recovery_days / 21  # примерно 21 торговый день в месяце
                                else:
                                    recovery_days = 0
                                    recovery_months = 0

                                # Добавляем сценарий в результаты
                                extreme_results.append({
                                    'Сценарий': scenario["name"],
                                    'Описание': scenario["description"],
                                    'Шок (%)': impact * 100,
                                    'Потеря ($)': portfolio_loss,
                                    'Стоимость после шока ($)': portfolio_after_shock,
                                    'VaR при {0}% ($)'.format(confidence_level): var_amount,
                                    'CVaR при {0}% ($)'.format(confidence_level): cvar_amount,
                                    'Восстановление (мес.)': recovery_months,
                                    'key': scenario_key,
                                    'details': scenario["details"]
                                })

                            # Отображаем результаты
                            st.subheader("Результаты анализа экстремальных сценариев")

                            # Создаем DataFrame для визуализации
                            result_df = pd.DataFrame(extreme_results)
                            result_df_display = result_df[[
                                'Сценарий', 'Шок (%)', 'Потеря ($)', 'Стоимость после шока ($)', 'Восстановление (мес.)'
                            ]]

                            # Отображаем результаты как таблицу
                            st.dataframe(result_df_display.style.format({
                                'Шок (%)': '{:.1f}%',
                                'Потеря ($)': '${:.2f}',
                                'Стоимость после шока ($)': '${:.2f}',
                                'Восстановление (мес.)': '{:.1f}'
                            }), use_container_width=True)

                            # Визуализация потерь по сценариям
                            fig_extreme = px.bar(
                                result_df,
                                x='Сценарий',
                                y='Потеря ($)',
                                color='Шок (%)',
                                color_continuous_scale='RdYlGn',
                                title='Влияние экстремальных сценариев на портфель'
                            )

                            fig_extreme.update_layout(
                                xaxis_title='Сценарий',
                                yaxis_title='Потеря стоимости ($)',
                                height=500
                            )

                            st.plotly_chart(fig_extreme, use_container_width=True)

                            # Детальная информация о каждом сценарии
                            st.subheader("Детальное описание сценариев")

                            for scenario_key in selected_extreme_scenarios:
                                scenario = extreme_scenarios[scenario_key]
                                scenario_result = next((r for r in extreme_results if r['key'] == scenario_key), None)

                                if scenario_result:
                                    with st.expander(f"{scenario['name']} - Подробности"):
                                        st.write(f"**Описание**: {scenario['description']}")
                                        st.write(f"**Детали**: {scenario['details']}")

                                        col1, col2, col3 = st.columns(3)

                                        with col1:
                                            st.metric(
                                                "Прогнозируемая потеря",
                                                f"${abs(scenario_result['Потеря ($)']):.2f}",
                                                f"{scenario_result['Шок (%)']}%",
                                                delta_color="inverse"
                                            )

                                        with col2:
                                            st.metric(
                                                f"VaR при {confidence_level}%",
                                                f"${abs(scenario_result['VaR при {0}% ($)'.format(confidence_level)]):.2f}"
                                            )

                                        with col3:
                                            st.metric(
                                                "Время восстановления",
                                                f"{scenario_result['Восстановление (мес.)']:.1f} мес."
                                            )

                            # Дополнительная информация о распределении потенциальных потерь
                            st.subheader("Распределение потенциальных потерь")

                            # Визуализация распределения потерь через симуляцию
                            sim_losses = []
                            for _ in range(monte_carlo_sims):
                                # Используем t-распределение для тяжелых хвостов
                                t_random = np.random.standard_t(4)  # 4 степени свободы
                                # Масштабируем к волатильности портфеля
                                portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Годовая волатильность
                                # Симулируем экстремальные потери
                                extreme_loss = -abs(
                                    t_random) * portfolio_volatility * fat_tail_factor * extreme_portfolio_value * 0.2
                                sim_losses.append(extreme_loss)

                            # Создаем гистограмму потерь
                            fig_dist = px.histogram(
                                sim_losses,
                                nbins=50,
                                title='Распределение потенциальных экстремальных потерь',
                                labels={'value': 'Потеря ($)', 'count': 'Частота'},
                                color_discrete_sequence=['rgba(255, 0, 0, 0.6)']
                            )

                            # Добавляем вертикальные линии для потерь по сценариям
                            for scenario_result in extreme_results:
                                fig_dist.add_vline(
                                    x=scenario_result['Потеря ($)'],
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=scenario_result['Сценарий'],
                                    annotation_position="top right"
                                )

                            fig_dist.update_layout(
                                xaxis_title='Потенциальная потеря ($)',
                                yaxis_title='Частота',
                                showlegend=False
                            )

                            st.plotly_chart(fig_dist, use_container_width=True)

                            # Добавление предупреждения об интерпретации
                            st.info("""
                            **Важное замечание о моделировании экстремальных событий:**

                            Экстремальные сценарии представляют собой редкие и маловероятные события, которые сложно точно смоделировать.
                            Реальные потери могут отличаться от прогнозируемых. Данный анализ следует рассматривать как иллюстративный,
                            а не как точный прогноз. Кроме того, влияние экстремальных событий на разные классы активов может существенно
                            отличаться и меняться со временем.
                            """)

    # В portfolio_analysis.py
    with tabs[6]:  # Вкладка "Скользящие метрики"
        st.subheader("Скользящие метрики")

        # Создаем подвкладки для разных видов скользящих метрик
        rolling_tabs = st.tabs([
            "Волатильность",
            "Коэффициенты",
            "Скользящая бета/альфа",
            "Раздельный анализ"
        ])

        with rolling_tabs[0]:
            st.subheader("Скользящая волатильность")

            # Параметры расчета
            window_size = st.slider(
                "Размер окна (дней) для волатильности",
                min_value=21,
                max_value=252,
                value=63,
                step=21
            )

            # Расчет скользящей волатильности
            if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                rolling_vol = portfolio_returns.rolling(window=window_size).std() * np.sqrt(252) * 100

                if benchmark_returns is not None:
                    benchmark_rolling_vol = benchmark_returns.rolling(window=window_size).std() * np.sqrt(252) * 100

                    # Визуализация скользящей волатильности
                    fig_rolling_vol = go.Figure()

                    fig_rolling_vol.add_trace(go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol.values,
                        mode='lines',
                        name='Портфель',
                        line=dict(color='blue', width=2)
                    ))

                    fig_rolling_vol.add_trace(go.Scatter(
                        x=benchmark_rolling_vol.index,
                        y=benchmark_rolling_vol.values,
                        mode='lines',
                        name=benchmark,
                        line=dict(color='orange', width=2, dash='dash')
                    ))

                    fig_rolling_vol.update_layout(
                        title=f'Скользящая волатильность ({window_size} дней)',
                        xaxis_title='Дата',
                        yaxis_title='Волатильность (%)',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig_rolling_vol, use_container_width=True)

                    # Статистика волатильности
                    st.subheader("Статистика волатильности")

                    vol_stats = pd.DataFrame({
                        'Метрика': ['Средняя волатильность (%)', 'Медианная волатильность (%)',
                                    'Мин. волатильность (%)', 'Макс. волатильность (%)'],
                        'Портфель': [
                            rolling_vol.mean(),
                            rolling_vol.median(),
                            rolling_vol.min(),
                            rolling_vol.max()
                        ],
                        'Бенчмарк': [
                            benchmark_rolling_vol.mean(),
                            benchmark_rolling_vol.median(),
                            benchmark_rolling_vol.min(),
                            benchmark_rolling_vol.max()
                        ]
                    })

                    st.dataframe(vol_stats.style.format({
                        'Портфель': '{:.2f}',
                        'Бенчмарк': '{:.2f}'
                    }), use_container_width=True)

        with rolling_tabs[1]:
            st.subheader("Скользящие коэффициенты риска")

            # Параметры расчета
            coef_window_size = st.slider(
                "Размер окна (дней) для коэффициентов",
                min_value=63,
                max_value=252,
                value=126,
                step=21
            )

            if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                # Создаем функцию для расчета Sharpe и Sortino в скользящем окне
                def rolling_sharpe_sortino(returns, window, risk_free=0):
                    rolling_sharpe = []
                    rolling_sortino = []

                    for i in range(window, len(returns) + 1):
                        window_returns = returns.iloc[i - window:i]

                        # Sharpe
                        excess_returns = window_returns - risk_free / 252  # дневной risk-free
                        sharpe = excess_returns.mean() / window_returns.std() * np.sqrt(252)
                        rolling_sharpe.append(sharpe)

                        # Sortino
                        negative_returns = window_returns[window_returns < risk_free / 252]
                        if len(negative_returns) > 0:
                            downside_deviation = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(252)
                            sortino = excess_returns.mean() * 252 / downside_deviation if downside_deviation > 0 else np.nan
                        else:
                            sortino = np.nan

                        rolling_sortino.append(sortino)

                    return pd.Series(rolling_sharpe, index=returns.index[window - 1:]), pd.Series(rolling_sortino,
                                                                                                  index=returns.index[
                                                                                                        window - 1:])

                # Рассчитываем скользящие коэффициенты
                rolling_sharpe, rolling_sortino = rolling_sharpe_sortino(
                    portfolio_returns, coef_window_size, config.RISK_FREE_RATE
                )

                # Рассчитываем для бенчмарка, если он доступен
                if benchmark_returns is not None:
                    benchmark_rolling_sharpe, benchmark_rolling_sortino = rolling_sharpe_sortino(
                        benchmark_returns, coef_window_size, config.RISK_FREE_RATE
                    )

                    # Визуализация скользящего коэффициента Шарпа
                    fig_rolling_sharpe = go.Figure()

                    fig_rolling_sharpe.add_trace(go.Scatter(
                        x=rolling_sharpe.index,
                        y=rolling_sharpe.values,
                        mode='lines',
                        name='Портфель',
                        line=dict(color='blue', width=2)
                    ))

                    fig_rolling_sharpe.add_trace(go.Scatter(
                        x=benchmark_rolling_sharpe.index,
                        y=benchmark_rolling_sharpe.values,
                        mode='lines',
                        name=benchmark,
                        line=dict(color='orange', width=2, dash='dash')
                    ))

                    # Добавляем линию нулевого Шарпа
                    fig_rolling_sharpe.add_hline(y=0, line_dash="dash", line_color="red",
                                                 annotation_text="Нулевой Шарп")

                    fig_rolling_sharpe.update_layout(
                        title=f'Скользящий коэффициент Шарпа ({coef_window_size} дней)',
                        xaxis_title='Дата',
                        yaxis_title='Коэффициент Шарпа',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig_rolling_sharpe, use_container_width=True)

                    # Визуализация скользящего коэффициента Сортино
                    fig_rolling_sortino = go.Figure()

                    fig_rolling_sortino.add_trace(go.Scatter(
                        x=rolling_sortino.index,
                        y=rolling_sortino.values,
                        mode='lines',
                        name='Портфель',
                        line=dict(color='blue', width=2)
                    ))

                    fig_rolling_sortino.add_trace(go.Scatter(
                        x=benchmark_rolling_sortino.index,
                        y=benchmark_rolling_sortino.values,
                        mode='lines',
                        name=benchmark,
                        line=dict(color='orange', width=2, dash='dash')
                    ))

                    # Добавляем линию нулевого Сортино
                    fig_rolling_sortino.add_hline(y=0, line_dash="dash", line_color="red",
                                                  annotation_text="Нулевой Сортино")

                    fig_rolling_sortino.update_layout(
                        title=f'Скользящий коэффициент Сортино ({coef_window_size} дней)',
                        xaxis_title='Дата',
                        yaxis_title='Коэффициент Сортино',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig_rolling_sortino, use_container_width=True)

        with rolling_tabs[2]:
            st.subheader("Скользящая бета и альфа")

            # Параметры расчета
            beta_window_size = st.slider(
                "Размер окна (дней) для беты и альфы",
                min_value=63,
                max_value=252,
                value=126,
                step=21,
                key="beta_window_size"
            )

            if isinstance(portfolio_returns.index, pd.DatetimeIndex) and benchmark_returns is not None:
                # Рассчитываем скользящую бету
                rolling_cov = portfolio_returns.rolling(window=beta_window_size).cov(benchmark_returns)
                rolling_var = benchmark_returns.rolling(window=beta_window_size).var()
                rolling_beta = rolling_cov / rolling_var

                # Рассчитываем скользящую альфу
                rolling_portfolio_return = portfolio_returns.rolling(window=beta_window_size).mean() * 252
                rolling_benchmark_return = benchmark_returns.rolling(window=beta_window_size).mean() * 252
                rolling_alpha = rolling_portfolio_return - (
                            config.RISK_FREE_RATE + rolling_beta * (rolling_benchmark_return - config.RISK_FREE_RATE))

                # Визуализация скользящей беты
                fig_rolling_beta = go.Figure()

                fig_rolling_beta.add_trace(go.Scatter(
                    x=rolling_beta.index,
                    y=rolling_beta.values,
                    mode='lines',
                    name='Бета',
                    line=dict(color='purple', width=2)
                ))

                # Добавляем линии для уровней беты
                fig_rolling_beta.add_hline(y=1, line_dash="dash", line_color="black",
                                           annotation_text="Бета = 1")
                fig_rolling_beta.add_hline(y=0, line_dash="dash", line_color="green",
                                           annotation_text="Бета = 0")

                fig_rolling_beta.update_layout(
                    title=f'Скользящая бета ({beta_window_size} дней)',
                    xaxis_title='Дата',
                    yaxis_title='Бета',
                    hovermode='x unified'
                )

                st.plotly_chart(fig_rolling_beta, use_container_width=True)

                # Визуализация скользящей альфы
                fig_rolling_alpha = go.Figure()

                fig_rolling_alpha.add_trace(go.Scatter(
                    x=rolling_alpha.index,
                    y=rolling_alpha.values * 100,  # переводим в проценты
                    mode='lines',
                    name='Альфа',
                    line=dict(color='green', width=2)
                ))

                # Добавляем линию нулевой альфы
                fig_rolling_alpha.add_hline(y=0, line_dash="dash", line_color="red",
                                            annotation_text="Альфа = 0")

                fig_rolling_alpha.update_layout(
                    title=f'Скользящая альфа ({beta_window_size} дней)',
                    xaxis_title='Дата',
                    yaxis_title='Альфа (%)',
                    hovermode='x unified'
                )

                st.plotly_chart(fig_rolling_alpha, use_container_width=True)

        with rolling_tabs[3]:
            st.subheader("Раздельный анализ бычьих и медвежьих периодов")

            if isinstance(portfolio_returns.index, pd.DatetimeIndex) and benchmark_returns is not None:
                # Создаем маски для бычьих и медвежьих периодов
                bull_market = benchmark_returns > 0
                bear_market = benchmark_returns < 0

                # Рассчитываем доходность в бычьи и медвежьи периоды
                bull_portfolio_return = portfolio_returns[bull_market].mean() * 252 * 100
                bear_portfolio_return = portfolio_returns[bear_market].mean() * 252 * 100

                bull_benchmark_return = benchmark_returns[bull_market].mean() * 252 * 100
                bear_benchmark_return = benchmark_returns[bear_market].mean() * 252 * 100

                # Рассчитываем беты в разные периоды
                if bull_market.sum() > 0:
                    bull_beta = portfolio_returns[bull_market].cov(benchmark_returns[bull_market]) / benchmark_returns[
                        bull_market].var()
                else:
                    bull_beta = 0

                if bear_market.sum() > 0:
                    bear_beta = portfolio_returns[bear_market].cov(benchmark_returns[bear_market]) / benchmark_returns[
                        bear_market].var()
                else:
                    bear_beta = 0

                # Создаем DataFrame для отображения
                market_conditions_df = pd.DataFrame({
                    'Метрика': ['Доходность портфеля (%)', 'Доходность бенчмарка (%)', 'Бета', 'Разница (%)'],
                    'Растущий рынок': [
                        bull_portfolio_return,
                        bull_benchmark_return,
                        bull_beta,
                        bull_portfolio_return - bull_benchmark_return
                    ],
                    'Падающий рынок': [
                        bear_portfolio_return,
                        bear_benchmark_return,
                        bear_beta,
                        bear_portfolio_return - bear_benchmark_return
                    ]
                })

                # Отображаем таблицу
                st.dataframe(market_conditions_df.style.format({
                    'Растущий рынок': '{:.2f}',
                    'Падающий рынок': '{:.2f}'
                }), use_container_width=True)

                # Визуализация доходности в разные периоды
                fig_market_conditions = go.Figure()

                fig_market_conditions.add_trace(go.Bar(
                    x=['Растущий рынок', 'Падающий рынок'],
                    y=[bull_portfolio_return, bear_portfolio_return],
                    name='Портфель',
                    marker_color='blue'
                ))

                fig_market_conditions.add_trace(go.Bar(
                    x=['Растущий рынок', 'Падающий рынок'],
                    y=[bull_benchmark_return, bear_benchmark_return],
                    name=benchmark,
                    marker_color='orange'
                ))

                fig_market_conditions.update_layout(
                    title='Сравнение доходности в разные периоды рынка',
                    xaxis_title='Состояние рынка',
                    yaxis_title='Годовая доходность (%)',
                    barmode='group',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig_market_conditions, use_container_width=True)

                # Скользящая производительность в разные периоды
                bull_window_size = st.slider(
                    "Размер окна (дней) для анализа периодов",
                    min_value=63,
                    max_value=252,
                    value=126,
                    step=21,
                    key="bull_window_size"
                )

                # Рассчитываем скользящую бету в бычьи/медвежьи периоды
                # Для каждого окна определяем, преобладает ли бычий или медвежий рынок
                rolling_bull_beta = []
                rolling_bear_beta = []
                dates = []

                for i in range(bull_window_size, len(benchmark_returns)):
                    window_benchmark = benchmark_returns.iloc[i - bull_window_size:i]
                    window_portfolio = portfolio_returns.iloc[i - bull_window_size:i]
                    window_date = benchmark_returns.index[i]

                    # Определяем бычий/медвежий периоды в окне
                    window_bull = window_benchmark > 0
                    window_bear = window_benchmark < 0

                    # Рассчитываем беты, если есть достаточно данных
                    if window_bull.sum() > 10:  # Минимум 10 дней
                        bull_beta_val = window_portfolio[window_bull].cov(window_benchmark[window_bull]) / \
                                        window_benchmark[window_bull].var()
                        rolling_bull_beta.append(bull_beta_val)
                    else:
                        rolling_bull_beta.append(np.nan)

                    if window_bear.sum() > 10:  # Минимум 10 дней
                        bear_beta_val = window_portfolio[window_bear].cov(window_benchmark[window_bear]) / \
                                        window_benchmark[window_bear].var()
                        rolling_bear_beta.append(bear_beta_val)
                    else:
                        rolling_bear_beta.append(np.nan)

                    dates.append(window_date)

                # Создаем Series из рассчитанных значений
                rolling_bull_beta_series = pd.Series(rolling_bull_beta, index=dates)
                rolling_bear_beta_series = pd.Series(rolling_bear_beta, index=dates)

                # Визуализация скользящих бет
                fig_bull_bear_beta = go.Figure()

                fig_bull_bear_beta.add_trace(go.Scatter(
                    x=rolling_bull_beta_series.index,
                    y=rolling_bull_beta_series.values,
                    mode='lines',
                    name='Бета в растущий рынок',
                    line=dict(color='green', width=2)
                ))

                fig_bull_bear_beta.add_trace(go.Scatter(
                    x=rolling_bear_beta_series.index,
                    y=rolling_bear_beta_series.values,
                    mode='lines',
                    name='Бета в падающий рынок',
                    line=dict(color='red', width=2)
                ))

                # Добавляем линию единичной беты
                fig_bull_bear_beta.add_hline(y=1, line_dash="dash", line_color="black",
                                             annotation_text="Бета = 1")

                fig_bull_bear_beta.update_layout(
                    title=f'Скользящие беты в разные периоды рынка ({bull_window_size} дней)',
                    xaxis_title='Дата',
                    yaxis_title='Бета',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode='x unified'
                )

                st.plotly_chart(fig_bull_bear_beta, use_container_width=True)

    with tabs[7]:  # Вкладка "Расширенный анализ"
        st.subheader("Расширенный анализ")

        # Создаем подвкладки для разных видов расширенного анализа
        advanced_tabs = st.tabs([
            "Календарь доходности",
            "Сезонный анализ",
            "Квантили распределения",
            "Множественные метрики"
        ])

        with advanced_tabs[0]:
            st.subheader("Календарь месячных доходностей")

            # Создаем календарь месячных доходностей
            if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

                # Создаем сводную таблицу
                monthly_df = pd.DataFrame({
                    'year': monthly_returns.index.year,
                    'month': monthly_returns.index.month,
                    'return': monthly_returns.values * 100  # В процентах
                })

                # Создаем сводную таблицу с годами по строкам и месяцами по столбцам
                heatmap_data = monthly_df.pivot(index='year', columns='month', values='return')

                # Переименовываем столбцы в названия месяцев
                month_names = {
                    1: 'Янв', 2: 'Фев', 3: 'Мар', 4: 'Апр', 5: 'Май', 6: 'Июн',
                    7: 'Июл', 8: 'Авг', 9: 'Сен', 10: 'Окт', 11: 'Ноя', 12: 'Дек'
                }
                heatmap_data = heatmap_data.rename(columns=month_names)

                # Визуализируем тепловую карту
                fig_heatmap = px.imshow(
                    heatmap_data,
                    labels=dict(x="Месяц", y="Год", color="Доходность (%)"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale='RdYlGn',
                    aspect="auto",
                    text_auto='.1f'
                )

                fig_heatmap.update_layout(
                    title='Календарь месячных доходностей (%)'
                )

                st.plotly_chart(fig_heatmap, use_container_width=True)

                # Добавляем столбец с годовой доходностью
                if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                    annual_returns = portfolio_returns.resample('A').apply(
                        lambda x: (1 + x).prod() - 1
                    ) * 100

                    annual_returns.index = annual_returns.index.year

                    # Проверяем, есть ли годы в heatmap_data
                    years_in_heatmap = heatmap_data.index.tolist()
                    annual_returns = annual_returns[annual_returns.index.isin(years_in_heatmap)]

                    if not annual_returns.empty:
                        # Добавляем столбец с годовой доходностью
                        heatmap_data['Год'] = annual_returns.values

                        # Отображаем обновленную таблицу
                        st.dataframe(heatmap_data.style.format('{:.2f}%').background_gradient(
                            cmap='RdYlGn', axis=None
                        ), use_container_width=True)

        with advanced_tabs[1]:
            st.subheader("Сезонный анализ")

            if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                # Создаем копию с добавленными колонками для дня недели и месяца
                seasonal_returns = portfolio_returns.copy()
                seasonal_returns = pd.DataFrame(seasonal_returns)
                seasonal_returns.columns = ['returns']

                # Добавляем информацию о дне недели и месяце
                seasonal_returns['day_of_week'] = seasonal_returns.index.day_name()
                seasonal_returns['month'] = seasonal_returns.index.month_name()
                seasonal_returns['year'] = seasonal_returns.index.year
                seasonal_returns['quarter'] = seasonal_returns.index.quarter

                col1, col2 = st.columns(2)

                with col1:
                    # Анализ по дням недели
                    day_of_week_returns = seasonal_returns.groupby('day_of_week')['returns'].mean() * 100

                    # Переставляем дни недели в правильном порядке
                    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                    day_of_week_returns = day_of_week_returns.reindex(days_order)

                    fig_days = px.bar(
                        x=day_of_week_returns.index,
                        y=day_of_week_returns.values,
                        title='Средняя доходность по дням недели (%)',
                        labels={'x': 'День недели', 'y': 'Средняя доходность (%)'},
                        color=day_of_week_returns.values,
                        color_continuous_scale='RdYlGn'
                    )

                    st.plotly_chart(fig_days, use_container_width=True)

                with col2:
                    # Анализ по месяцам
                    month_returns = seasonal_returns.groupby('month')['returns'].mean() * 100

                    # Переставляем месяцы в правильном порядке
                    months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                                    'July', 'August', 'September', 'October', 'November', 'December']
                    month_returns = month_returns.reindex(months_order)

                    fig_months = px.bar(
                        x=month_returns.index,
                        y=month_returns.values,
                        title='Средняя доходность по месяцам (%)',
                        labels={'x': 'Месяц', 'y': 'Средняя доходность (%)'},
                        color=month_returns.values,
                        color_continuous_scale='RdYlGn'
                    )

                    st.plotly_chart(fig_months, use_container_width=True)

                    # Анализ по кварталам
                quarter_returns = seasonal_returns.groupby('quarter')['returns'].mean() * 100

                fig_quarters = px.bar(
                    x=quarter_returns.index,
                    y=quarter_returns.values,
                    title='Средняя доходность по кварталам (%)',
                    labels={'x': 'Квартал', 'y': 'Средняя доходность (%)'},
                    color=quarter_returns.values,
                    color_continuous_scale='RdYlGn'
                )

                st.plotly_chart(fig_quarters, use_container_width=True)

            with advanced_tabs[2]:
                st.subheader("Квантили распределения доходности")

                # Рассчитываем квантили дневной доходности
                daily_quantiles = portfolio_returns.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]) * 100

                # Создаем DataFrame для отображения
                quantiles_df = pd.DataFrame({
                    'Квантиль': ['1%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '99%'],
                    'Дневная доходность (%)': daily_quantiles.values.round(2)
                })

                st.dataframe(quantiles_df, use_container_width=True)

                # Визуализируем квантили
                fig_quantiles = go.Figure()

                # Добавляем боксплот
                fig_quantiles.add_trace(go.Box(
                    y=portfolio_returns * 100,
                    name='Портфель',
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8,
                    marker=dict(color='blue')
                ))

                # Если есть бенчмарк, добавляем его квантили
                if benchmark_returns is not None:
                    fig_quantiles.add_trace(go.Box(
                        y=benchmark_returns * 100,
                        name=benchmark,
                        boxpoints='outliers',
                        jitter=0.3,
                        pointpos=-1.8,
                        marker=dict(color='orange')
                    ))

                fig_quantiles.update_layout(
                    title='Распределение дневной доходности (%)',
                    yaxis_title='Доходность (%)',
                    boxmode='group',
                    showlegend=True
                )

                st.plotly_chart(fig_quantiles, use_container_width=True)

                # Сравнение с нормальным распределением
                st.subheader("Сравнение с нормальным распределением")

                # Q-Q plot
                fig_qq = go.Figure()

                # Сортируем и нормализуем доходности
                sorted_returns = sorted(portfolio_returns * 100)
                n = len(sorted_returns)

                # Квантили нормального распределения
                from scipy import stats
                theoretical_quantiles = [stats.norm.ppf((i + 0.5) / n) for i in range(n)]
                theoretical_quantiles = np.array(
                    theoretical_quantiles) * portfolio_returns.std() * 100 + portfolio_returns.mean() * 100

                fig_qq.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_returns,
                    mode='markers',
                    name='Q-Q Plot',
                    marker=dict(color='blue', size=5)
                ))

                # Добавляем диагональную линию для идеального соответствия
                min_val = min(min(theoretical_quantiles), min(sorted_returns))
                max_val = max(max(theoretical_quantiles), max(sorted_returns))

                fig_qq.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Идеальное соответствие',
                    line=dict(color='red', dash='dash')
                ))

                fig_qq.update_layout(
                    title='Q-Q Plot (Сравнение с нормальным распределением)',
                    xaxis_title='Теоретические квантили',
                    yaxis_title='Фактические квантили',
                    hovermode='closest'
                )

                st.plotly_chart(fig_qq, use_container_width=True)

                # Статистические тесты
                st.subheader("Статистические тесты")

                try:
                    # Явно импортируем модуль stats здесь
                    from scipy import stats as scipy_stats

                    # Тест Шапиро-Уилка на нормальность
                    shapiro_stat, shapiro_p = scipy_stats.shapiro(portfolio_returns)

                    # Тест Жарке-Бера
                    jb_stat, jb_p = scipy_stats.jarque_bera(portfolio_returns)

                    # Статистика распределения
                    skewness = portfolio_returns.skew()
                    kurtosis = portfolio_returns.kurtosis()

                    # Отображаем результаты
                    stats_results = pd.DataFrame({
                        'Тест/Метрика': ['Тест Шапиро-Уилка (p-value)', 'Тест Жарке-Бера (p-value)', 'Асимметрия',
                                         'Эксцесс'],
                        'Значение': [shapiro_p, jb_p, skewness, kurtosis]
                    })

                    # Интерпретация результатов
                    interpretations = []

                    if shapiro_p < 0.05:
                        interpretations.append("Тест Шапиро-Уилка: Распределение не является нормальным (p < 0.05)")
                    else:
                        interpretations.append(
                            "Тест Шапиро-Уилка: Невозможно отвергнуть гипотезу о нормальности (p >= 0.05)")

                    if jb_p < 0.05:
                        interpretations.append("Тест Жарке-Бера: Распределение не является нормальным (p < 0.05)")
                    else:
                        interpretations.append(
                            "Тест Жарке-Бера: Невозможно отвергнуть гипотезу о нормальности (p >= 0.05)")

                    if abs(skewness) > 0.5:
                        direction = "положительную" if skewness > 0 else "отрицательную"
                        interpretations.append(
                            f"Распределение имеет {direction} асимметрию (толстый хвост в {'правой' if skewness > 0 else 'левой'} части)")
                    else:
                        interpretations.append("Распределение примерно симметрично")

                    if kurtosis > 0.5:
                        interpretations.append("Распределение имеет тяжелые хвосты (лептокуртозис)")
                    elif kurtosis < -0.5:
                        interpretations.append("Распределение имеет легкие хвосты (платикуртозис)")
                    else:
                        interpretations.append("Эксцесс близок к нормальному распределению")

                    st.dataframe(stats_results.style.format({
                        'Значение': '{:.4f}'
                    }), use_container_width=True)

                    for interpretation in interpretations:
                        st.write(f"• {interpretation}")

                except Exception as e:
                    st.error(f"Не удалось выполнить статистические тесты: {e}")

            with advanced_tabs[3]:
                st.subheader("Множественные метрики эффективности")

                # Создаем таблицу с различными метриками для портфеля и бенчмарка
                # Создаем таблицу с различными метриками для портфеля и бенчмарка
                metrics_data = []  # Создаем список вместо DataFrame

                # Добавляем расширенные метрики доходности
                metrics_data.append({
                    'Метрика': 'Общая доходность (%)',
                    'Портфель': portfolio_metrics.get('total_return', 0) * 100,
                    'Бенчмарк': portfolio_metrics.get('benchmark_return', 0) * 100,
                    'Разница': (portfolio_metrics.get('total_return', 0) - portfolio_metrics.get('benchmark_return',
                                                                                                 0)) * 100
                })

                metrics_data.append({
                    'Метрика': 'Годовая доходность (%)',
                    'Портфель': portfolio_metrics.get('annualized_return', 0) * 100,
                    'Бенчмарк': portfolio_metrics.get('benchmark_annualized_return',
                                                      0) * 100 if 'benchmark_annualized_return' in portfolio_metrics else 0,
                    'Разница': (portfolio_metrics.get('annualized_return', 0) - portfolio_metrics.get(
                        'benchmark_annualized_return',
                        0)) * 100 if 'benchmark_annualized_return' in portfolio_metrics else 0
                })

                # Добавляем метрики риска
                metrics_data.append({
                    'Метрика': 'Волатильность (%)',
                    'Портфель': portfolio_metrics.get('volatility', 0) * 100,
                    'Бенчмарк': portfolio_metrics.get('benchmark_volatility',
                                                      0) * 100 if 'benchmark_volatility' in portfolio_metrics else 0,
                    'Разница': (portfolio_metrics.get('benchmark_volatility', 0) - portfolio_metrics.get('volatility',
                                                                                                         0)) * 100 if 'benchmark_volatility' in portfolio_metrics else 0
                })

                metrics_data.append({
                    'Метрика': 'Максимальная просадка (%)',
                    'Портфель': portfolio_metrics.get('max_drawdown', 0) * 100,
                    'Бенчмарк': portfolio_metrics.get('benchmark_max_drawdown',
                                                      0) * 100 if 'benchmark_max_drawdown' in portfolio_metrics else 0,
                    'Разница': (portfolio_metrics.get('benchmark_max_drawdown', 0) - portfolio_metrics.get(
                        'max_drawdown', 0)) * 100 if 'benchmark_max_drawdown' in portfolio_metrics else 0
                })

                # Добавляем коэффициенты
                metrics_data.append({
                    'Метрика': 'Коэффициент Шарпа',
                    'Портфель': portfolio_metrics.get('sharpe_ratio', 0),
                    'Бенчмарк': portfolio_metrics.get('benchmark_sharpe_ratio',
                                                      0) if 'benchmark_sharpe_ratio' in portfolio_metrics else 0,
                    'Разница': portfolio_metrics.get('sharpe_ratio', 0) - portfolio_metrics.get(
                        'benchmark_sharpe_ratio', 0) if 'benchmark_sharpe_ratio' in portfolio_metrics else 0
                })

                metrics_data.append({
                    'Метрика': 'Коэффициент Сортино',
                    'Портфель': portfolio_metrics.get('sortino_ratio', 0),
                    'Бенчмарк': portfolio_metrics.get('benchmark_sortino_ratio',
                                                      0) if 'benchmark_sortino_ratio' in portfolio_metrics else 0,
                    'Разница': portfolio_metrics.get('sortino_ratio', 0) - portfolio_metrics.get(
                        'benchmark_sortino_ratio', 0) if 'benchmark_sortino_ratio' in portfolio_metrics else 0
                })

                metrics_data.append({
                    'Метрика': 'Коэффициент Кальмара',
                    'Портфель': portfolio_metrics.get('calmar_ratio', 0),
                    'Бенчмарк': portfolio_metrics.get('benchmark_calmar_ratio',
                                                      0) if 'benchmark_calmar_ratio' in portfolio_metrics else 0,
                    'Разница': portfolio_metrics.get('calmar_ratio', 0) - portfolio_metrics.get(
                        'benchmark_calmar_ratio', 0) if 'benchmark_calmar_ratio' in portfolio_metrics else 0
                })

                metrics_data.append({
                    'Метрика': 'Бета',
                    'Портфель': portfolio_metrics.get('beta', 0),
                    'Бенчмарк': 1.0,
                    'Разница': portfolio_metrics.get('beta', 0) - 1.0
                })

                metrics_data.append({
                    'Метрика': 'Альфа (%)',
                    'Портфель': portfolio_metrics.get('alpha', 0) * 100,
                    'Бенчмарк': 0.0,
                    'Разница': portfolio_metrics.get('alpha', 0) * 100
                })

                metrics_data.append({
                    'Метрика': 'Информационный коэффициент',
                    'Портфель': portfolio_metrics.get('information_ratio', 0),
                    'Бенчмарк': 0.0,
                    'Разница': portfolio_metrics.get('information_ratio', 0)
                })

                # Добавляем метрики по результативности
                metrics_data.append({
                    'Метрика': 'Доля выигрышей (%)',
                    'Портфель': portfolio_metrics.get('win_rate', 0) * 100,
                    'Бенчмарк': 0.0,
                    'Разница': 0.0
                })

                metrics_data.append({
                    'Метрика': 'Коэффициент выплат',
                    'Портфель': portfolio_metrics.get('payoff_ratio', 0),
                    'Бенчмарк': 0.0,
                    'Разница': 0.0
                })

                metrics_data.append({
                    'Метрика': 'Фактор прибыли',
                    'Портфель': portfolio_metrics.get('profit_factor', 0),
                    'Бенчмарк': 0.0,
                    'Разница': 0.0
                })

                # Создаем DataFrame из списка данных
                metrics_df = pd.DataFrame(metrics_data)

                # Стилизуем DataFrame
                def color_diff(val):
                    if isinstance(val, float):
                        if val > 0:
                            return 'background-color: rgba(75, 192, 192, 0.2); color: green'
                        elif val < 0:
                            return 'background-color: rgba(255, 99, 132, 0.2); color: red'
                    return ''

                # Отображаем таблицу
                st.dataframe(metrics_df.style.format({
                    'Портфель': '{:.2f}',
                    'Бенчмарк': '{:.2f}',
                    'Разница': '{:.2f}'
                }).applymap(color_diff, subset=['Разница']), use_container_width=True)

                # Визуализация сравнения метрик
                st.subheader("Визуальное сравнение с бенчмарком")

                # Выбираем метрики для визуализации
                metrics_to_plot = ['Общая доходность (%)', 'Годовая доходность (%)', 'Волатильность (%)',
                                   'Максимальная просадка (%)', 'Коэффициент Шарпа', 'Коэффициент Сортино']

                # Создаем новый DataFrame только с выбранными метриками
                plot_df = metrics_df[metrics_df['Метрика'].isin(metrics_to_plot)].copy()

                # Визуализация
                fig_metrics = go.Figure()

                fig_metrics.add_trace(go.Bar(
                    x=plot_df['Метрика'],
                    y=plot_df['Портфель'],
                    name='Портфель',
                    marker_color='blue'
                ))

                fig_metrics.add_trace(go.Bar(
                    x=plot_df['Метрика'],
                    y=plot_df['Бенчмарк'],
                    name='Бенчмарк',
                    marker_color='orange'
                ))

                fig_metrics.update_layout(
                    title='Сравнение ключевых метрик с бенчмарком',
                    xaxis_title='Метрика',
                    yaxis_title='Значение',
                    barmode='group',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig_metrics, use_container_width=True)