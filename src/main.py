import streamlit as st
from datetime import datetime
import pandas as pd
import os
import sys

# Добавляем корень проекта в путь Python
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.utils.data_fetcher import DataFetcher, PortfolioDataManager
from src.utils.calculations import PortfolioAnalytics
from src.utils.risk_management import RiskManagement
from src.utils.visualization import PortfolioVisualization
import src.config as config
from src.pages import portfolio_creation, portfolio_analysis, portfolio_optimization
from src.utils.scenario_chaining import scenario_chaining_page
from src.utils.advanced_visualizations import create_stress_impact_heatmap, create_interactive_stress_impact_chart, create_risk_tree_visualization
from src.utils.historical_context import display_historical_context, historical_analogy_page

# Настройка страницы
st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация состояния сессии
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Главная"


def main():
    # Инициализация данных
    data_fetcher = DataFetcher(cache_dir=str(config.CACHE_DIR), cache_expiry_days=config.CACHE_EXPIRY_DAYS)
    portfolio_manager = PortfolioDataManager(data_fetcher, storage_dir=str(config.PORTFOLIO_DIR))

    # Задаем API ключи из конфигурации
    if config.ALPHA_VANTAGE_API_KEY:
        data_fetcher.api_keys['alpha_vantage'] = config.ALPHA_VANTAGE_API_KEY

    # Боковая панель навигации
    st.sidebar.title("Навигация")

    # Список страниц
    pages = [
        "Главная",
        "Создание портфеля",
        "Анализ портфеля",
        "Оптимизация портфеля",
        "Цепочки стресс-событий",  # Новая страница
        "Исторические аналогии"  # Новая страница
    ]

    # Выбор страницы
    selected_page = st.sidebar.radio("Выберите раздел:", pages)

    # Обновляем текущую страницу в состоянии сессии
    st.session_state.current_page = selected_page

    # Отображение выбранной страницы
    if selected_page == "Главная":
        show_home_page(data_fetcher, portfolio_manager)
    elif selected_page == "Создание портфеля":
        portfolio_creation.run(data_fetcher, portfolio_manager)
    elif selected_page == "Анализ портфеля":
        portfolio_analysis.run(data_fetcher, portfolio_manager)
    elif selected_page == "Оптимизация портфеля":
        portfolio_optimization.run(data_fetcher, portfolio_manager)
    elif selected_page == "Цепочки стресс-событий":
        scenario_chaining_page()  # Новая страница
    elif selected_page == "Исторические аналогии":
        historical_analogy_page()  # Новая страница

    # Дополнительная информация в боковой панели
    with st.sidebar.expander("О программе"):
        st.write("""
        **Система управления инвестиционным портфелем** позволяет создавать, 
        анализировать и оптимизировать инвестиционные портфели с использованием 
        различных стратегий и моделей.

        Автор: Keril & Claude AI 
        Версия: 1.0.0
        """)

    # Статус API
    with st.sidebar.expander("Статус API"):
        if config.ALPHA_VANTAGE_API_KEY:
            st.success("Alpha Vantage API: Подключено")
        else:
            st.warning("Alpha Vantage API: Не настроено")

        # Отображаем счетчики вызовов API
        st.write("Счетчики вызовов API:")
        st.write(f"- yFinance: {data_fetcher.api_call_counts['yfinance']}")
        st.write(f"- Alpha Vantage: {data_fetcher.api_call_counts['alpha_vantage']}")

        # Кнопка для очистки кеша
        if st.button("Очистить кеш данных"):
            data_fetcher.clear_cache()
            st.success("Кеш данных очищен!")


def show_home_page(data_fetcher, portfolio_manager):
    """
    Функция для отображения главной страницы

    Args:
        data_fetcher: Экземпляр DataFetcher для загрузки данных
        portfolio_manager: Экземпляр PortfolioDataManager для работы с портфелями
    """
    # Заголовок приложения
    st.title("Investment Portfolio Management System")

    # Приветственный текст
    st.write("""
    Welcome to our advanced portfolio management system! This application helps investors create, analyze, optimize, 
    and monitor investment portfolios using sophisticated financial models and interactive visualizations.
    """)

    # Разделяем экран на две колонки
    col1, col2 = st.columns(2)

    with col1:
        # Краткий обзор функций
        st.subheader("Key Capabilities")

        st.markdown("""
        #### 📈 Portfolio Management
        - Create and track multiple investment portfolios
        - Import/export portfolio data from/to CSV and Excel
        - Real-time data fetching for market information
        - Customizable views for different analysis needs

        #### 📊 Advanced Analytics
        - Comprehensive performance metrics (returns, volatility, drawdowns)
        - Risk-adjusted measurements (Sharpe, Sortino, Calmar)
        - Benchmark comparison against major indices
        - Calendar-based analysis and seasonal patterns

        #### 🔍 Risk Assessment
        - Multi-dimensional risk analysis (VaR, CVaR, drawdowns)
        - Stress testing against historical and hypothetical scenarios
        - Correlation analysis to identify portfolio vulnerabilities
        - Risk contribution breakdown by asset and sector

        #### 🧮 Portfolio Optimization
        - Multiple optimization methodologies (Markowitz, Risk Parity)
        - Efficient frontier visualization with interactive selection
        - Custom constraint implementation for real-world limitations
        - Tactical asset allocation recommendations
        """)

    with col2:
        # Получаем список портфелей
        portfolios = portfolio_manager.list_portfolios()

        st.subheader("Your Portfolios")

        if portfolios:
            # Создаем таблицу с портфелями
            portfolios_df = pd.DataFrame({
                'Название': [p['name'] for p in portfolios],
                'Активов': [p['asset_count'] for p in portfolios],
                'Последнее обновление': [p['last_updated'] for p in portfolios]
            })

            st.dataframe(portfolios_df, use_container_width=True)

            # Кнопки для быстрых действий
            st.subheader("Quick Actions")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Create New Portfolio"):
                    st.session_state.current_page = "Создание портфеля"
                    st.experimental_rerun()

            with col2:
                if st.button("Analyze Existing Portfolio"):
                    st.session_state.current_page = "Анализ портфеля"
                    st.experimental_rerun()
        else:
            st.info("You don't have any portfolios yet. Start by creating your first portfolio.")

            if st.button("Create First Portfolio"):
                st.session_state.current_page = "Создание портфеля"
                st.experimental_rerun()

    # Информационная секция
    st.subheader("Getting Started")

    with st.expander("Guide for Beginners"):
        st.write("""
        ### Step-by-Step Guide

        1. **Create or Import a Portfolio**: Navigate to the "Create Portfolio" section to:
           - Add investments manually with ticker search
           - Import from CSV with your existing holdings
           - Use templates for common investment strategies
           - Specify weights or dollar amounts for each position

        2. **Analyze Your Portfolio**: In the "Portfolio Analysis" section, you can:
           - Review key performance metrics and risk measures
           - Compare against benchmarks and historical periods
           - Explore asset correlations and diversification metrics
           - Conduct stress tests and scenario analysis

        3. **Optimize Your Portfolio**: The "Portfolio Optimization" section allows you to:
           - Visualize the efficient frontier for your asset universe
           - Find the optimal portfolio based on your risk preference
           - Apply different optimization methodologies
           - Set constraints on asset allocations

        4. **Explore Advanced Features**: Additional specialized tools include:
           - Stress scenario chains for modeling complex market events
           - Historical analogies for market condition comparison
           - Rolling metrics to observe changing performance characteristics
           - Monte Carlo simulations for future projections
        """)

    # Practical Applications Section
    st.subheader("Practical Applications")

    st.write("""
    - **Long-term investors**: Create and monitor diversified portfolios aligned with your investment goals
    - **Active traders**: Analyze risk exposures and optimize position sizing
    - **Financial advisors**: Demonstrate portfolio characteristics and potential improvements to clients
    - **Students and researchers**: Explore financial theories with real market data

    This system combines modern portfolio theory, quantitative risk management, and interactive data visualization 
    to provide powerful insights for investment decision-making.
    """)


if __name__ == "__main__":
    main()