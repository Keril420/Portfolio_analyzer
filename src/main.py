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
# Настройка страницы
st.set_page_config(
    page_title="Система управления инвестиционным портфелем",
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
        "Оптимизация портфеля"
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

    # Дополнительная информация в боковой панели
    with st.sidebar.expander("О программе"):
        st.write("""
        **Система управления инвестиционным портфелем** позволяет создавать, 
        анализировать и оптимизировать инвестиционные портфели с использованием 
        различных стратегий и моделей.

        Автор: Имя Автора
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
    st.title("Система управления инвестиционным портфелем")

    # Приветственный текст
    st.write("""
    Добро пожаловать в систему управления инвестиционным портфелем! 

    Это приложение поможет вам создавать, анализировать и оптимизировать инвестиционные 
    портфели с использованием современных финансовых моделей и аналитических инструментов.
    """)

    # Разделяем экран на две колонки
    col1, col2 = st.columns(2)

    with col1:
        # Краткий обзор функций
        st.subheader("Возможности системы")

        st.markdown("""
        #### 📈 Создание портфеля
        - Ручной ввод тикеров и весов
        - Импорт из CSV-файла
        - Готовые шаблоны портфелей
        - Поиск активов по названию

        #### 📊 Анализ портфеля
        - Расчет ключевых показателей доходности и риска
        - Анализ корреляций между активами
        - Визуализация результатов
        - Стресс-тестирование

        #### 🔍 Оптимизация портфеля
        - Оптимизация по различным методам (Марковиц, равный риск и т.д.)
        - Тактическое распределение активов
        - Монте-Карло симуляция
        """)

    with col2:
        # Получаем список портфелей
        portfolios = portfolio_manager.list_portfolios()

        st.subheader("Ваши портфели")

        if portfolios:
            # Создаем таблицу с портфелями
            portfolios_df = pd.DataFrame({
                'Название': [p['name'] for p in portfolios],
                'Активов': [p['asset_count'] for p in portfolios],
                'Последнее обновление': [p['last_updated'] for p in portfolios]
            })

            st.dataframe(portfolios_df, use_container_width=True)

            # Кнопки для быстрых действий
            st.subheader("Быстрые действия")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Создать новый портфель"):
                    st.session_state.current_page = "Создание портфеля"
                    st.experimental_rerun()

            with col2:
                if st.button("Анализировать существующий"):
                    st.session_state.current_page = "Анализ портфеля"
                    st.experimental_rerun()
        else:
            st.info("У вас пока нет созданных портфелей. Начните с создания нового портфеля.")

            if st.button("Создать первый портфель"):
                st.session_state.current_page = "Создание портфеля"
                st.experimental_rerun()

    # Информационная секция
    st.subheader("Как начать работу")

    with st.expander("Инструкция для начинающих"):
        st.write("""
        ### Пошаговое руководство

        1. **Создайте портфель**: Перейдите в раздел "Создание портфеля" и выберите один из методов создания:
           - Ручной ввод тикеров и весов
           - Импорт из CSV-файла
           - Использование шаблона

        2. **Анализируйте портфель**: После создания портфеля, перейдите в раздел "Анализ портфеля" для
           просмотра ключевых показателей, графиков доходности и оценки рисков.

        3. **Оптимизируйте портфель**: В разделе "Оптимизация портфеля" вы можете улучшить свой портфель,
           используя различные методы оптимизации, тактическое распределение активов или
           проверить будущую доходность с помощью Монте-Карло симуляции.
        """)


if __name__ == "__main__":
    main()