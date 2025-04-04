# src/utils/risk_management.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('risk_management')


class RiskManagement:
    """
    Class for risk management and analysis.
    Provides methods for calculating risk metrics, stress testing, scenario analysis, etc.
    """

    @staticmethod
    def calculate_var_parametric(returns: pd.Series, confidence_level: float = 0.95,
                                 time_horizon: int = 1) -> float:
        """
        Calculate Value at Risk (VaR) using parametric method (assumes normal distribution)

        Args:
            returns: Series with returns data
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days

        Returns:
            VaR as a positive float (loss)
        """
        if returns.empty:
            return 0.0

        import scipy.stats as stats

        # Calculate mean and standard deviation
        mu = returns.mean()
        sigma = returns.std()

        # Calculate the z-score for the confidence level
        z = stats.norm.ppf(1 - confidence_level)

        # Calculate VaR
        var = -(mu * time_horizon + z * sigma * np.sqrt(time_horizon))

        return max(0, var)  # Return positive value (loss)

    @staticmethod
    def calculate_var_historical(returns: pd.Series, confidence_level: float = 0.95,
                                 time_horizon: int = 1) -> float:
        """
        Calculate Value at Risk (VaR) using historical method

        Args:
            returns: Series with returns data
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days

        Returns:
            VaR as a positive float (loss)
        """
        if returns.empty:
            return 0.0

        # Calculate the percentile for VaR
        var = -np.percentile(returns, 100 * (1 - confidence_level))

        # Scale VaR for the time horizon
        var_horizon = var * np.sqrt(time_horizon)

        return max(0, var_horizon)  # Return positive value (loss)

    @staticmethod
    def calculate_var_monte_carlo(returns: pd.Series, confidence_level: float = 0.95,
                                  time_horizon: int = 1, simulations: int = 10000) -> float:
        """
        Calculate Value at Risk (VaR) using Monte Carlo simulation

        Args:
            returns: Series with returns data
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            simulations: Number of Monte Carlo simulations

        Returns:
            VaR as a positive float (loss)
        """
        if returns.empty:
            return 0.0

        # Calculate mean and standard deviation
        mu = returns.mean()
        sigma = returns.std()

        # Set the random seed for reproducibility
        np.random.seed(42)

        # Generate random returns
        random_returns = np.random.normal(mu * time_horizon, sigma * np.sqrt(time_horizon), simulations)

        # Calculate VaR
        var = -np.percentile(random_returns, 100 * (1 - confidence_level))

        return max(0, var)  # Return positive value (loss)

    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall

        Args:
            returns: Series with returns data
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            CVaR as a positive float (expected loss)
        """
        if returns.empty:
            return 0.0

        # Calculate VaR
        var = -np.percentile(returns, 100 * (1 - confidence_level))

        # CVaR is the average of returns that are worse than VaR
        cvar = -returns[returns <= -var].mean()

        if np.isnan(cvar):
            return 0.0

        return cvar

    # Улучшаем модель восстановления в методе perform_stress_test
    @staticmethod
    def perform_stress_test(returns: pd.Series, scenario: str, portfolio_value: float = 10000) -> Dict:
        """
        Perform stress testing on a portfolio using historical scenarios

        Args:
            returns: Series with portfolio returns
            scenario: Stress scenario ('financial_crisis_2008', 'covid_2020', etc.)
            portfolio_value: Current portfolio value

        Returns:
            Dictionary with stress test results
        """
        if returns.empty:
            return {'error': 'No returns data provided'}

        # Define historical stress scenarios (percentage losses)
        scenarios = {
            'financial_crisis_2008': {
                'shock': -0.50,  # 50% drop during 2008 financial crisis
                'duration_days': 517,  # ~1.5 years from peak to trough
                'recovery_multiplier': 1.2  # Recovery typically takes 1.2x the decline period
            },
            'covid_2020': {
                'shock': -0.35,  # 35% drop during COVID-19 crash
                'duration_days': 33,  # About 1 month
                'recovery_multiplier': 0.8  # Recovery was relatively quick
            },
            # Другие сценарии...
        }

        # Get the scenario shock parameters
        if scenario not in scenarios:
            return {'error': f'Unknown scenario: {scenario}'}

        scenario_params = scenarios[scenario]
        shock_percentage = scenario_params['shock']
        shock_duration = scenario_params['duration_days']
        recovery_multiplier = scenario_params['recovery_multiplier']

        # Calculate portfolio value after shock
        portfolio_loss = portfolio_value * shock_percentage
        portfolio_after_shock = portfolio_value + portfolio_loss

        # Calculate the number of standard deviations of the shock
        daily_std = returns.std()
        annual_std = daily_std * np.sqrt(252)
        std_deviations = shock_percentage / annual_std

        # Calculate recovery time based on historical data and mean return
        mean_daily_return = returns.mean()

        if mean_daily_return > 0:
            # Учитываем исторический паттерн восстановления для данного сценария
            historical_recovery_days = shock_duration * recovery_multiplier

            # Рассчитываем теоретическое время восстановления на основе средней доходности
            theoretical_recovery_days = -np.log(1 + shock_percentage) / np.log(1 + mean_daily_return)

            # Используем средневзвешенное значение двух подходов
            recovery_days = 0.7 * theoretical_recovery_days + 0.3 * historical_recovery_days
            recovery_months = recovery_days / 21  # Assuming 21 trading days per month
        else:
            recovery_days = float('inf')
            recovery_months = float('inf')

        return {
            'scenario': scenario,
            'shock_percentage': shock_percentage,
            'portfolio_value': portfolio_value,
            'portfolio_loss': portfolio_loss,
            'portfolio_after_shock': portfolio_after_shock,
            'std_deviations': std_deviations,
            'recovery_days': recovery_days,
            'recovery_months': recovery_months,
            'shock_duration_days': shock_duration
        }

    @staticmethod
    def perform_custom_stress_test(returns: pd.DataFrame, weights: Dict[str, float],
                                   shocks: Dict[str, float], portfolio_value: float = 10000) -> Dict:
        """
        Perform custom stress testing on a portfolio using specified shocks for each asset

        Args:
            returns: DataFrame with returns for each asset
            weights: Dictionary with asset weights {ticker: weight}
            shocks: Dictionary with shock percentages for each asset {ticker: shock_percentage}
            portfolio_value: Current portfolio value

        Returns:
            Dictionary with stress test results
        """
        if returns.empty or not weights or not shocks:
            return {'error': 'Missing required data'}

        # Calculate position values
        position_values = {ticker: portfolio_value * weight for ticker, weight in weights.items()}

        # Calculate losses for each position
        position_losses = {}
        total_loss = 0

        for ticker, value in position_values.items():
            if ticker in shocks:
                loss = value * shocks[ticker]
                position_losses[ticker] = loss
                total_loss += loss
            else:
                position_losses[ticker] = 0

        # Calculate portfolio value after shock
        portfolio_after_shock = portfolio_value + total_loss

        return {
            'portfolio_value': portfolio_value,
            'portfolio_loss': total_loss,
            'portfolio_after_shock': portfolio_after_shock,
            'loss_percentage': total_loss / portfolio_value,
            'position_losses': position_losses
        }

    # Добавим этот метод в класс RiskManagement в файле src/utils/risk_management.py

    @staticmethod
    def perform_historical_stress_test(data_fetcher, current_portfolio_tickers, weights, scenario_name,
                                       portfolio_value=10000):
        """
        Проводит стресс-тестирование на основе исторических данных конкретного сценария

        Args:
            data_fetcher: Экземпляр DataFetcher для загрузки исторических данных
            current_portfolio_tickers: Список тикеров в портфеле
            weights: Словарь с весами активов {ticker: weight}
            scenario_name: Название сценария
            portfolio_value: Текущая стоимость портфеля

        Returns:
            Dictionary с результатами стресс-теста
        """
        # Определяем исторические периоды для известных кризисов
        historical_scenarios = {
            'financial_crisis_2008': {
                'name': 'Финансовый кризис 2008',
                'start': '2008-09-01',
                'end': '2009-03-01',
                'description': 'Мировой финансовый кризис после банкротства Lehman Brothers',
                'index_ticker': 'SPY',  # Тикер для отслеживания общей динамики рынка
                'market_impact': -0.50  # Примерное падение индекса для экстраполяции активов без исторических данных
            },
            'covid_2020': {
                'name': 'Пандемия COVID-19',
                'start': '2020-02-15',
                'end': '2020-03-23',
                'description': 'Обвал рынков в начале пандемии COVID-19',
                'index_ticker': 'SPY',
                'market_impact': -0.35
            },
            'tech_bubble_2000': {
                'name': 'Крах доткомов',
                'start': '2000-03-01',
                'end': '2002-10-01',
                'description': 'Крах рынка технологических компаний (2000-2002)',
                'index_ticker': 'SPY',
                'market_impact': -0.45
            },
            'black_monday_1987': {
                'name': 'Черный понедельник',
                'start': '1987-10-14',
                'end': '1987-10-19',
                'description': 'Резкое падение мировых фондовых рынков 19 октября 1987 года',
                'index_ticker': 'SPY',  # Для акций без истории используем SPY
                'market_impact': -0.22
            },
            'inflation_shock': {
                'name': 'Инфляционный шок',
                'start': '2021-11-01',
                'end': '2022-06-16',
                'description': 'Период высокой инфляции 2021-2022',
                'index_ticker': 'SPY',
                'market_impact': -0.20
            },
            'rate_hike_2018': {
                'name': 'Повышение ставок 2018',
                'start': '2018-10-01',
                'end': '2018-12-24',
                'description': 'Падение рынка при повышении ставок ФРС 2018',
                'index_ticker': 'SPY',
                'market_impact': -0.18
            },
            'moderate_recession': {
                'name': 'Умеренная рецессия',
                'start': '2018-10-01',  # Использует период из rate_hike_2018
                'end': '2018-12-24',
                'description': 'Моделирование умеренной рецессии',
                'index_ticker': 'SPY',
                'market_impact': -0.25
            },
            'severe_recession': {
                'name': 'Тяжелая рецессия',
                'start': '2008-09-01',  # Использует период из financial_crisis_2008
                'end': '2009-03-01',
                'description': 'Моделирование тяжелой рецессии на основе кризиса 2008',
                'index_ticker': 'SPY',
                'market_impact': -0.45
            }
        }

        # Проверяем, существует ли сценарий
        if scenario_name not in historical_scenarios:
            return {
                'error': f'Неизвестный сценарий: {scenario_name}',
                'available_scenarios': list(historical_scenarios.keys())
            }

        scenario = historical_scenarios[scenario_name]
        start_date = scenario['start']
        end_date = scenario['end']

        # Добавляем индекс в список тикеров, если его нет
        tickers_to_check = current_portfolio_tickers.copy()
        if scenario['index_ticker'] not in tickers_to_check:
            tickers_to_check.append(scenario['index_ticker'])

        # Получаем исторические данные для периода сценария
        try:
            historical_data = data_fetcher.get_batch_data(tickers_to_check, start_date, end_date)
        except Exception as e:
            # В случае ошибки возвращаем резервный вариант с фиксированным процентом
            return {
                'scenario': scenario_name,
                'scenario_description': scenario['description'],
                'shock_percentage': scenario['market_impact'],
                'portfolio_value': portfolio_value,
                'portfolio_loss': portfolio_value * scenario['market_impact'],
                'portfolio_after_shock': portfolio_value + (portfolio_value * scenario['market_impact']),
                'error_msg': f"Не удалось загрузить исторические данные: {str(e)}. Используем фиксированный коэффициент."
            }

        # Для каждого актива рассчитываем изменение цены за период
        asset_price_changes = {}
        index_price_change = None

        # Проверяем, есть ли данные для индекса
        if scenario['index_ticker'] in historical_data and not historical_data[scenario['index_ticker']].empty:
            index_data = historical_data[scenario['index_ticker']]
            if len(index_data) >= 2:
                first_index_price = index_data['Close'].iloc[0]
                last_index_price = index_data['Close'].iloc[-1]
                index_price_change = (last_index_price - first_index_price) / first_index_price

        # Если не удалось получить изменение индекса, используем заданный market_impact
        if index_price_change is None:
            index_price_change = scenario['market_impact']

        # Рассчитываем изменение цены для каждого актива
        for ticker in current_portfolio_tickers:
            if ticker in historical_data and not historical_data[ticker].empty:
                ticker_data = historical_data[ticker]
                if len(ticker_data) >= 2:
                    first_price = ticker_data['Close'].iloc[0]
                    last_price = ticker_data['Close'].iloc[-1]
                    price_change = (last_price - first_price) / first_price
                    asset_price_changes[ticker] = price_change
                else:
                    # Если недостаточно данных, используем изменение индекса с коэффициентом бета = 1
                    asset_price_changes[ticker] = index_price_change
            else:
                # Если нет данных для актива, используем изменение индекса с коэффициентом бета = 1
                asset_price_changes[ticker] = index_price_change

        # Рассчитываем общий эффект на портфель
        portfolio_impact = 0
        position_impacts = {}

        for ticker, weight in weights.items():
            if ticker in asset_price_changes:
                ticker_impact = asset_price_changes[ticker] * weight
                position_value = portfolio_value * weight
                position_loss = position_value * asset_price_changes[ticker]

                portfolio_impact += ticker_impact
                position_impacts[ticker] = {
                    'weight': weight,
                    'price_change': asset_price_changes[ticker],
                    'position_value': position_value,
                    'position_loss': position_loss
                }

        portfolio_loss = portfolio_value * portfolio_impact
        portfolio_after_shock = portfolio_value + portfolio_loss

        # Рассчитываем примерное время восстановления
        avg_annual_return = 0.07  # Предполагаемая среднегодовая доходность рынка 7%
        daily_return = (1 + avg_annual_return) ** (1 / 252) - 1

        if portfolio_impact < 0:
            # Рассчитываем количество дней для восстановления
            recovery_days = -np.log(1 + portfolio_impact) / np.log(1 + daily_return)
            recovery_months = recovery_days / 21  # примерно 21 торговый день в месяце
        else:
            recovery_days = 0
            recovery_months = 0

        # Формируем результат
        result = {
            'scenario': scenario_name,
            'scenario_name': scenario['name'],
            'scenario_description': scenario['description'],
            'period': f"{start_date} - {end_date}",
            'shock_percentage': portfolio_impact,
            'portfolio_value': portfolio_value,
            'portfolio_loss': portfolio_loss,
            'portfolio_after_shock': portfolio_after_shock,
            'recovery_days': recovery_days,
            'recovery_months': recovery_months,
            'position_impacts': position_impacts,
            'index_price_change': index_price_change
        }

        return result

    # Добавим этот метод в класс RiskManagement в файле src/utils/risk_management.py

    @staticmethod
    def perform_advanced_custom_stress_test(returns, weights, custom_shocks, asset_sectors=None, portfolio_value=10000,
                                            correlation_adjusted=True, use_beta=True):
        """
        Проводит пользовательский стресс-тест с учетом корреляций между активами

        Args:
            returns: DataFrame с историческими доходностями для каждого актива
            weights: Словарь с весами активов {ticker: weight}
            custom_shocks: Словарь с шоковыми изменениями для рынка/активов/секторов
            asset_sectors: Словарь с секторной принадлежностью активов {ticker: sector}
            portfolio_value: Текущая стоимость портфеля
            correlation_adjusted: Использовать ли корреляционные эффекты
            use_beta: Использовать ли бету для оценки воздействия рыночного шока

        Returns:
            Dictionary с результатами стресс-теста
        """
        if returns.empty or not weights or not custom_shocks:
            return {'error': 'Отсутствуют необходимые данные'}

        # Приводим словари к множеству ключей, имеющихся в обоих
        tickers = set(weights.keys()).intersection(set(returns.columns))

        # Общий рыночный шок (если задан)
        market_shock = custom_shocks.get('market', 0)

        # Рассчитываем позиционные значения
        position_values = {ticker: portfolio_value * weight for ticker, weight in weights.items() if ticker in tickers}

        # Рассчитываем беты для каждого актива относительно рынка
        betas = {}
        if use_beta and market_shock != 0 and 'SPY' in returns.columns:
            market_returns = returns['SPY']  # Используем SPY как маркет-индекс
            market_var = market_returns.var()

            for ticker in tickers:
                asset_returns = returns[ticker]
                if market_var > 0:
                    asset_cov = asset_returns.cov(market_returns)
                    beta = asset_cov / market_var
                    betas[ticker] = beta
                else:
                    betas[ticker] = 1.0  # По умолчанию бета = 1
        else:
            # Если нет рыночного индекса или бета не используется, устанавливаем все беты = 1
            betas = {ticker: 1.0 for ticker in tickers}

        # Матрица корреляций (для корректировки по корреляциям)
        correlations = None
        if correlation_adjusted:
            correlations = returns[list(tickers)].corr()

        # Рассчитываем шок для каждого актива с учетом:
        # 1. Прямого шока, заданного для актива в custom_shocks
        # 2. Рыночного шока * бета (если используется бета)
        # 3. Секторного шока, если актив принадлежит к сектору с заданным шоком
        asset_shocks = {}
        sector_shocks = {k: v for k, v in custom_shocks.items() if k != 'market' and k != 'assets'}
        asset_specific_shocks = custom_shocks.get('assets', {})

        for ticker in tickers:
            # Начинаем с рыночного шока, скорректированного по бете
            shock = market_shock * betas.get(ticker, 1.0)

            # Добавляем секторный шок, если указан сектор для актива
            if asset_sectors and ticker in asset_sectors:
                sector = asset_sectors[ticker]
                if sector in sector_shocks:
                    shock += sector_shocks[sector]

            # Добавляем специфический шок для актива, если он задан
            if ticker in asset_specific_shocks:
                shock += asset_specific_shocks[ticker]

            asset_shocks[ticker] = shock

        # Применяем корреляционную корректировку, если задано
        if correlation_adjusted and correlations is not None:
            for ticker1 in tickers:
                for ticker2 in tickers:
                    if ticker1 != ticker2:
                        # Корректируем шок с учетом корреляции между активами
                        # Чем выше корреляция, тем больше влияние шока одного актива на другой
                        corr = correlations.loc[ticker1, ticker2]
                        asset_shocks[ticker1] += 0.1 * corr * asset_shocks[
                            ticker2]  # Коэффициент 0.1 уменьшает эффект для большей реалистичности

        # Рассчитываем потери для каждой позиции
        position_losses = {}
        total_loss = 0

        for ticker, value in position_values.items():
            if ticker in asset_shocks:
                loss = value * asset_shocks[ticker]
                position_losses[ticker] = loss
                total_loss += loss

        # Рассчитываем потерю портфеля и стоимость после шока
        portfolio_after_shock = portfolio_value + total_loss
        loss_percentage = total_loss / portfolio_value if portfolio_value > 0 else 0

        # Создаем подробную информацию о шоках для каждого актива
        detailed_impacts = {}
        for ticker in tickers:
            if ticker in asset_shocks and ticker in position_values:
                detailed_impacts[ticker] = {
                    'weight': weights.get(ticker, 0),
                    'beta': betas.get(ticker, 1.0),
                    'shock_percentage': asset_shocks[ticker],
                    'position_value': position_values.get(ticker, 0),
                    'position_loss': position_losses.get(ticker, 0)
                }

        # Оцениваем примерное время восстановления
        recovery_calculation = {
            'avg_annual_return': 0.07,  # Предполагаемая среднегодовая доходность 7%
            'daily_return': (1 + 0.07) ** (1 / 252) - 1
        }

        if loss_percentage < 0:
            recovery_days = -np.log(1 + loss_percentage) / np.log(1 + recovery_calculation['daily_return'])
            recovery_months = recovery_days / 21  # примерно 21 торговый день в месяце
        else:
            recovery_days = 0
            recovery_months = 0

        return {
            'portfolio_value': portfolio_value,
            'portfolio_loss': total_loss,
            'portfolio_after_shock': portfolio_after_shock,
            'loss_percentage': loss_percentage,
            'position_losses': position_losses,
            'detailed_impacts': detailed_impacts,
            'recovery_days': recovery_days,
            'recovery_months': recovery_months
        }

    @staticmethod
    def calculate_risk_contribution(returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate risk contribution of each asset to portfolio risk

        Args:
            returns: DataFrame with returns for each asset
            weights: Dictionary with asset weights {ticker: weight}

        Returns:
            Dictionary with risk contribution percentages {ticker: contribution}
        """
        if returns.empty or not weights:
            return {}

        # Filter returns to include only assets in weights
        tickers = list(weights.keys())
        filtered_returns = returns[tickers].copy()

        # Calculate covariance matrix
        cov_matrix = filtered_returns.cov()

        # Create weights array
        weight_array = np.array([weights.get(ticker, 0) for ticker in filtered_returns.columns])

        # Calculate portfolio variance
        portfolio_variance = np.dot(weight_array.T, np.dot(cov_matrix, weight_array))

        # Calculate marginal contribution to risk
        mcr = np.dot(cov_matrix, weight_array)

        # Calculate component contribution to risk
        ccr = weight_array * mcr

        # Calculate percentage contribution to risk
        pcr = ccr / portfolio_variance

        # Create dictionary with risk contributions
        risk_contribution = {ticker: pcr[i] for i, ticker in enumerate(filtered_returns.columns)}

        return risk_contribution

    @staticmethod
    def perform_monte_carlo_simulation(returns: pd.Series, initial_value: float = 10000,
                                       years: int = 10, simulations: int = 1000,
                                       annual_contribution: float = 0) -> Dict:
        """
        Perform Monte Carlo simulation to project portfolio value

        Args:
            returns: Series with portfolio returns
            initial_value: Initial portfolio value
            years: Number of years to simulate
            simulations: Number of Monte Carlo simulations
            annual_contribution: Annual contribution to the portfolio

        Returns:
            Dictionary with simulation results
        """
        if returns.empty:
            return {'error': 'No returns data provided'}

        # Calculate mean and standard deviation of returns
        mu = returns.mean()
        sigma = returns.std()

        # Calculate annualized mean and standard deviation
        annual_mu = mu * 252
        annual_sigma = sigma * np.sqrt(252)

        # Calculate daily mean and standard deviation for simulation
        daily_mu = annual_mu / 252
        daily_sigma = annual_sigma / np.sqrt(252)

        # Calculate number of trading days to simulate
        trading_days = years * 252

        # Set random seed for reproducibility
        np.random.seed(42)

        # Initialize array for simulation results
        simulation_results = np.zeros((simulations, trading_days + 1))
        simulation_results[:, 0] = initial_value

        # Daily contribution (if annual contribution is provided)
        daily_contribution = annual_contribution / 252 if annual_contribution else 0

        # Run Monte Carlo simulation
        for sim in range(simulations):
            for day in range(trading_days):
                # Generate random return
                random_return = np.random.normal(daily_mu, daily_sigma)

                # Calculate new portfolio value
                simulation_results[sim, day + 1] = simulation_results[sim, day] * (
                            1 + random_return) + daily_contribution

        # Calculate statistics from simulation results
        final_values = simulation_results[:, -1]
        percentiles = {
            'min': np.min(final_values),
            'max': np.max(final_values),
            'median': np.median(final_values),
            'mean': np.mean(final_values),
            'p10': np.percentile(final_values, 10),
            'p25': np.percentile(final_values, 25),
            'p75': np.percentile(final_values, 75),
            'p90': np.percentile(final_values, 90)
        }

        # Calculate probability of reaching certain targets
        targets = {
            'double': initial_value * 2,
            'triple': initial_value * 3,
            'quadruple': initial_value * 4
        }

        probabilities = {
            f'prob_reaching_{name}': np.mean(final_values >= target) for name, target in targets.items()
        }

        return {
            'initial_value': initial_value,
            'years': years,
            'simulations': simulations,
            'annual_contribution': annual_contribution,
            'annual_mean_return': annual_mu,
            'annual_volatility': annual_sigma,
            'percentiles': percentiles,
            'probabilities': probabilities,
            'simulation_data': simulation_results  # Full simulation data for visualization
        }

    # Добавить в класс RiskManagement в файле src/utils/risk_management.py

    @staticmethod
    def analyze_drawdowns(returns: pd.Series) -> pd.DataFrame:
        """
        Анализирует периоды просадок и возвращает подробную информацию о них

        Args:
            returns: Серия доходностей

        Returns:
            DataFrame с информацией о просадках
        """
        if returns.empty:
            return pd.DataFrame()

        # Нормализуем индекс, чтобы устранить проблемы с часовыми поясами
        returns_index = returns.index.tz_localize(None) if returns.index.tz else returns.index
        returns = returns.copy()
        returns.index = returns_index

        # Рассчитываем кумулятивную доходность
        cum_returns = (1 + returns).cumprod()

        # Находим пики (максимумы)
        peak = cum_returns.cummax()

        # Рассчитываем просадки
        drawdowns = (cum_returns / peak - 1)

        # Находим периоды просадок
        is_drawdown = drawdowns < 0

        # Если нет просадок, возвращаем пустой DataFrame
        if not is_drawdown.any():
            return pd.DataFrame(columns=['start_date', 'valley_date', 'recovery_date', 'depth', 'length', 'recovery'])

        # Группируем последовательные периоды просадок
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        valley_date = None
        valley_value = 0

        for date, value in drawdowns.items():
            if value < 0 and not in_drawdown:
                # Начало новой просадки
                in_drawdown = True
                start_date = date
                valley_date = date
                valley_value = value
            elif value < 0 and in_drawdown:
                # Продолжение просадки
                if value < valley_value:
                    valley_date = date
                    valley_value = value
            elif value >= 0 and in_drawdown:
                # Конец просадки (восстановление)
                drawdown_periods.append({
                    'start_date': start_date,
                    'valley_date': valley_date,
                    'recovery_date': date,
                    'depth': valley_value,
                    'length': (date - start_date).days,
                    'recovery': (date - valley_date).days
                })
                in_drawdown = False

        # Если мы все еще в просадке на последнюю дату
        if in_drawdown:
            drawdown_periods.append({
                'start_date': start_date,
                'valley_date': valley_date,
                'recovery_date': None,
                'depth': valley_value,
                'length': (returns.index[-1] - start_date).days,
                'recovery': None
            })

        # Создаем DataFrame и сортируем по глубине просадки
        dd_df = pd.DataFrame(drawdown_periods)
        if not dd_df.empty:
            dd_df = dd_df.sort_values('depth')

        return dd_df

    @staticmethod
    def calculate_underwater_series(returns: pd.Series) -> pd.Series:
        """
        Рассчитывает серию подводных значений (underwater) для визуализации

        Args:
            returns: Серия доходностей

        Returns:
            Серия подводных значений
        """
        if returns.empty:
            return pd.Series()

        # Рассчитываем кумулятивную доходность
        cum_returns = (1 + returns).cumprod()

        # Находим пики (максимумы)
        peak = cum_returns.cummax()

        # Рассчитываем просадки
        underwater = (cum_returns / peak - 1)

        return underwater