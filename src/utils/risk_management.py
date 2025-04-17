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

    # Improving the recovery model in the perform_stress_test method
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
            # Consider the historical recovery pattern for this scenario
            historical_recovery_days = shock_duration * recovery_multiplier

            # Calculate theoretical recovery time based on average yield
            theoretical_recovery_days = -np.log(1 + shock_percentage) / np.log(1 + mean_daily_return)

            # We use the weighted average of the two approaches
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

    # Change the function declaration:
    @staticmethod
    def perform_historical_stress_test(data_fetcher, current_portfolio_tickers, weights, scenario_name,
                                       portfolio_value=10000, portfolio_data=None):
        """
        Conducts stress testing based on historical data of a specific scenario

        Args:
           data_fetcher: DataFetcher instance for loading historical data
           current_portfolio_tickers: List of tickers in the portfolio
           weights: Dictionary with asset weights {ticker: weight}
           scenario_name: Scenario name
           portfolio_value: Current portfolio value
           portfolio_data: Portfolio data (optional)

        Returns:
            Dictionary with stress test results
        """
        # We define historical periods for known crises
        historical_scenarios = {
            'financial_crisis_2008': {
                'name': 'Financial crisis 2008',
                'start': '2008-09-01',
                'end': '2009-03-01',
                'description': 'The global financial crisis after the bankruptcy of Lehman Brothers',
                'index_ticker': 'SPY',  # Ticker for tracking overall market dynamics
                'market_impact': -0.50  # Approximate index decline for asset extrapolation without historical data
            },
            'covid_2020': {
                'name': 'COVID-19 Pandemic',
                'start': '2020-02-15',
                'end': '2020-03-23',
                'description': 'Markets Crash at the Beginning of the COVID-19 Pandemic',
                'index_ticker': 'SPY',
                'market_impact': -0.35
            },
            'tech_bubble_2000': {
                'name': 'The Dot-com Crash',
                'start': '2000-03-01',
                'end': '2002-10-01',
                'description': 'Tech Market Crash (2000-2002)',
                'index_ticker': 'SPY',
                'market_impact': -0.45
            },
            'black_monday_1987': {
                'name': 'Black Monday',
                'start': '1987-10-14',
                'end': '1987-10-19',
                'description': 'The sharp fall of world stock markets on October 19, 1987',
                'index_ticker': 'SPY',
                'market_impact': -0.22
            },
            'inflation_shock': {
                'name': 'Inflation shock',
                'start': '2021-11-01',
                'end': '2022-06-16',
                'description': 'High inflation period 2021-2022',
                'index_ticker': 'SPY',
                'market_impact': -0.20
            },
            'rate_hike_2018': {
                'name': 'Rate hike 2018',
                'start': '2018-10-01',
                'end': '2018-12-24',
                'description': 'Market Falls as Fed Rates Hike 2018',
                'index_ticker': 'SPY',
                'market_impact': -0.18
            },
            'moderate_recession': {
                'name': 'Moderate recession',
                'start': '2018-10-01',  # Uses period from rate_hike_2018
                'end': '2018-12-24',
                'description': 'Modeling a Moderate Recession',
                'index_ticker': 'SPY',
                'market_impact': -0.25
            },
            'severe_recession': {
                'name': 'Severe recession',
                'start': '2008-09-01',  # Uses period from financial_crisis_2008
                'end': '2009-03-01',
                'description': 'Modeling a severe recession based on the 2008 crisis',
                'index_ticker': 'SPY',
                'market_impact': -0.45
            }
        }

        # Check if the script exists
        if scenario_name not in historical_scenarios:
            return {
                'error': f'Unknown scenario: {scenario_name}',
                'available_scenarios': list(historical_scenarios.keys())
            }

        scenario = historical_scenarios[scenario_name]
        start_date = scenario['start']
        end_date = scenario['end']

        # Add index to ticker list if it is not there
        tickers_to_check = current_portfolio_tickers.copy()
        if scenario['index_ticker'] not in tickers_to_check:
            tickers_to_check.append(scenario['index_ticker'])

        # Get historical data for the scenario period
        try:
            historical_data = data_fetcher.get_batch_data(tickers_to_check, start_date, end_date)
        except Exception as e:
            # In case of error, return the fallback option with a fixed percentage
            return {
                'scenario': scenario_name,
                'scenario_description': scenario['description'],
                'shock_percentage': scenario['market_impact'],
                'portfolio_value': portfolio_value,
                'portfolio_loss': portfolio_value * scenario['market_impact'],
                'portfolio_after_shock': portfolio_value + (portfolio_value * scenario['market_impact']),
                'error_msg': f"Failed to load historical data: {str(e)}. use a fixed coefficient."
            }

        # For each asset we calculate the price change over the period
        asset_price_changes = {}
        index_price_change = None

        # Check if there is data for the index
        if scenario['index_ticker'] in historical_data and not historical_data[scenario['index_ticker']].empty:
            index_data = historical_data[scenario['index_ticker']]
            if len(index_data) >= 2:
                first_index_price = index_data['Close'].iloc[0]
                last_index_price = index_data['Close'].iloc[-1]
                index_price_change = (last_index_price - first_index_price) / first_index_price

        # If unable to get index change, use specified market_impact
        if index_price_change is None:
            index_price_change = scenario['market_impact']

        # Calculate the price change for each asset
        for ticker in current_portfolio_tickers:
            if ticker in historical_data and not historical_data[ticker].empty:
                ticker_data = historical_data[ticker]
                if len(ticker_data) >= 2:
                    first_price = ticker_data['Close'].iloc[0]
                    last_price = ticker_data['Close'].iloc[-1]
                    price_change = (last_price - first_price) / first_price
                    asset_price_changes[ticker] = price_change

                if ticker not in historical_data or historical_data[ticker].empty:
                    # Initialize a dictionary to keep track of approximation types
                    if 'approximation_types' not in locals():
                        approximation_types = {}

                    # Search for assets of the same sector
                    sector_tickers = []
                    ticker_sector = None

                    # Determine the sector of the current asset
                    if portfolio_data and 'assets' in portfolio_data:
                        for asset in portfolio_data['assets']:
                            if asset['ticker'] == ticker and 'sector' in asset:
                                ticker_sector = asset['sector']
                                break

                    # If the sector is known, we collect tickers from the same sector
                    if ticker_sector and portfolio_data and 'assets' in portfolio_data:
                        for asset in portfolio_data['assets']:
                            if 'sector' in asset and asset['sector'] == ticker_sector and asset['ticker'] != ticker:
                                sector_tickers.append(asset['ticker'])

                    # If assets from the same sector with the data are found, use their average change
                    if sector_tickers:
                        sector_changes = []
                        for sector_ticker in sector_tickers:
                            if sector_ticker in historical_data and not historical_data[sector_ticker].empty:
                                s_data = historical_data[sector_ticker]
                                if len(s_data) >= 2:
                                    s_first = s_data['Close'].iloc[0]
                                    s_last = s_data['Close'].iloc[-1]
                                    s_change = (s_last - s_first) / s_first
                                    sector_changes.append(s_change)

                        if sector_changes:
                            # Use the average change per sector
                            asset_price_changes[ticker] = sum(sector_changes) / len(sector_changes)
                            approximation_types[ticker] = "sector_proxy"
                            continue

                    # If there is no data on the sector or other assets in the sector, use the beta-adjusted index
                    # Get the current beta if possible
                    current_beta = 1.0  # Default

                    # Initialize the beta dictionary if it doesn't exist yet
                    if 'betas' not in locals():
                        betas = {}


                    if ticker in betas:
                        current_beta = betas[ticker]

                    asset_price_changes[ticker] = index_price_change * current_beta
                    approximation_types[ticker] = "market_proxy"
                else:
                    # If there is not enough data, we use the index change with beta = 1
                    asset_price_changes[ticker] = index_price_change
            else:
                # If there is no data for an asset, we use the change in the index with a beta coefficient of 1
                asset_price_changes[ticker] = index_price_change

        # Calculate the overall effect on the portfolio
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

        # Calculate the approximate recovery time
        avg_annual_return = 0.07  # Estimated average annual market return 7%
        daily_return = (1 + avg_annual_return) ** (1 / 252) - 1

        if portfolio_impact < 0:
            # Calculate the number of days for recovery
            recovery_days = -np.log(1 + portfolio_impact) / np.log(1 + daily_return)
            recovery_months = recovery_days / 21  # approximately 21 trading days per month
        else:
            recovery_days = 0
            recovery_months = 0

        # Forming the result
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

    @staticmethod
    def perform_advanced_custom_stress_test(returns, weights, custom_shocks, asset_sectors=None, portfolio_value=10000,
                                            correlation_adjusted=True, use_beta=True):
        """
        Conducts a custom stress test taking into account correlations between assets

        Args:
           returns: DataFrame with historical returns for each asset
           weights: Dictionary with asset weights {ticker: weight}
           custom_shocks: Dictionary with market/asset/sector shocks
           asset_sectors: Dictionary with sector assignments of assets {ticker: sector}
           portfolio_value: Current portfolio value
           correlation_adjusted: Whether to use correlation effects
           use_beta: Whether to use beta to estimate the impact of a market shock

        Returns:
            Dictionary with stress test results
        """
        if returns.empty or not weights or not custom_shocks:
            return {'error': 'Required data is missing'}

        # Reduce dictionaries to a set of keys that are present in both
        tickers = set(weights.keys()).intersection(set(returns.columns))

        # General market shock (if given)
        market_shock = custom_shocks.get('market', 0)

        # Calculate positional values
        position_values = {ticker: portfolio_value * weight for ticker, weight in weights.items() if ticker in tickers}

        # Calculate betas for each asset relative to the market
        betas = {}
        if use_beta and market_shock != 0 and 'SPY' in returns.columns:
            market_returns = returns['SPY']  # Using SPY as a Market Index
            market_var = market_returns.var()

            for ticker in tickers:
                asset_returns = returns[ticker]
                if market_var > 0:
                    asset_cov = asset_returns.cov(market_returns)
                    beta = asset_cov / market_var
                    betas[ticker] = beta
                else:
                    betas[ticker] = 1.0
        else:

            betas = {ticker: 1.0 for ticker in tickers}

        # Correlation matrix (for correlation adjustment)
        correlations = None
        if correlation_adjusted:
            correlations = returns[list(tickers)].corr()

        # Calculate the shock for each asset taking into account:
        # 1. Direct shock specified for the asset in custom_shocks
        # 2. Market shock * beta (if beta is used)
        # 3. Sector shock, if the asset belongs to a sector with the specified shock
        asset_shocks = {}
        sector_shocks = {k: v for k, v in custom_shocks.items() if k != 'market' and k != 'assets'}
        asset_specific_shocks = custom_shocks.get('assets', {})

        for ticker in tickers:

            shock = market_shock * betas.get(ticker, 1.0)


            if asset_sectors and ticker in asset_sectors:
                sector = asset_sectors[ticker]
                if sector in sector_shocks:
                    shock += sector_shocks[sector]


            if ticker in asset_specific_shocks:
                shock += asset_specific_shocks[ticker]

            asset_shocks[ticker] = shock

        # Apply correlation correction if specified
        if correlation_adjusted and correlations is not None:
            for ticker1 in tickers:
                for ticker2 in tickers:
                    if ticker1 != ticker2:
                        # Adjust the shock for the correlation between assets
                        # The higher the correlation, the greater the impact of a shock to one asset on another
                        corr = correlations.loc[ticker1, ticker2]
                        asset_shocks[ticker1] += 0.1 * corr * asset_shocks[
                            ticker2]  # Factor 0.1 reduces the effect for greater realism

        # Calculate losses for each position
        position_losses = {}
        total_loss = 0

        for ticker, value in position_values.items():
            if ticker in asset_shocks:
                loss = value * asset_shocks[ticker]
                position_losses[ticker] = loss
                total_loss += loss

        # Calculate portfolio loss and value after shock
        portfolio_after_shock = portfolio_value + total_loss
        loss_percentage = total_loss / portfolio_value if portfolio_value > 0 else 0

        # Create detailed shock information for each asset
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

        # Estimate the approximate recovery time
        recovery_calculation = {
            'avg_annual_return': 0.07,
            'daily_return': (1 + 0.07) ** (1 / 252) - 1
        }

        if loss_percentage < 0:
            recovery_days = -np.log(1 + loss_percentage) / np.log(1 + recovery_calculation['daily_return'])
            recovery_months = recovery_days / 21  # approximately 21 trading days per month
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
        Analyzes periods of drawdowns and returns detailed information about them

        Args:
            returns: Returns Series

        Returns:
            DataFrame with drawdown information
        """
        if returns.empty:
            return pd.DataFrame()


        returns_index = returns.index.tz_localize(None) if returns.index.tz else returns.index
        returns = returns.copy()
        returns.index = returns_index

        cum_returns = (1 + returns).cumprod()

        peak = cum_returns.cummax()

        drawdowns = (cum_returns / peak - 1)

        is_drawdown = drawdowns < 0

        if not is_drawdown.any():
            return pd.DataFrame(columns=['start_date', 'valley_date', 'recovery_date', 'depth', 'length', 'recovery'])

        drawdown_periods = []
        in_drawdown = False
        start_date = None
        valley_date = None
        valley_value = 0

        for date, value in drawdowns.items():
            if value < 0 and not in_drawdown:
                in_drawdown = True
                start_date = date
                valley_date = date
                valley_value = value
            elif value < 0 and in_drawdown:
                if value < valley_value:
                    valley_date = date
                    valley_value = value
            elif value >= 0 and in_drawdown:
                drawdown_periods.append({
                    'start_date': start_date,
                    'valley_date': valley_date,
                    'recovery_date': date,
                    'depth': valley_value,
                    'length': (date - start_date).days,
                    'recovery': (date - valley_date).days
                })
                in_drawdown = False

        if in_drawdown:
            drawdown_periods.append({
                'start_date': start_date,
                'valley_date': valley_date,
                'recovery_date': None,
                'depth': valley_value,
                'length': (returns.index[-1] - start_date).days,
                'recovery': None
            })

        dd_df = pd.DataFrame(drawdown_periods)
        if not dd_df.empty:
            dd_df = dd_df.sort_values('depth')

        return dd_df

    @staticmethod
    def calculate_underwater_series(returns: pd.Series) -> pd.Series:
        """
        Calculates a series of underwater values for visualization

        Args:
            returns: Returns Series

        Returns:
            A series of underwater meanings
        """
        if returns.empty:
            return pd.Series()

        # Calculate cumulative yield
        cum_returns = (1 + returns).cumprod()

        # Find peaks (maximums)
        peak = cum_returns.cummax()

        # Calculating drawdowns
        underwater = (cum_returns / peak - 1)

        return underwater