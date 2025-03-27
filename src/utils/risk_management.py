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

    @staticmethod
    def perform_stress_test(returns: pd.Series, scenario: str, portfolio_value: float = 10000) -> Dict:
        """
        Perform stress testing on a portfolio using historical scenarios

        Args:
            returns: Series with portfolio returns
            scenario: Stress scenario ('financial_crisis_2008', 'covid_2020', 'tech_bubble_2000', etc.)
            portfolio_value: Current portfolio value

        Returns:
            Dictionary with stress test results
        """
        if returns.empty:
            return {'error': 'No returns data provided'}

        # Define historical stress scenarios (percentage losses)
        scenarios = {
            'financial_crisis_2008': -0.50,  # 50% drop during 2008 financial crisis
            'covid_2020': -0.35,  # 35% drop during COVID-19 crash
            'tech_bubble_2000': -0.45,  # 45% drop during dot-com bubble
            'black_monday_1987': -0.22,  # 22% drop on Black Monday
            'inflation_shock': -0.15,  # 15% drop during inflation shock
            'rate_hike': -0.10,  # 10% drop during aggressive rate hike
            'moderate_recession': -0.25,  # 25% drop during moderate recession
            'severe_recession': -0.40  # 40% drop during severe recession
        }

        # Get the scenario shock percentage
        if scenario not in scenarios:
            return {'error': f'Unknown scenario: {scenario}'}

        shock_percentage = scenarios[scenario]

        # Calculate portfolio value after shock
        portfolio_loss = portfolio_value * shock_percentage
        portfolio_after_shock = portfolio_value + portfolio_loss

        # Calculate the number of standard deviations of the shock
        daily_std = returns.std()
        annual_std = daily_std * np.sqrt(252)
        std_deviations = shock_percentage / annual_std

        # Calculate recovery time (assuming mean return recovers the loss)
        mean_daily_return = returns.mean()
        if mean_daily_return > 0:
            recovery_days = -np.log(1 + shock_percentage) / np.log(1 + mean_daily_return)
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
            'recovery_months': recovery_months
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