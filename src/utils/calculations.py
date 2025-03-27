# src/utils/calculations.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('calculations')


class PortfolioAnalytics:
    """
    Class for portfolio analytics and performance calculations.
    Provides methods for calculating returns, risk metrics, performance ratios, etc.
    """

    @staticmethod
    def calculate_returns(prices: pd.DataFrame, period: str = 'daily') -> pd.DataFrame:
        """
        Calculate returns for the specified period

        Args:
            prices: DataFrame with price data
            period: Period for returns calculation ('daily', 'weekly', 'monthly', 'annual')

        Returns:
            DataFrame with returns data
        """
        if prices.empty:
            return pd.DataFrame()

        if period == 'daily':
            return prices.pct_change().dropna()
        elif period == 'weekly':
            return prices.resample('W').last().pct_change().dropna()
        elif period == 'monthly':
            return prices.resample('M').last().pct_change().dropna()
        elif period == 'annual':
            return prices.resample('Y').last().pct_change().dropna()
        else:
            logger.warning(f"Unsupported period: {period}, using 'daily' as default")
            return prices.pct_change().dropna()

    @staticmethod
    def calculate_portfolio_return(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """
        Calculate weighted portfolio returns

        Args:
            returns: DataFrame with returns for each asset
            weights: Dictionary with asset weights {ticker: weight}

        Returns:
            Series with portfolio returns
        """
        if returns.empty or not weights:
            return pd.Series()

        # Filter returns to include only assets in weights
        assets_in_weights = list(weights.keys())
        filtered_returns = returns[assets_in_weights].copy()

        # Calculate portfolio returns
        portfolio_weights = np.array([weights.get(ticker, 0) for ticker in filtered_returns.columns])
        portfolio_returns = filtered_returns.dot(portfolio_weights)

        return portfolio_returns

    @staticmethod
    def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
        """
        Calculate cumulative returns

        Args:
            returns: Series with returns data

        Returns:
            Series with cumulative returns
        """
        if returns.empty:
            return pd.Series()

        cumulative_returns = (1 + returns).cumprod() - 1
        return cumulative_returns

    @staticmethod
    def calculate_annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized return

        Args:
            returns: Series with returns data
            periods_per_year: Number of periods in a year (252 for daily, 52 for weekly, 12 for monthly)

        Returns:
            Annualized return as a float
        """
        if returns.empty:
            return 0.0

        total_return = (1 + returns).prod() - 1
        years = len(returns) / periods_per_year

        if years <= 0:
            return 0.0

        annualized_return = (1 + total_return) ** (1 / years) - 1
        return annualized_return

    @staticmethod
    def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility

        Args:
            returns: Series with returns data
            periods_per_year: Number of periods in a year

        Returns:
            Annualized volatility as a float
        """
        if returns.empty:
            return 0.0

        volatility = returns.std() * np.sqrt(periods_per_year)
        return volatility

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio

        Args:
            returns: Series with returns data
            risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)
            periods_per_year: Number of periods in a year

        Returns:
            Sharpe ratio as a float
        """
        if returns.empty:
            return 0.0

        # Convert annual risk-free rate to period rate
        period_risk_free = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

        excess_returns = returns - period_risk_free
        annualized_excess_return = excess_returns.mean() * periods_per_year
        annualized_volatility = returns.std() * np.sqrt(periods_per_year)

        if annualized_volatility == 0:
            return 0.0

        sharpe_ratio = annualized_excess_return / annualized_volatility
        return sharpe_ratio

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio

        Args:
            returns: Series with returns data
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year

        Returns:
            Sortino ratio as a float
        """
        if returns.empty:
            return 0.0

        # Convert annual risk-free rate to period rate
        period_risk_free = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

        excess_returns = returns - period_risk_free
        annualized_excess_return = excess_returns.mean() * periods_per_year

        # Calculate downside deviation (only negative returns)
        negative_returns = returns[returns < period_risk_free] - period_risk_free

        if len(negative_returns) == 0:
            return float('inf')  # No negative returns

        downside_deviation = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(periods_per_year)

        if downside_deviation == 0:
            return 0.0

        sortino_ratio = annualized_excess_return / downside_deviation
        return sortino_ratio

    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """
        Calculate maximum drawdown

        Args:
            returns: Series with returns data

        Returns:
            Maximum drawdown as a float (positive value)
        """
        if returns.empty:
            return 0.0

        cumulative_returns = (1 + returns).cumprod()
        peak_values = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / peak_values) - 1
        max_drawdown = drawdowns.min()

        # Return positive value for easier interpretation
        return abs(max_drawdown)

    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio (annualized return / maximum drawdown)

        Args:
            returns: Series with returns data
            periods_per_year: Number of periods in a year

        Returns:
            Calmar ratio as a float
        """
        if returns.empty:
            return 0.0

        annualized_return = PortfolioAnalytics.calculate_annualized_return(returns, periods_per_year)
        max_drawdown = PortfolioAnalytics.calculate_max_drawdown(returns)

        if max_drawdown == 0:
            return 0.0

        calmar_ratio = annualized_return / max_drawdown
        return calmar_ratio

    @staticmethod
    def calculate_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate beta (systematic risk) relative to a benchmark

        Args:
            returns: Series with portfolio returns
            benchmark_returns: Series with benchmark returns

        Returns:
            Beta as a float
        """
        if returns.empty or benchmark_returns.empty:
            return 0.0

        # Align the series
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) == 0:
            return 0.0

        returns = returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        # Calculate beta using covariance and variance
        covariance = returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()

        if benchmark_variance == 0:
            return 0.0

        beta = covariance / benchmark_variance
        return beta

    @staticmethod
    def calculate_alpha(returns: pd.Series, benchmark_returns: pd.Series,
                        risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate Jensen's alpha

        Args:
            returns: Series with portfolio returns
            benchmark_returns: Series with benchmark returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year

        Returns:
            Alpha as a float
        """
        if returns.empty or benchmark_returns.empty:
            return 0.0

        # Convert annual risk-free rate to period rate
        period_risk_free = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

        # Align the series
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) == 0:
            return 0.0

        returns = returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        # Calculate beta
        beta = PortfolioAnalytics.calculate_beta(returns, benchmark_returns)

        # Calculate alpha (annualized)
        alpha = (returns.mean() - period_risk_free - beta * (
                    benchmark_returns.mean() - period_risk_free)) * periods_per_year
        return alpha

    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) using historical method

        Args:
            returns: Series with returns data
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR as a positive float (loss)
        """
        if returns.empty:
            return 0.0

        # VaR is a percentile of the returns distribution
        var = -np.percentile(returns, 100 * (1 - confidence_level))
        return var

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
        var = PortfolioAnalytics.calculate_var(returns, confidence_level)

        # CVaR is the average of returns that are worse than VaR
        cvar = -returns[returns <= -var].mean()

        if np.isnan(cvar):
            return 0.0

        return cvar

    @staticmethod
    def calculate_portfolio_metrics(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                                    risk_free_rate: float = 0.0, periods_per_year: int = 252) -> Dict:
        """
        Calculate comprehensive portfolio performance metrics

        Args:
            returns: Series with portfolio returns
            benchmark_returns: Series with benchmark returns (optional)
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year

        Returns:
            Dictionary with performance metrics
        """
        metrics = {}

        if returns.empty:
            return metrics

        # Basic return metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = PortfolioAnalytics.calculate_annualized_return(returns, periods_per_year)
        metrics['volatility'] = PortfolioAnalytics.calculate_volatility(returns, periods_per_year)

        # Risk-adjusted performance ratios
        metrics['sharpe_ratio'] = PortfolioAnalytics.calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        metrics['sortino_ratio'] = PortfolioAnalytics.calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
        metrics['max_drawdown'] = PortfolioAnalytics.calculate_max_drawdown(returns)
        metrics['calmar_ratio'] = PortfolioAnalytics.calculate_calmar_ratio(returns, periods_per_year)

        # Risk metrics
        metrics['var_95'] = PortfolioAnalytics.calculate_var(returns, 0.95)
        metrics['cvar_95'] = PortfolioAnalytics.calculate_cvar(returns, 0.95)
        metrics['var_99'] = PortfolioAnalytics.calculate_var(returns, 0.99)
        metrics['cvar_99'] = PortfolioAnalytics.calculate_cvar(returns, 0.99)

        # Benchmark-related metrics
        if benchmark_returns is not None:
            common_index = returns.index.intersection(benchmark_returns.index)
            if len(common_index) > 0:
                aligned_returns = returns.loc[common_index]
                aligned_benchmark = benchmark_returns.loc[common_index]

                metrics['beta'] = PortfolioAnalytics.calculate_beta(aligned_returns, aligned_benchmark)
                metrics['alpha'] = PortfolioAnalytics.calculate_alpha(
                    aligned_returns, aligned_benchmark, risk_free_rate, periods_per_year
                )
                metrics['benchmark_return'] = (1 + aligned_benchmark).prod() - 1
                metrics['tracking_error'] = (aligned_returns - aligned_benchmark).std() * np.sqrt(periods_per_year)

                if metrics['tracking_error'] > 0:
                    metrics['information_ratio'] = (
                            (aligned_returns.mean() - aligned_benchmark.mean()) * periods_per_year / metrics[
                        'tracking_error']
                    )
                else:
                    metrics['information_ratio'] = 0.0

        return metrics