import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Dict, List, Tuple, Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('optimization')


class PortfolioOptimizer:
    """
    Class for portfolio optimization using various methodologies.
    """

    @staticmethod
    def optimize_portfolio(returns: pd.DataFrame, method: str = 'markowitz',
                           risk_free_rate: float = 0.0, **kwargs) -> Dict:
        """
        Optimize portfolio based on selected method

        Args:
            returns: DataFrame with asset returns
            method: Optimization method ('markowitz', 'risk_parity', 'minimum_variance', 'maximum_sharpe', 'equal_weight')
            risk_free_rate: Risk-free rate (annual)
            **kwargs: Additional parameters specific to each method

        Returns:
            Dictionary with optimization results
        """
        if returns.empty:
            return {'error': 'No returns data provided'}

        # Select optimization method
        if method == 'markowitz':
            return PortfolioOptimizer.markowitz_optimization(returns, risk_free_rate, **kwargs)
        elif method == 'risk_parity':
            return PortfolioOptimizer.risk_parity_optimization(returns, **kwargs)
        elif method == 'minimum_variance':
            return PortfolioOptimizer.minimum_variance_optimization(returns, **kwargs)
        elif method == 'maximum_sharpe':
            return PortfolioOptimizer.maximum_sharpe_optimization(returns, risk_free_rate, **kwargs)
        elif method == 'equal_weight':
            return PortfolioOptimizer.equal_weight_optimization(returns, **kwargs)
        else:
            return {'error': f'Unknown optimization method: {method}'}

    @staticmethod
    def markowitz_optimization(returns: pd.DataFrame, risk_free_rate: float = 0.0,
                               target_return: Optional[float] = None, target_risk: Optional[float] = None,
                               min_weight: float = 0.0, max_weight: float = 1.0) -> Dict:
        """
        Perform Markowitz Mean-Variance Optimization

        Args:
            returns: DataFrame with asset returns
            risk_free_rate: Risk-free rate (annual)
            target_return: Target portfolio return (annual)
            target_risk: Target portfolio risk/volatility (annual)
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint

        Returns:
            Dictionary with optimization results
        """
        if returns.empty:
            return {'error': 'No returns data provided'}

        # Number of assets
        n_assets = len(returns.columns)

        # Calculate expected returns (annualized)
        expected_returns = returns.mean() * 252

        # Calculate covariance matrix (annualized)
        cov_matrix = returns.cov() * 252

        # Constraints
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        # Function to calculate portfolio statistics
        def portfolio_stats(weights):
            weights = np.array(weights)
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_return, portfolio_risk

        # Function to minimize for Sharpe Ratio
        def neg_sharpe_ratio(weights):
            portfolio_return, portfolio_risk = portfolio_stats(weights)
            return -(portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0

        # Function to minimize for Portfolio Variance
        def portfolio_variance(weights):
            return portfolio_stats(weights)[1] ** 2

        # Constraint that weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        # Additional constraint for target return (if specified)
        if target_return is not None:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(expected_returns * x) - target_return}
            )

        # Additional constraint for target risk (if specified)
        if target_risk is not None:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))) - target_risk}
            )

        # Initial guess (equal weights)
        init_guess = np.array([1.0 / n_assets] * n_assets)

        # Optimize portfolio
        if target_return is not None:
            # Minimize variance for target return
            result = sco.minimize(portfolio_variance, init_guess, method='SLSQP',
                                  bounds=bounds, constraints=constraints)
        elif target_risk is not None:
            # Maximize return for target risk
            def neg_portfolio_return(weights):
                return -portfolio_stats(weights)[0]

            result = sco.minimize(neg_portfolio_return, init_guess, method='SLSQP',
                                  bounds=bounds, constraints=constraints)
        else:
            # Maximize Sharpe ratio
            result = sco.minimize(neg_sharpe_ratio, init_guess, method='SLSQP',
                                  bounds=bounds, constraints=constraints)

        # Check if optimization was successful
        if not result['success']:
            return {'error': f'Optimization failed: {result["message"]}'}

        # Extract optimal weights
        optimal_weights = result['x']

        # Calculate portfolio statistics
        portfolio_return, portfolio_risk = portfolio_stats(optimal_weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0

        # Generate weights dictionary
        weights_dict = {ticker: weight for ticker, weight in zip(returns.columns, optimal_weights)}

        # Generate efficient frontier
        target_returns = np.linspace(expected_returns.min(), expected_returns.max(), 50)
        efficient_frontier = []

        for target in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(expected_returns * x) - target}
            )

            result = sco.minimize(portfolio_variance, init_guess, method='SLSQP',
                                  bounds=bounds, constraints=constraints)

            if result['success']:
                weights = result['x']
                ret, risk = portfolio_stats(weights)
                efficient_frontier.append({
                    'return': ret,
                    'risk': risk,
                    'sharpe': (ret - risk_free_rate) / risk if risk > 0 else 0
                })

        return {
            'method': 'markowitz',
            'optimal_weights': weights_dict,
            'expected_return': portfolio_return,
            'expected_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'efficient_frontier': efficient_frontier
        }

    @staticmethod
    def risk_parity_optimization(returns: pd.DataFrame, risk_budget: Optional[Dict[str, float]] = None,
                                 min_weight: float = 0.01, max_weight: float = 1.0) -> Dict:
        """
        Perform Risk Parity Optimization

        Args:
            returns: DataFrame with asset returns
            risk_budget: Dictionary with risk allocation for each asset {ticker: allocation}
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint

        Returns:
            Dictionary with optimization results
        """
        if returns.empty:
            return {'error': 'No returns data provided'}

        # Number of assets
        n_assets = len(returns.columns)
        tickers = returns.columns

        # Calculate covariance matrix (annualized)
        cov_matrix = returns.cov() * 252

        # Default risk budget (equal risk)
        if risk_budget is None:
            risk_budget = {ticker: 1.0 / n_assets for ticker in tickers}

        # Ensure all assets have a risk budget
        for ticker in tickers:
            if ticker not in risk_budget:
                risk_budget[ticker] = 0.0

        # Convert risk budget to array
        risk_target = np.array([risk_budget.get(ticker, 0) for ticker in tickers])

        # Normalize risk target to sum to 1
        if sum(risk_target) > 0:
            risk_target = risk_target / sum(risk_target)
        else:
            risk_target = np.ones(n_assets) / n_assets

        # Function to minimize for risk parity
        def risk_parity_objective(weights):
            # Calculate portfolio risk
            weights = np.array(weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            # Calculate risk contribution for each asset
            if portfolio_risk > 0:
                # Marginal contribution to risk
                mcr = np.dot(cov_matrix, weights)

                # Risk contribution
                rc = weights * mcr / portfolio_risk

                # Calculate risk contribution error
                rc_error = sum((rc - risk_target * portfolio_risk) ** 2)
                return rc_error
            else:
                return 1e6  # High penalty for zero risk

        # Constraints
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        # Initial guess (equal weights)
        init_guess = np.array([1.0 / n_assets] * n_assets)

        # Optimize portfolio
        result = sco.minimize(risk_parity_objective, init_guess, method='SLSQP',
                              bounds=bounds, constraints=constraints)

        # Check if optimization was successful
        if not result['success']:
            return {'error': f'Optimization failed: {result["message"]}'}

        # Extract optimal weights
        optimal_weights = result['x']

        # Calculate expected returns (annualized)
        expected_returns = returns.mean() * 252

        # Calculate portfolio statistics
        portfolio_return = np.sum(expected_returns * optimal_weights)
        portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

        # Calculate risk contribution
        if portfolio_risk > 0:
            mcr = np.dot(cov_matrix, optimal_weights)
            rc = optimal_weights * mcr / portfolio_risk
            risk_contribution = {ticker: rc[i] for i, ticker in enumerate(tickers)}
        else:
            risk_contribution = {ticker: 0 for ticker in tickers}

        # Generate weights dictionary
        weights_dict = {ticker: weight for ticker, weight in zip(tickers, optimal_weights)}

        return {
            'method': 'risk_parity',
            'optimal_weights': weights_dict,
            'expected_return': portfolio_return,
            'expected_risk': portfolio_risk,
            'risk_contribution': risk_contribution
        }

    @staticmethod
    def minimum_variance_optimization(returns: pd.DataFrame, min_weight: float = 0.0,
                                      max_weight: float = 1.0) -> Dict:
        """
        Perform Minimum Variance Optimization

        Args:
            returns: DataFrame with asset returns
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint

        Returns:
            Dictionary with optimization results
        """
        if returns.empty:
            return {'error': 'No returns data provided'}

        # Number of assets
        n_assets = len(returns.columns)

        # Calculate covariance matrix (annualized)
        cov_matrix = returns.cov() * 252

        # Function to minimize for Portfolio Variance
        def portfolio_variance(weights):
            weights = np.array(weights)
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Constraints
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        # Initial guess (equal weights)
        init_guess = np.array([1.0 / n_assets] * n_assets)

        # Optimize portfolio
        result = sco.minimize(portfolio_variance, init_guess, method='SLSQP',
                              bounds=bounds, constraints=constraints)

        # Check if optimization was successful
        if not result['success']:
            return {'error': f'Optimization failed: {result["message"]}'}

        # Extract optimal weights
        optimal_weights = result['x']

        # Calculate expected returns (annualized)
        expected_returns = returns.mean() * 252

        # Calculate portfolio statistics
        portfolio_return = np.sum(expected_returns * optimal_weights)
        portfolio_risk = np.sqrt(portfolio_variance(optimal_weights))

        # Generate weights dictionary
        weights_dict = {ticker: weight for ticker, weight in zip(returns.columns, optimal_weights)}

        return {
            'method': 'minimum_variance',
            'optimal_weights': weights_dict,
            'expected_return': portfolio_return,
            'expected_risk': portfolio_risk
        }

    @staticmethod
    def maximum_sharpe_optimization(returns: pd.DataFrame, risk_free_rate: float = 0.0,
                                    min_weight: float = 0.0, max_weight: float = 1.0) -> Dict:
        """
        Perform Maximum Sharpe Ratio Optimization

        Args:
            returns: DataFrame with asset returns
            risk_free_rate: Risk-free rate (annual)
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint

        Returns:
            Dictionary with optimization results
        """
        if returns.empty:
            return {'error': 'No returns data provided'}

        # Number of assets
        n_assets = len(returns.columns)

        # Calculate expected returns (annualized)
        expected_returns = returns.mean() * 252

        # Calculate covariance matrix (annualized)
        cov_matrix = returns.cov() * 252

        # Function to calculate portfolio statistics
        def portfolio_stats(weights):
            weights = np.array(weights)
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_return, portfolio_risk

        # Function to minimize for negative Sharpe Ratio
        def neg_sharpe_ratio(weights):
            portfolio_return, portfolio_risk = portfolio_stats(weights)
            return -(portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0

        # Constraints
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        # Initial guess (equal weights)
        init_guess = np.array([1.0 / n_assets] * n_assets)

        # Optimize portfolio
        result = sco.minimize(neg_sharpe_ratio, init_guess, method='SLSQP',
                              bounds=bounds, constraints=constraints)

        # Check if optimization was successful
        if not result['success']:
            return {'error': f'Optimization failed: {result["message"]}'}

        # Extract optimal weights
        optimal_weights = result['x']

        # Calculate portfolio statistics
        portfolio_return, portfolio_risk = portfolio_stats(optimal_weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0

        # Generate weights dictionary
        weights_dict = {ticker: weight for ticker, weight in zip(returns.columns, optimal_weights)}

        return {
            'method': 'maximum_sharpe',
            'optimal_weights': weights_dict,
            'expected_return': portfolio_return,
            'expected_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio
        }

    @staticmethod
    def equal_weight_optimization(returns: pd.DataFrame) -> Dict:
        """
        Perform Equal Weight 'Optimization' (passive strategy)

        Args:
            returns: DataFrame with asset returns

        Returns:
            Dictionary with optimization results
        """
        if returns.empty:
            return {'error': 'No returns data provided'}

        # Number of assets
        n_assets = len(returns.columns)

        # Equal weights
        optimal_weights = np.array([1.0 / n_assets] * n_assets)

        # Calculate expected returns (annualized)
        expected_returns = returns.mean() * 252

        # Calculate covariance matrix (annualized)
        cov_matrix = returns.cov() * 252

        # Calculate portfolio statistics
        portfolio_return = np.sum(expected_returns * optimal_weights)
        portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

        # Generate weights dictionary
        weights_dict = {ticker: weight for ticker, weight in zip(returns.columns, optimal_weights)}

        return {
            'method': 'equal_weight',
            'optimal_weights': weights_dict,
            'expected_return': portfolio_return,
            'expected_risk': portfolio_risk
        }