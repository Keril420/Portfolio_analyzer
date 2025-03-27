# tests/test_calculations.py

import unittest
import pandas as pd
import numpy as np
import sys
import os
import datetime

# Добавляем корневую директорию проекта в путь для импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import PortfolioAnalytics


class TestPortfolioAnalytics(unittest.TestCase):

    def setUp(self):
        """Set up test data"""
        # Create sample returns data
        dates = pd.date_range(start='2020-01-01', end='2021-01-01', freq='B')
        np.random.seed(42)  # For reproducibility

        # Create returns for two assets
        asset1_returns = np.random.normal(0.001, 0.02, size=len(dates))
        asset2_returns = np.random.normal(0.0005, 0.01, size=len(dates))

        self.returns_df = pd.DataFrame({
            'ASSET1': asset1_returns,
            'ASSET2': asset2_returns
        }, index=dates)

        # Create portfolio returns (simple weighted average)
        self.weights = {'ASSET1': 0.6, 'ASSET2': 0.4}
        self.portfolio_returns = self.returns_df['ASSET1'] * 0.6 + self.returns_df['ASSET2'] * 0.4

    def test_calculate_returns(self):
        """Test returns calculation"""
        # Create price data from returns
        prices = (1 + self.returns_df).cumprod()

        # Calculate returns using the function
        calculated_returns = PortfolioAnalytics.calculate_returns(prices)

        # Check result dimensions
        # Примечание: размерность вычисленных доходностей может быть на 1 меньше
        # из-за потери первого значения при вычислении процентных изменений
        self.assertEqual(calculated_returns.shape[1], self.returns_df.shape[1])  # Проверяем только количество столбцов

        # Выравниваем индексы для сравнения
        common_index = calculated_returns.dropna().index.intersection(self.returns_df.index)
        calculated_subset = calculated_returns.loc[common_index]
        original_subset = self.returns_df.loc[common_index]

        # Check values (allowing for small numerical differences)
        pd.testing.assert_frame_equal(
            calculated_subset,
            original_subset,
            check_exact=False,
            rtol=1e-10
        )

    def test_calculate_portfolio_return(self):
        """Test portfolio returns calculation"""
        # Calculate portfolio return
        calculated_portfolio_returns = PortfolioAnalytics.calculate_portfolio_return(
            self.returns_df, self.weights
        )

        # Check result dimensions
        self.assertEqual(len(calculated_portfolio_returns), len(self.portfolio_returns))

        # Check values
        np.testing.assert_array_almost_equal(
            calculated_portfolio_returns.values,
            self.portfolio_returns.values,
            decimal=10
        )

    def test_calculate_cumulative_returns(self):
        """Test cumulative returns calculation"""
        # Calculate cumulative returns
        cumulative_returns = PortfolioAnalytics.calculate_cumulative_returns(self.portfolio_returns)

        # Manual calculation
        manual_cumulative = (1 + self.portfolio_returns).cumprod() - 1

        # Check result dimensions
        self.assertEqual(len(cumulative_returns), len(manual_cumulative))

        # Check values
        np.testing.assert_array_almost_equal(
            cumulative_returns.values,
            manual_cumulative.values,
            decimal=10
        )

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        # Calculate Sharpe ratio
        sharpe = PortfolioAnalytics.calculate_sharpe_ratio(self.portfolio_returns, risk_free_rate=0.0)

        # Manual calculation
        returns_mean = self.portfolio_returns.mean() * 252  # Annualized
        returns_std = self.portfolio_returns.std() * np.sqrt(252)  # Annualized
        manual_sharpe = returns_mean / returns_std

        # Check result
        self.assertAlmostEqual(sharpe, manual_sharpe, places=10)

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation"""
        # Calculate max drawdown
        max_dd = PortfolioAnalytics.calculate_max_drawdown(self.portfolio_returns)

        # Manual calculation
        cumulative = (1 + self.portfolio_returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative / peak) - 1
        manual_max_dd = abs(drawdown.min())

        # Check result
        self.assertAlmostEqual(max_dd, manual_max_dd, places=10)

    def test_calculate_portfolio_metrics(self):
        """Test portfolio metrics calculation"""
        # Calculate portfolio metrics
        metrics = PortfolioAnalytics.calculate_portfolio_metrics(self.portfolio_returns)

        # Check that key metrics are present
        self.assertIn('total_return', metrics)
        self.assertIn('annualized_return', metrics)
        self.assertIn('volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)

        # Check calculation of total return
        manual_total_return = (1 + self.portfolio_returns).prod() - 1
        self.assertAlmostEqual(metrics['total_return'], manual_total_return, places=10)


if __name__ == "__main__":
    unittest.main()