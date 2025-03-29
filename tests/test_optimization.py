# tests/test_optimization.py

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Добавляем путь к директории src для импорта
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from utils.optimization import PortfolioOptimizer


class TestPortfolioOptimizer(unittest.TestCase):

    def setUp(self):
        """Set up test data"""
        # Create sample returns data
        dates = pd.date_range(start='2020-01-01', end='2021-01-01', freq='B')
        np.random.seed(42)  # For reproducibility

        # Create returns for three assets with different characteristics
        asset1_returns = np.random.normal(0.001, 0.02, size=len(dates))  # Higher risk, higher return
        asset2_returns = np.random.normal(0.0005, 0.01, size=len(dates))  # Lower risk, lower return
        asset3_returns = np.random.normal(0.0008, 0.015, size=len(dates))  # Medium risk, medium return

        self.returns_df = pd.DataFrame({
            'ASSET1': asset1_returns,
            'ASSET2': asset2_returns,
            'ASSET3': asset3_returns
        }, index=dates)

    def test_minimum_variance_optimization(self):
        """Test minimum variance optimization"""
        # Run optimization
        result = PortfolioOptimizer.minimum_variance_optimization(self.returns_df)

        # Check result structure
        self.assertIn('optimal_weights', result)
        self.assertIn('expected_return', result)
        self.assertIn('expected_risk', result)

        # Weights should sum to 1
        weights_sum = sum(result['optimal_weights'].values())
        self.assertAlmostEqual(weights_sum, 1.0, places=10)

        # Lower risk asset should have higher weight
        self.assertGreater(result['optimal_weights']['ASSET2'], result['optimal_weights']['ASSET1'])

    def test_maximum_sharpe_optimization(self):
        """Test maximum Sharpe ratio optimization"""
        # Run optimization
        result = PortfolioOptimizer.maximum_sharpe_optimization(self.returns_df, risk_free_rate=0.0)

        # Check result structure
        self.assertIn('optimal_weights', result)
        self.assertIn('expected_return', result)
        self.assertIn('expected_risk', result)
        self.assertIn('sharpe_ratio', result)

        # Weights should sum to 1
        weights_sum = sum(result['optimal_weights'].values())
        self.assertAlmostEqual(weights_sum, 1.0, places=10)

        # Sharpe ratio should be positive
        self.assertGreater(result['sharpe_ratio'], 0)

    def test_equal_weight_optimization(self):
        """Test equal weight optimization"""
        # Run optimization
        result = PortfolioOptimizer.equal_weight_optimization(self.returns_df)

        # Check result structure
        self.assertIn('optimal_weights', result)
        self.assertIn('expected_return', result)
        self.assertIn('expected_risk', result)

        # All weights should be equal
        for asset, weight in result['optimal_weights'].items():
            self.assertAlmostEqual(weight, 1 / 3, places=10)

    def test_markowitz_optimization(self):
        """Test Markowitz optimization"""
        # Run optimization with target return
        target_return = 0.2  # Annual
        result = PortfolioOptimizer.markowitz_optimization(
            self.returns_df, risk_free_rate=0.0, target_return=target_return
        )

        # Check result structure
        self.assertIn('optimal_weights', result)
        self.assertIn('expected_return', result)
        self.assertIn('expected_risk', result)
        self.assertIn('efficient_frontier', result)

        # Weights should sum to 1
        weights_sum = sum(result['optimal_weights'].values())
        self.assertAlmostEqual(weights_sum, 1.0, places=10)

        # Check if target return is achieved (approximately)
        self.assertAlmostEqual(result['expected_return'], target_return, delta=0.01)

    def test_risk_parity_optimization(self):
        """Test risk parity optimization"""
        # Run optimization
        result = PortfolioOptimizer.risk_parity_optimization(self.returns_df)

        # Check result structure
        self.assertIn('optimal_weights', result)
        self.assertIn('expected_return', result)
        self.assertIn('expected_risk', result)
        self.assertIn('risk_contribution', result)

        # Weights should sum to 1
        weights_sum = sum(result['optimal_weights'].values())
        self.assertAlmostEqual(weights_sum, 1.0, places=10)

        # Risk contributions should be approximately equal
        rc_values = list(result['risk_contribution'].values())
        std_dev = np.std(rc_values)
        self.assertLess(std_dev, 0.05)  # Risk contributions should be similar


if __name__ == "__main__":
    unittest.main()