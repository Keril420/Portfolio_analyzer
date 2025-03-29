# tests/test_risk_management.py

import unittest
import pandas as pd
import numpy as np
import sys
import os
import datetime

# Добавляем путь к директории src для импорта
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from utils.risk_managemen import RiskManagement


class TestRiskManagement(unittest.TestCase):

    def setUp(self):
        """Set up test data"""
        # Create sample returns data
        dates = pd.date_range(start='2020-01-01', end='2021-01-01', freq='B')
        np.random.seed(42)  # For reproducibility

        # Create returns for portfolio and assets
        portfolio_returns = np.random.normal(0.001, 0.015, size=len(dates))
        asset1_returns = np.random.normal(0.001, 0.02, size=len(dates))
        asset2_returns = np.random.normal(0.0005, 0.01, size=len(dates))

        self.portfolio_returns = pd.Series(portfolio_returns, index=dates)
        self.asset_returns = pd.DataFrame({
            'ASSET1': asset1_returns,
            'ASSET2': asset2_returns
        }, index=dates)

        self.weights = {'ASSET1': 0.6, 'ASSET2': 0.4}

    def test_calculate_var_parametric(self):
        """Test parametric VaR calculation"""
        # Calculate parametric VaR
        var_95 = RiskManagement.calculate_var_parametric(self.portfolio_returns, confidence_level=0.95)

        # Check that VaR is positive (represents a loss)
        self.assertGreater(var_95, 0)

        # Lower confidence should lead to smaller VaR
        var_90 = RiskManagement.calculate_var_parametric(self.portfolio_returns, confidence_level=0.90)
        self.assertLess(var_90, var_95)

    def test_calculate_var_historical(self):
        """Test historical VaR calculation"""
        # Calculate historical VaR
        var_95 = RiskManagement.calculate_var_historical(self.portfolio_returns, confidence_level=0.95)

        # Check that VaR is positive
        self.assertGreater(var_95, 0)

        # Manual calculation
        percentile_5 = np.percentile(self.portfolio_returns, 5)
        manual_var = -percentile_5

        # Verify the value (adjusting for time horizon differences)
        self.assertAlmostEqual(var_95, manual_var, places=10)

    def test_calculate_var_monte_carlo(self):
        """Test Monte Carlo VaR calculation"""
        # Calculate Monte Carlo VaR
        var_95 = RiskManagement.calculate_var_monte_carlo(
            self.portfolio_returns, confidence_level=0.95, simulations=1000
        )

        # Check that VaR is positive
        self.assertGreater(var_95, 0)

        # Increasing simulations should converge to a stable value
        var_95_more_sims = RiskManagement.calculate_var_monte_carlo(
            self.portfolio_returns, confidence_level=0.95, simulations=5000
        )

        # Values should be relatively close
        self.assertAlmostEqual(var_95, var_95_more_sims, delta=0.005)

    def test_calculate_cvar(self):
        """Test CVaR calculation"""
        # Calculate CVaR
        cvar_95 = RiskManagement.calculate_cvar(self.portfolio_returns, confidence_level=0.95)

        # Check that CVaR is positive
        self.assertGreater(cvar_95, 0)

        # CVaR should be greater than or equal to VaR
        var_95 = RiskManagement.calculate_var_historical(self.portfolio_returns, confidence_level=0.95)
        self.assertGreaterEqual(cvar_95, var_95)

    def test_perform_stress_test(self):
        """Test stress testing"""
        # Perform stress test
        result = RiskManagement.perform_stress_test(
            self.portfolio_returns, scenario='financial_crisis_2008', portfolio_value=10000
        )

        # Check result structure
        self.assertIn('scenario', result)
        self.assertIn('portfolio_value', result)
        self.assertIn('portfolio_loss', result)
        self.assertIn('portfolio_after_shock', result)

        # Financial crisis should lead to significant loss
        self.assertLess(result['portfolio_after_shock'], result['portfolio_value'])

    def test_calculate_risk_contribution(self):
        """Test risk contribution calculation"""
        # Calculate risk contribution
        risk_contrib = RiskManagement.calculate_risk_contribution(self.asset_returns, self.weights)

        # Check result structure
        self.assertIn('ASSET1', risk_contrib)
        self.assertIn('ASSET2', risk_contrib)

        # Contributions should sum to approximately 1
        total_contrib = sum(risk_contrib.values())
        self.assertAlmostEqual(total_contrib, 1.0, places=10)

        # Higher volatility asset should have higher risk contribution
        self.assertGreater(risk_contrib['ASSET1'], risk_contrib['ASSET2'])


if __name__ == "__main__":
    unittest.main()