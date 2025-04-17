import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import io
import base64

class PortfolioVisualization:
    """
    Class for creating visualizations of portfolio data and analysis.
    """

    @staticmethod
    def returns_chart(returns: pd.Series, title: str = "Portfolio Returns",
                      figsize: Tuple[int, int] = (10, 6)) -> str:
        """
        Create a line chart of portfolio returns

        Args:
            returns: Series with portfolio returns
            title: Chart title
            figsize: Figure size (width, height)

        Returns:
            Base64 encoded image
        """
        fig, ax = plt.subplots(figsize=figsize)
        returns.plot(ax=ax)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Returns")
        ax.grid(True, alpha=0.3)

        return PortfolioVisualization._fig_to_base64(fig)

    @staticmethod
    def cumulative_returns_chart(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                                 title: str = "Cumulative Returns", figsize: Tuple[int, int] = (10, 6)) -> str:
        """
        Create a line chart of cumulative returns

        Args:
            returns: Series with portfolio returns
            benchmark_returns: Series with benchmark returns
            title: Chart title
            figsize: Figure size (width, height)

        Returns:
            Base64 encoded image
        """
        cumulative_returns = (1 + returns).cumprod() - 1

        fig, ax = plt.subplots(figsize=figsize)
        cumulative_returns.plot(ax=ax, label="Portfolio")

        if benchmark_returns is not None:
            # Align benchmark returns with portfolio returns
            aligned_benchmark = benchmark_returns.reindex(returns.index, method='ffill')
            benchmark_cumulative = (1 + aligned_benchmark).cumprod() - 1
            benchmark_cumulative.plot(ax=ax, label="Benchmark")

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return PortfolioVisualization._fig_to_base64(fig)

    @staticmethod
    def drawdown_chart(returns: pd.Series, title: str = "Portfolio Drawdown",
                       figsize: Tuple[int, int] = (10, 6)) -> str:
        """
        Create a drawdown chart

        Args:
            returns: Series with portfolio returns
            title: Chart title
            figsize: Figure size (width, height)

        Returns:
            Base64 encoded image
        """
        cumulative_returns = (1 + returns).cumprod()
        peak_values = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / peak_values) - 1

        fig, ax = plt.subplots(figsize=figsize)
        drawdowns.plot(ax=ax)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.grid(True, alpha=0.3)
        ax.fill_between(drawdowns.index, drawdowns.values, 0, color='red', alpha=0.3)

        return PortfolioVisualization._fig_to_base64(fig)

    @staticmethod
    def asset_allocation_pie(weights: Dict[str, float], title: str = "Asset Allocation",
                             figsize: Tuple[int, int] = (10, 6)) -> str:
        """
        Create a pie chart showing asset allocation

        Args:
            weights: Dictionary with asset weights {ticker: weight}
            title: Chart title
            figsize: Figure size (width, height)

        Returns:
            Base64 encoded image
        """
        fig, ax = plt.subplots(figsize=figsize)

        labels = list(weights.keys())
        sizes = [weights[ticker] * 100 for ticker in labels]

        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_title(title)

        return PortfolioVisualization._fig_to_base64(fig)

    @staticmethod
    def correlation_heatmap(returns: pd.DataFrame, title: str = "Correlation Matrix",
                            figsize: Tuple[int, int] = (10, 8)) -> str:
        """
        Create a heatmap of asset correlations

        Args:
            returns: DataFrame with asset returns
            title: Chart title
            figsize: Figure size (width, height)

        Returns:
            Base64 encoded image
        """
        corr_matrix = returns.corr()

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, center=0, fmt='.2f')

        ax.set_title(title)

        return PortfolioVisualization._fig_to_base64(fig)

    @staticmethod
    def efficient_frontier(efficient_frontier: List[Dict], current_portfolio: Optional[Dict] = None,
                           title: str = "Efficient Frontier", figsize: Tuple[int, int] = (10, 6)) -> str:
        """
        Create an efficient frontier chart

        Args:
            efficient_frontier: List of dictionaries with 'risk' and 'return' values
            current_portfolio: Dictionary with 'risk' and 'return' values for current portfolio
            title: Chart title
            figsize: Figure size (width, height)

        Returns:
            Base64 encoded image
        """
        risks = [point['risk'] for point in efficient_frontier]
        returns = [point['return'] for point in efficient_frontier]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(risks, returns, 'b-', linewidth=2)

        # Mark the maximum Sharpe ratio portfolio
        if 'sharpe' in efficient_frontier[0]:
            sharpe_ratios = [point.get('sharpe', 0) for point in efficient_frontier]
            max_sharpe_idx = sharpe_ratios.index(max(sharpe_ratios))
            ax.scatter(risks[max_sharpe_idx], returns[max_sharpe_idx], marker='*', color='r', s=150,
                       label='Maximum Sharpe Ratio')

        # Mark the minimum variance portfolio
        min_risk_idx = risks.index(min(risks))
        ax.scatter(risks[min_risk_idx], returns[min_risk_idx], marker='o', color='g', s=150,
                   label='Minimum Variance')

        # Mark the current portfolio
        if current_portfolio:
            ax.scatter(current_portfolio['risk'], current_portfolio['return'], marker='D', color='purple', s=150,
                       label='Current Portfolio')

        ax.set_title(title)
        ax.set_xlabel('Annualized Risk (Volatility)')
        ax.set_ylabel('Annualized Return')
        ax.grid(True, alpha=0.3)
        ax.legend()

        return PortfolioVisualization._fig_to_base64(fig)

    @staticmethod
    def monte_carlo_simulation(simulation_results: np.ndarray, initial_value: float,
                               title: str = "Monte Carlo Simulation", figsize: Tuple[int, int] = (10, 6)) -> str:
        """
        Create a chart visualizing Monte Carlo simulation results

        Args:
            simulation_results: 2D array with simulation results [simulations, time_periods]
            initial_value: Initial portfolio value
            title: Chart title
            figsize: Figure size (width, height)

        Returns:
            Base64 encoded image
        """
        num_simulations, num_periods = simulation_results.shape

        fig, ax = plt.subplots(figsize=figsize)

        # Plot a subset of simulations (for better performance)
        max_plots = 100
        step = max(1, num_simulations // max_plots)

        for i in range(0, num_simulations, step):
            ax.plot(simulation_results[i], 'b-', alpha=0.1)

        # Plot percentiles
        percentiles = [10, 50, 90]
        percentile_data = np.percentile(simulation_results, percentiles, axis=0)

        for i, p in enumerate(percentiles):
            ax.plot(percentile_data[i], linewidth=2, label=f'{p}th Percentile')

        # Add initial value
        ax.axhline(y=initial_value, color='r', linestyle='--', label='Initial Value')

        ax.set_title(title)
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Portfolio Value')
        ax.grid(True, alpha=0.3)
        ax.legend()

        return PortfolioVisualization._fig_to_base64(fig)

    @staticmethod
    def risk_contribution_bar(risk_contribution: Dict[str, float], title: str = "Risk Contribution",
                              figsize: Tuple[int, int] = (10, 6)) -> str:
        """
        Create a bar chart showing risk contribution by asset

        Args:
            risk_contribution: Dictionary with risk contribution percentages {ticker: contribution}
            title: Chart title
            figsize: Figure size (width, height)

        Returns:
            Base64 encoded image
        """
        fig, ax = plt.subplots(figsize=figsize)

        tickers = list(risk_contribution.keys())
        contributions = [risk_contribution[ticker] * 100 for ticker in tickers]

        ax.bar(tickers, contributions)

        ax.set_title(title)
        ax.set_xlabel('Assets')
        ax.set_ylabel('Risk Contribution (%)')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on top of bars
        for i, v in enumerate(contributions):
            ax.text(i, v + 0.5, f'{v:.1f}%', ha='center')

        # Rotate x-axis labels if there are many assets
        if len(tickers) > 5:
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        return PortfolioVisualization._fig_to_base64(fig)

    @staticmethod
    def _fig_to_base64(fig):
        """Convert matplotlib figure to base64 encoded string"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str


    @staticmethod
    def create_monthly_returns_heatmap(returns: pd.Series) -> pd.DataFrame:
        """
        Creates a table of monthly returns for visualization as a heat map

        Args:
            returns: Income Series

        Returns:
            DataFrame of monthly returns indexed by year and month
        """
        if returns.empty or not isinstance(returns.index, pd.DatetimeIndex):
            return pd.DataFrame()

        # Calculate monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Create a pivot table
        monthly_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })

        # Create a pivot table with years in rows and months in columns
        heatmap_data = monthly_df.pivot(index='year', columns='month', values='return')

        # Rename columns to month names
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        heatmap_data = heatmap_data.rename(columns=month_names)

        # Add a column with annual yield
        annual_returns = returns.resample('A').apply(lambda x: (1 + x).prod() - 1)
        annual_returns.index = annual_returns.index.year

        years_in_heatmap = heatmap_data.index.tolist()
        annual_returns = annual_returns[annual_returns.index.isin(years_in_heatmap)]

        if not annual_returns.empty:
            heatmap_data['Год'] = annual_returns.values

        # Calculating reasonable boundaries for the color scale
        # Using percentiles instead of min/max to eliminate outliers
        flat_data = heatmap_data.values.flatten()
        flat_data = flat_data[~np.isnan(flat_data)]

        if len(flat_data) > 0:
            vmin = np.percentile(flat_data, 5)  # 5th percentile
            vmax = np.percentile(flat_data, 95)  # 95th percentile

            # Balancing the scale for better perception
            abs_max = max(abs(vmin), abs(vmax))
            heatmap_data.attrs['vmin'] = -abs_max
            heatmap_data.attrs['vmax'] = abs_max

        return heatmap_data

    @staticmethod
    def create_worst_periods_table(returns: pd.Series, benchmark_returns: pd.Series) -> pd.DataFrame:
        """
        Creates a table of worst periods relative to the benchmark

        Args:
            returns: Portfolio return series
            benchmark_returns: Benchmark return series

            Returns:
                DataFrame with worst-period information
        """
        if returns.empty or benchmark_returns.empty:
            return pd.DataFrame()

        # Normalize time zones in indexes
        returns = returns.copy()
        benchmark_returns = benchmark_returns.copy()

        if returns.index.tz is not None:
            returns.index = returns.index.tz_localize(None)

        if benchmark_returns.index.tz is not None:
            benchmark_returns.index = benchmark_returns.index.tz_localize(None)

        # Aligning series
        common_index = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        # Calculating cumulative returns
        cum_returns = (1 + returns).cumprod()
        cum_benchmark = (1 + benchmark_returns).cumprod()

        #Calculating relative performance
        relative_performance = cum_returns / cum_benchmark

        # We define periods for analysis (3 months, 6 months, 1 year)
        periods = {
            '3 месяца': 63,  # ~63 trading days
            '6 месяцев': 126,  # ~126 trading days
            '1 год': 252  # ~252 trading days
        }

        worst_periods = []

        for period_name, days in periods.items():
            if len(relative_performance) >= days:
                try:
                    # Calculate the sliding change in relative performance
                    rolling_change = relative_performance.pct_change(days).dropna()

                    # Find the worst period
                    worst_date = rolling_change.idxmin()
                    worst_change = rolling_change.loc[worst_date]

                    # Find the starting date for the worst period
                    start_date = worst_date - pd.Timedelta(days=days)

                    # Check if the start date exists in the index
                    if start_date not in cum_returns.index:
                        # Find the nearest available date
                        available_dates = cum_returns.index[cum_returns.index <= start_date]
                        if len(available_dates) > 0:
                            start_date = available_dates[-1]
                        else:
                            continue

                    # We calculate the yield for this period
                    period_return = (cum_returns.loc[worst_date] / cum_returns.loc[start_date]) - 1
                    period_benchmark = (cum_benchmark.loc[worst_date] / cum_benchmark.loc[start_date]) - 1

                    worst_periods.append({
                        'Period': period_name,
                        'Start': start_date.strftime('%Y-%m-%d'),
                        'End': worst_date.strftime('%Y-%m-%d'),
                        'Return': period_return,
                        'Benchmark': period_benchmark,
                        'Difference': period_return - period_benchmark
                    })
                except Exception as e:
                    continue  # Skip period on error

        return pd.DataFrame(worst_periods)