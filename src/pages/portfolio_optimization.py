import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.optimization import PortfolioOptimizer
from src.utils.calculations import PortfolioAnalytics
from src.utils.risk_management import RiskManagement
from src.utils.visualization import PortfolioVisualization
import src.config as config


def run(data_fetcher, portfolio_manager):
    """
    Function to display the portfolio optimization page

    Args:
        data_fetcher: DataFetcher instance for loading data
        portfolio_manager: PortfolioDataManager instance for working with portfolios
    """
    st.title("Portfolio optimization")

    # Create tabs for different types of optimization
    tabs = st.tabs([
        "Existing Portfolio",
        "New Portfolio",
        "Tactical Allocation",
        "Monte Carlo Simulation"
    ])

    with tabs[0]:
        optimize_existing_portfolio(data_fetcher, portfolio_manager)

    with tabs[1]:
        optimize_new_portfolio(data_fetcher, portfolio_manager)

    with tabs[2]:
        tactical_allocation(data_fetcher, portfolio_manager)

    with tabs[3]:
        monte_carlo_simulation(data_fetcher, portfolio_manager)


def optimize_existing_portfolio(data_fetcher, portfolio_manager):
    """
    Function for optimizing an existing portfolio

    Args:
        data_fetcher: DataFetcher instance for loading data
        portfolio_manager: PortfolioDataManager instance for working with portfolios
    """
    st.header("Optimization of the existing portfolio")

    # Get a list of portfolios
    portfolios = portfolio_manager.list_portfolios()

    if not portfolios:
        st.info("No portfolios found. Create a portfolio in the 'Create a portfolio' section..")
        return

    # Selecting a portfolio for optimization
    portfolio_names = [p['name'] for p in portfolios]
    selected_portfolio = st.selectbox("Select a portfolio to optimize", portfolio_names)

    if not selected_portfolio:
        return

    # Loading portfolio data
    portfolio_data = portfolio_manager.load_portfolio(selected_portfolio)

    # Display basic information about the portfolio
    st.subheader(f"Portfolio: {portfolio_data['name']}")

    if 'description' in portfolio_data and portfolio_data['description']:
        st.write(portfolio_data['description'])

    # Optimization parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        method = st.selectbox(
            "Optimization method",
            [
                "maximum_sharpe",
                "minimum_variance",
                "risk_parity",
                "markowitz",
                "equal_weight"
            ],
            format_func=lambda x: {
                "maximum_sharpe": "Maximum Sharpe Ratio",
                "minimum_variance": "Minimum Variance",
                "risk_parity": "Risk Parity",
                "markowitz": "Markowitz (Efficient Frontier)",
                "equal_weight": "Equal Weights"
            }.get(x, x)
        )

    with col2:
        start_date = st.date_input(
            "Start date",
            datetime.now() - timedelta(days=365)
        )

    with col3:
        end_date = st.date_input(
            "End date",
            datetime.now()
        )

    # Additional parameters depending on the selected method
    if method == "markowitz":
        col1, col2 = st.columns(2)

        with col1:
            target_return = st.slider(
                "Target annual return (%)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5
            ) / 100

            st.write(f"**Target return:** {target_return * 100:.1f}%")

        with col2:
            # Risk-free bet
            risk_free_rate = st.slider(
                "Risk free rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=config.RISK_FREE_RATE * 100,
                step=0.25
            ) / 100

            st.write(f"**Risk free rate:** {risk_free_rate * 100:.2f}%")
    else:
        # For other methods only risk-free bet
        risk_free_rate = st.slider(
            "Risk free rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=config.RISK_FREE_RATE * 100,
            step=0.25
        ) / 100

        # Set target_return to None for other methods
        target_return = None

    # Scale Limitations
    col1, col2, col3 = st.columns(3)

    with col1:
        # Make the minimum weight adaptive depending on the number of assets
        n_assets = len(portfolio_data['assets'])
        suggested_min_weight = max(0.01, min(0.05, 1.0 / (n_assets * 2)))

        min_weight = st.slider(
            "Minimum asset weight (%)",
            min_value=0.0,
            max_value=min(50.0, 100.0 / n_assets),
            value=suggested_min_weight * 100,
            step=0.5
        ) / 100

    with col2:
        max_weight = st.slider(
            "Maximum weight of assets (%)",
            min_value=max(5.0, 100.0 / n_assets),
            max_value=100.0,
            value=min(30.0, 100.0 / (n_assets / 3)),
            step=5.0
        ) / 100

    with col3:
        # Add a switch for sector restrictions
        use_sector_constraints = st.checkbox("Sector restrictions", value=False)

        if use_sector_constraints:
            # Getting asset sectors
            sectors = {}
            for asset in portfolio_data['assets']:
                if 'sector' in asset and asset['sector'] != 'N/A':
                    sector = asset['sector']
                    if sector not in sectors:
                        sectors[sector] = []
                    sectors[sector].append(asset['ticker'])

            # If sectors are found, show additional settings
            if sectors:
                st.write("Maximum sector weight:")
                for sector, tickers in sectors.items():
                    # Calculate the current weight of the sector
                    current_sector_weight = sum(portfolio_data['assets'][i]['weight']
                                                for i, asset in enumerate(portfolio_data['assets'])
                                                if 'sector' in asset and asset['sector'] == sector)

                    # Set the limit for the sector (default value is current weight + 10%)
                    max_sector_weight = st.slider(
                        f"{sector}",
                        min_value=float(current_sector_weight * 100),
                        max_value=100.0,
                        value=min(current_sector_weight * 100 + 10, 100.0),
                        step=5.0
                    ) / 100

                    # Keep the sector limit
                    sectors[sector] = {'tickers': tickers, 'max_weight': max_sector_weight}

    # Button to start optimization
    if st.button("Optimize portfolio"):

        with st.spinner('Loading historical data...'):
            # We get tickers from the portfolio
            tickers = [asset['ticker'] for asset in portfolio_data['assets']]

            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')

            prices_data = data_fetcher.get_batch_data(tickers, start_date_str, end_date_str)

            if not prices_data or all(df.empty for df in prices_data.values()):
                st.error("Failed to load historical data. Please check tickers or change period..")
                return

            close_prices = pd.DataFrame()

            for ticker, df in prices_data.items():
                if not df.empty:
                    if 'Adj Close' in df.columns:
                        close_prices[ticker] = df['Adj Close']
                    elif 'Close' in df.columns:
                        close_prices[ticker] = df['Close']

            returns = PortfolioAnalytics.calculate_returns(close_prices)

        # Portfolio optimization
        with st.spinner('Portfolio optimization...'):
            # Get current weights from the portfolio
            current_weights = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

            # Performing optimization
            optimization_args = {
                'risk_free_rate': risk_free_rate,
                'min_weight': min_weight,
                'max_weight': max_weight
            }

            if method == "markowitz" and target_return is not None:
                optimization_args['target_return'] = target_return

            if use_sector_constraints and sectors:

                optimization_args['sector_constraints'] = sectors

            optimization_result = PortfolioOptimizer.optimize_portfolio(
                returns, method=method, **optimization_args
            )

            if 'error' in optimization_result:
                st.error(f"Optimization error: {optimization_result['error']}")
                return

        st.subheader("Optimization results")

        # Optimized Portfolio Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Expected annual return",
                f"{optimization_result['expected_return'] * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Expected volatility",
                f"{optimization_result['expected_risk'] * 100:.2f}%"
            )

        with col3:
            if 'sharpe_ratio' in optimization_result:
                st.metric(
                    "Sharpe ratio",
                    f"{optimization_result['sharpe_ratio']:.2f}"
                )

        # Comparison of current and optimal weights
        st.subheader("Comparison of scales")

        # Создаем DataFrame для сравнения
        weights_comparison = pd.DataFrame({
            'Ticker': list(current_weights.keys()),
            'Current weight (%)': [current_weights[ticker] * 100 for ticker in current_weights],
            'Optimal weight (%)': [optimization_result['optimal_weights'].get(ticker, 0) * 100 for ticker in
                                    current_weights]
        })

        # Adding the difference
        weights_comparison['Change (%)'] = weights_comparison['Optimal weight (%)'] - weights_comparison[
            'Current weight (%)']

        # Calculate the absolute change (for the color scale)
        weights_comparison['abs_change'] = abs(weights_comparison['Change (%)'])

        weights_comparison = weights_comparison.sort_values('abs_change', ascending=False)

        st.dataframe(
            weights_comparison[['Ticker', 'Current weight (%)', 'Optimal weight (%)', 'Change (%)']],
            use_container_width=True
        )

        # Visualization of weight changes
        fig_weights = go.Figure()

        fig_weights.add_trace(go.Bar(
            x=weights_comparison['Ticker'],
            y=weights_comparison['Current weight (%)'],
            name='Current weight (%)',
            marker_color='lightgrey'
        ))

        fig_weights.add_trace(go.Bar(
            x=weights_comparison['Ticker'],
            y=weights_comparison['Optimal weight (%)'],
            name='Optimal weight (%)',
            marker_color='royalblue'
        ))

        fig_weights.update_layout(
            title='Comparison of current and optimal weights',
            barmode='group',
            xaxis_title='Asset',
            yaxis_title='Вес (%)',
            legend_title='',
            hovermode='x unified'
        )

        st.plotly_chart(fig_weights, use_container_width=True)

        # Efficient Frontier for Markowitz Method
        if method == "markowitz" and 'efficient_frontier' in optimization_result:
            st.subheader("Efficient frontier")

            ef_df = pd.DataFrame(optimization_result['efficient_frontier'])

            #We calculate the current profitability and risk
            portfolio_return = np.sum([current_weights[ticker] * returns[ticker].mean() * 252
                                       for ticker in current_weights if ticker in returns.columns])

            filtered_returns = returns[[ticker for ticker in current_weights if ticker in returns.columns]]
            cov_matrix = filtered_returns.cov() * 252
            weight_array = np.array([current_weights[ticker] for ticker in filtered_returns.columns])
            portfolio_risk = np.sqrt(np.dot(weight_array.T, np.dot(cov_matrix, weight_array)))

            # Visualization of the efficient frontier
            fig_ef = go.Figure()

            # efficient frontier
            fig_ef.add_trace(go.Scatter(
                x=ef_df['risk'] * 100,
                y=ef_df['return'] * 100,
                mode='lines',
                name='Efficient frontier',
                line=dict(color='blue', width=2)
            ))

            # Current portfolio
            fig_ef.add_trace(go.Scatter(
                x=[portfolio_risk * 100],
                y=[portfolio_return * 100],
                mode='markers',
                name='Current portfolio',
                marker=dict(color='red', size=12, symbol='circle')
            ))

            # Optimal portfolio
            fig_ef.add_trace(go.Scatter(
                x=[optimization_result['expected_risk'] * 100],
                y=[optimization_result['expected_return'] * 100],
                mode='markers',
                name='Optimal portfolio',
                marker=dict(color='green', size=12, symbol='star')
            ))

            # Minimum variance
            min_var_idx = ef_df['risk'].idxmin()

            fig_ef.add_trace(go.Scatter(
                x=[ef_df.iloc[min_var_idx]['risk'] * 100],
                y=[ef_df.iloc[min_var_idx]['return'] * 100],
                mode='markers',
                name='Minimum variance',
                marker=dict(color='purple', size=12, symbol='triangle-up')
            ))

            # Maximum Sharpe
            if 'sharpe' in ef_df.columns:
                max_sharpe_idx = ef_df['sharpe'].idxmax()

                fig_ef.add_trace(go.Scatter(
                    x=[ef_df.iloc[max_sharpe_idx]['risk'] * 100],
                    y=[ef_df.iloc[max_sharpe_idx]['return'] * 100],
                    mode='markers',
                    name='Maximum Sharpe',
                    marker=dict(color='gold', size=12, symbol='diamond')
                ))

            fig_ef.update_layout(
                title='Efficient frontier',
                xaxis_title='Expected risk (%)',
                yaxis_title='Expected return (%)',
                legend_title='',
                hovermode='closest'
            )

            st.plotly_chart(fig_ef, use_container_width=True)

        # Specific visualization for the Risk Parity method
        if method == "risk_parity" and 'risk_contribution' in optimization_result:
            st.subheader("Impact on risk")

            # Create a DataFrame for risk contribution
            risk_contrib_df = pd.DataFrame({
                'Asset': list(optimization_result['risk_contribution'].keys()),
                'Impact on risk (%)': [v * 100 for v in optimization_result['risk_contribution'].values()]
            })

            # Visualization of risk contribution
            fig_rc = px.bar(
                risk_contrib_df,
                x='Asset',
                y='Impact on risk  (%)',
                title='Impact on risk after optimization',
                color='Impact on risk (%)',
                color_continuous_scale='Viridis'
            )

            st.plotly_chart(fig_rc, use_container_width=True)

        # Apply optimization button
        if st.button("Apply optimization to portfolio"):
            # Updating weights in the portfolio
            for asset in portfolio_data['assets']:
                ticker = asset['ticker']
                if ticker in optimization_result['optimal_weights']:
                    asset['weight'] = optimization_result['optimal_weights'][ticker]

            # Save the updated portfolio
            portfolio_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            portfolio_manager.save_portfolio(portfolio_data)

            st.success(f"Portfolio '{selected_portfolio}' has been successfully optimized!")


def optimize_new_portfolio(data_fetcher, portfolio_manager):
    """
    Function for creating and optimizing a new portfolio

    Args:
        data_fetcher: DataFetcher instance for loading data
        portfolio_manager: PortfolioDataManager instance for working with portfolios
    """
    st.header("Creation and optimization of a new portfolio")

    # Enter a name for the new portfolio
    portfolio_name = st.text_input("Name of the new portfolio")
    portfolio_description = st.text_area("Description (optional)")

    # Entering tickers
    st.subheader("Adding assets")
    st.write("Enter asset tickers separated by commas (e.g. AAPL, MSFT, GOOGL)")
    tickers_input = st.text_input("Tickers")

    # Check and search for tickers
    if tickers_input:
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]

        with st.spinner('Checking tickers...'):
            #Checking the validity of tickers
            valid_tickers, invalid_tickers = data_fetcher.validate_tickers(tickers)

            if invalid_tickers:
                st.warning(f"The following tickers were not found.: {', '.join(invalid_tickers)}")

            if not valid_tickers:
                st.error("None of the entered tickers were found. Please check the correctness of the tickers..")
                return

            st.success(f"{len(valid_tickers)} valid tickers found.")

        # Optimization parameters
        st.subheader("Optimization parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            method = st.selectbox(
                "Optimization method",
                [
                    "maximum_sharpe",
                    "minimum_variance",
                    "risk_parity",
                    "markowitz",
                    "equal_weight"
                ],
                format_func=lambda x: {
                    "maximum_sharpe": "Maximum Sharpe Ratio",
                    "minimum_variance": "Minimum Variance",
                    "risk_parity": "Risk Parity",
                    "markowitz": "Markowitz (Efficient Frontier)",
                    "equal_weight": "Equal Weights"
                }.get(x, x),
                key="new_portfolio_method"
            )

        with col2:
            start_date = st.date_input(
                "Start date",
                datetime.now() - timedelta(days=365 * 2),
                key="new_portfolio_start_date"
            )

        with col3:
            end_date = st.date_input(
                "End date",
                datetime.now(),
                key="new_portfolio_end_date"
            )

        # Additional parameters depending on the selected method
        if method == "markowitz":
            col1, col2 = st.columns(2)

            with col1:
                target_return = st.slider(
                    "Target annual return (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=10.0,
                    step=0.5,
                    key="new_portfolio_target_return"
                ) / 100

                st.write(f"**Target return:** {target_return * 100:.1f}%")

            with col2:
                # Risk free rate
                risk_free_rate = st.slider(
                    "Risk free rate (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=config.RISK_FREE_RATE * 100,
                    step=0.25,
                    key="new_portfolio_risk_free_rate"
                ) / 100

                st.write(f"**Risk free rate:** {risk_free_rate * 100:.2f}%")
        else:

            risk_free_rate = st.slider(
                "Risk free rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=config.RISK_FREE_RATE * 100,
                step=0.25,
                key="new_portfolio_risk_free_rate"
            ) / 100


            target_return = None

        col1, col2 = st.columns(2)

        with col1:
            min_weight = st.slider(
                "Minimum asset weight (%)",
                min_value=0.0,
                max_value=50.0,
                value=1.0,
                step=0.5,
                key="new_portfolio_min_weight"
            ) / 100

        with col2:
            max_weight = st.slider(
                "Maximum asset weight (%)",
                min_value=10.0,
                max_value=100.0,
                value=30.0,
                step=5.0,
                key="new_portfolio_max_weight"
            ) / 100

        # Button for creating and optimizing a portfolio
        if st.button("Create and optimize a portfolio") and portfolio_name:

            with st.spinner('Loading historical data...'):

                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')

                prices_data = data_fetcher.get_batch_data(valid_tickers, start_date_str, end_date_str)

                if not prices_data or all(df.empty for df in prices_data.values()):
                    st.error(
                        "Failed to load historical data. Please check tickers or change period..")
                    return

                # Create a DataFrame with closing prices
                close_prices = pd.DataFrame()

                for ticker, df in prices_data.items():
                    if not df.empty:
                        if 'Adj Close' in df.columns:
                            close_prices[ticker] = df['Adj Close']
                        elif 'Close' in df.columns:
                            close_prices[ticker] = df['Close']

                # Calculating profitability
                returns = PortfolioAnalytics.calculate_returns(close_prices)

            # Portfolio optimization
            with st.spinner('Portfolio optimization...'):

                optimization_args = {
                    'risk_free_rate': risk_free_rate,
                    'min_weight': min_weight,
                    'max_weight': max_weight
                }

                # Add target_return for Markowitz method
                if method == "markowitz" and target_return is not None:
                    optimization_args['target_return'] = target_return

                # Launch optimization
                optimization_result = PortfolioOptimizer.optimize_portfolio(
                    returns, method=method, **optimization_args
                )

                # Check the result
                if 'error' in optimization_result:
                    st.error(f"Optimization error: {optimization_result['error']}")
                    return

            # Create a new portfolio
            assets = []
            for ticker, weight in optimization_result['optimal_weights'].items():
                asset = {
                    'ticker': ticker,
                    'weight': weight
                }
                assets.append(asset)

            # Create a portfolio structure
            portfolio_data = {
                'name': portfolio_name,
                'description': portfolio_description,
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'assets': assets
            }

            # Enriching portfolio data with additional information
            portfolio_manager._enrich_portfolio_data(portfolio_data)

            # Save the portfolio
            portfolio_manager.save_portfolio(portfolio_data)

            st.success(f"Portfolio '{portfolio_name}' has been successfully created and optimized!")

            # Displaying optimization results
            st.subheader("Optimization results")

            # Optimized Portfolio Metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Expected annual return",
                    f"{optimization_result['expected_return'] * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Expected volatility",
                    f"{optimization_result['expected_risk'] * 100:.2f}%"
                )

            with col3:
                if 'sharpe_ratio' in optimization_result:
                    st.metric(
                        "Sharpe ratio",
                        f"{optimization_result['sharpe_ratio']:.2f}"
                    )

            # Visualization of portfolio weights
            weights_df = pd.DataFrame({
                'Asset': list(optimization_result['optimal_weights'].keys()),
                'Вес (%)': [w * 100 for w in optimization_result['optimal_weights'].values()]
            }).sort_values('Вес (%)', ascending=False)

            fig_weights = px.bar(
                weights_df,
                x='Asset',
                y='Вес (%)',
                title='Distribution of weights of the optimized portfolio',
                color='Вес (%)',
                color_continuous_scale='Viridis'
            )

            st.plotly_chart(fig_weights, use_container_width=True)

            # Efficient Frontier for Markowitz Method
            if method == "markowitz" and 'efficient_frontier' in optimization_result:
                st.subheader("Efficient frontier")

                # Create a DataFrame for the efficient frontier
                ef_df = pd.DataFrame(optimization_result['efficient_frontier'])

                # Visualization of the efficient frontier
                fig_ef = go.Figure()

                fig_ef.add_trace(go.Scatter(
                    x=ef_df['risk'] * 100,
                    y=ef_df['return'] * 100,
                    mode='lines',
                    name='Efficient frontier',
                    line=dict(color='blue', width=2)
                ))

                # Optimal portfolio
                fig_ef.add_trace(go.Scatter(
                    x=[optimization_result['expected_risk'] * 100],
                    y=[optimization_result['expected_return'] * 100],
                    mode='markers',
                    name='Optimal portfolio',
                    marker=dict(color='green', size=12, symbol='star')
                ))

                # Minimum variance
                min_var_idx = ef_df['risk'].idxmin()

                fig_ef.add_trace(go.Scatter(
                    x=[ef_df.iloc[min_var_idx]['risk'] * 100],
                    y=[ef_df.iloc[min_var_idx]['return'] * 100],
                    mode='markers',
                    name='Minimum variance',
                    marker=dict(color='purple', size=12, symbol='triangle-up')
                ))

                # Maximum Sharpe
                if 'sharpe' in ef_df.columns:
                    max_sharpe_idx = ef_df['sharpe'].idxmax()

                    fig_ef.add_trace(go.Scatter(
                        x=[ef_df.iloc[max_sharpe_idx]['risk'] * 100],
                        y=[ef_df.iloc[max_sharpe_idx]['return'] * 100],
                        mode='markers',
                        name='Maximum Sharpe',
                        marker=dict(color='gold', size=12, symbol='diamond')
                    ))

                fig_ef.update_layout(
                    title='Efficient Frontier',
                    xaxis_title='Expected risk (%)',
                    yaxis_title='Expected return (%)',
                    legend_title='',
                    hovermode='closest'
                )

                st.plotly_chart(fig_ef, use_container_width=True)

            # Specific visualization for the Risk Parity method
            if method == "risk_parity" and 'risk_contribution' in optimization_result:
                st.subheader("Impact on risk")

                risk_contrib_df = pd.DataFrame({
                    'Asset': list(optimization_result['risk_contribution'].keys()),
                    'Impact on risk (%)': [v * 100 for v in optimization_result['risk_contribution'].values()]
                })

                # Visualizing the contribution to risk
                fig_rc = px.bar(
                    risk_contrib_df,
                    x='Asset',
                    y='Impact on risk (%)',
                    title='Impact on risk of an optimized portfolio',
                    color='Impact on risk (%)',
                    color_continuous_scale='Viridis'
                )

                st.plotly_chart(fig_rc, use_container_width=True)


def tactical_allocation(data_fetcher, portfolio_manager):
    """
    Function for tactical asset allocation

    Args:
        data_fetcher: DataFetcher instance for loading data
        portfolio_manager: PortfolioDataManager instance for working with portfolios
    """
    st.header("Tactical Asset Allocation")

    # Get a list of portfolios
    portfolios = portfolio_manager.list_portfolios()

    if not portfolios:
        st.info("No portfolios found. Create a portfolio in the 'Create a portfolio' section.'.")
        return

    # Portfolio Selection for Tactical Allocation
    portfolio_names = [p['name'] for p in portfolios]
    selected_portfolio = st.selectbox("Select a portfolio for tactical allocation", portfolio_names)

    if not selected_portfolio:
        return

    # Loading portfolio data
    portfolio_data = portfolio_manager.load_portfolio(selected_portfolio)

    # Display basic information about the portfolio
    st.subheader(f"Portfolio: {portfolio_data['name']}")
    st.write(f"Number of assets# Tactical Distribution Parameters: {len(portfolio_data['assets'])}")

    # Tactical Distribution Parameters
    col1, col2 = st.columns(2)

    with col1:
        tactic_method = st.selectbox(
            "Tactical distribution method",
            [
                "market_momentum",
                "sector_rotation",
                "volatility_targeting",
                "equal_weight"
            ],
            format_func=lambda x: {
                "market_momentum": "Market Momentum",
                "sector_rotation": "Sector Rotation",
                "volatility_targeting": "Volatility Targeting",
                "equal_weight": "Equal Weights"
            }.get(x, x)
        )

    with col2:
        if tactic_method != "equal_weight":
            adjustment_strength = st.slider(
                "The power of adjustment (%)",
                min_value=0,
                max_value=100,
                value=50,
                step=10
            ) / 100

    # Loading historical data
    st.subheader("Analysis parameters")

    col1, col2 = st.columns(2)

    with col1:
        lookback_period = st.selectbox(
            "Analysis period",
            ["1M", "3M", "6M", "1Y", "2Y"],
            index=2,
            format_func=lambda x: {
                "1M": "1 month",
                "3M": "3 months",
                "6M": "6 months",
                "1Y": "1 year",
                "2Y": "2 years"
            }.get(x, x)
        )

    with col2:
        benchmark = st.selectbox(
            "Benchmark",
            ["SPY", "QQQ", "IWM", "VTI", "None"],
            index=0,
            format_func=lambda x: "Нет" if x == "None" else x
        )
        if benchmark == "None":
            benchmark = None

    # Button to start tactical distribution
    if st.button("Calculate tactical distribution"):
        with st.spinner("Tactical asset allocation is in progress..."):

            tickers = [asset['ticker'] for asset in portfolio_data['assets']]


            end_date = datetime.now().strftime('%Y-%m-%d')

            if lookback_period == "1M":
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            elif lookback_period == "3M":
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            elif lookback_period == "6M":
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            elif lookback_period == "1Y":
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            else:  # 2Y
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

            if benchmark:
                tickers_with_benchmark = tickers + [benchmark] if benchmark not in tickers else tickers
                prices_data = data_fetcher.get_batch_data(tickers_with_benchmark, start_date, end_date)
            else:
                prices_data = data_fetcher.get_batch_data(tickers, start_date, end_date)

            if not prices_data or all(df.empty for df in prices_data.values()):
                st.error("Failed to load historical data. Please check tickers or change period..")
                return

            # Processing price data
            close_prices = pd.DataFrame()
            for ticker, df in prices_data.items():
                if not df.empty:
                    if 'Adj Close' in df.columns:
                        close_prices[ticker] = df['Adj Close']
                    elif 'Close' in df.columns:
                        close_prices[ticker] = df['Close']

            returns = close_prices.pct_change().dropna()

            current_weights = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

            # We apply the selected method of tactical distribution
            if tactic_method == "market_momentum":

                momentum_window = min(len(returns), 60)
                momentum_scores = returns.iloc[-momentum_window:].mean() * 100

                momentum_norm = (momentum_scores - momentum_scores.min()) / (
                            momentum_scores.max() - momentum_scores.min() + 1e-10)

                new_weights = {}
                for ticker in current_weights:
                    if ticker in momentum_norm:

                        adjust_factor = 1 + (momentum_norm[ticker] - 0.5) * adjustment_strength
                        new_weights[ticker] = current_weights[ticker] * adjust_factor

                total_weight = sum(new_weights.values())
                for ticker in new_weights:
                    new_weights[ticker] /= total_weight

            elif tactic_method == "sector_rotation":

                sector_data = {}
                for asset in portfolio_data['assets']:
                    ticker = asset['ticker']
                    if 'sector' in asset and asset['sector'] != 'N/A':
                        sector = asset['sector']
                    else:

                        info = data_fetcher.get_company_info(ticker)
                        sector = info.get('sector', 'Unknown')

                    if sector not in sector_data:
                        sector_data[sector] = []
                    sector_data[sector].append(ticker)

                sector_performance = data_fetcher.get_sector_performance()

                sector_weights = {}
                if not sector_performance.empty and '3M' in sector_performance.columns:

                    perf_3m = sector_performance.set_index('Sector')['3M']
                    perf_norm = (perf_3m - perf_3m.min()) / (perf_3m.max() - perf_3m.min() + 1e-10)

                    for sector in sector_data:
                        if sector in perf_norm.index:

                            sector_weights[sector] = 0.5 + perf_norm[sector] * adjustment_strength
                        else:
                            sector_weights[sector] = 0.5
                else:

                    for sector in sector_data:
                        sector_weights[sector] = 1.0

                total_sector_weight = sum(sector_weights.values())
                for sector in sector_weights:
                    sector_weights[sector] /= total_sector_weight

                new_weights = {}
                for sector, tickers in sector_data.items():
                    sector_weight = sector_weights.get(sector, 0)
                    ticker_weight = sector_weight / len(tickers)
                    for ticker in tickers:
                        new_weights[ticker] = ticker_weight

            elif tactic_method == "volatility_targeting":

                volatility_window = min(len(returns), 60)
                volatilities = returns.iloc[-volatility_window:].std() * np.sqrt(252)

                inv_vol = 1 / (volatilities + 1e-10)

                vol_weights = inv_vol / inv_vol.sum()

                new_weights = {}
                for ticker in current_weights:
                    if ticker in vol_weights:

                        new_weights[ticker] = (1 - adjustment_strength) * current_weights[
                            ticker] + adjustment_strength * vol_weights[ticker]

                total_weight = sum(new_weights.values())
                for ticker in new_weights:
                    new_weights[ticker] /= total_weight

            else:

                equal_weight = 1.0 / len(current_weights)
                new_weights = {ticker: equal_weight for ticker in current_weights}

            # Results of tactical distribution
            st.subheader("Results of tactical distribution")

            # Create a table with current and new weights
            weights_df = pd.DataFrame({
                'Asset': list(current_weights.keys()),
                'Current weight (%)': [current_weights[t] * 100 for t in current_weights],
                'New weight (%)': [new_weights.get(t, 0) * 100 for t in current_weights],
                'Change (%)': [(new_weights.get(t, 0) - current_weights[t]) * 100 for t in current_weights]
            })

            # Sort by absolute change
            weights_df['Abs. change'] = weights_df['Change (%)'].abs()
            weights_df = weights_df.sort_values('Abs. change', ascending=False)
            weights_df = weights_df.drop('Abs. change', axis=1)

            st.dataframe(weights_df, use_container_width=True)

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=weights_df['Asset'],
                y=weights_df['Current weight (%)'],
                name='Current weight (%)',
                marker_color='lightgrey'
            ))

            fig.add_trace(go.Bar(
                x=weights_df['Asset'],
                y=weights_df['New weight (%)'],
                name='New weight (%)',
                marker_color='royalblue'
            ))

            fig.update_layout(
                title='Comparison of current and new scales',
                barmode='group',
                xaxis_title='Asset',
                yaxis_title='Вес (%)',
                legend_title='',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Button to apply new distribution
            if st.button("Apply new allocation to portfolio"):
                # Updating weights in the portfolio
                for asset in portfolio_data['assets']:
                    ticker = asset['ticker']
                    if ticker in new_weights:
                        asset['weight'] = new_weights[ticker]

                portfolio_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                portfolio_manager.save_portfolio(portfolio_data)

                st.success(f"Portfolio '{selected_portfolio}' has been successfully updated with a new tactical allocation!")

    # Display current distribution
    st.subheader("Current Asset Allocation")
    weights_df = pd.DataFrame({
        'Asset': [asset['ticker'] for asset in portfolio_data['assets']],
        'Вес (%)': [asset['weight'] * 100 for asset in portfolio_data['assets']]
    }).sort_values('Вес (%)', ascending=False)

    st.dataframe(weights_df, use_container_width=True)

    fig = px.pie(
        weights_df,
        values='Вес (%)',
        names='Asset',
        title='Current Asset Allocation'
    )
    st.plotly_chart(fig, use_container_width=True)


def monte_carlo_simulation(data_fetcher, portfolio_manager):
    """
    Function for Monte Carlo portfolio simulation

    Args:
        data_fetcher: DataFetcher instance for loading data
        portfolio_manager: PortfolioDataManager instance for working with portfolios
    """
    st.header("Monte Carlo Simulation")

    # Get a list of portfolios
    portfolios = portfolio_manager.list_portfolios()

    if not portfolios:
        st.info("No portfolios found. Create a portfolio in the 'Create a portfolio' section..")
        return

    # Selecting a portfolio for simulation
    portfolio_names = [p['name'] for p in portfolios]
    selected_portfolio = st.selectbox("Select a portfolio to simulate", portfolio_names)

    if not selected_portfolio:
        return

    # Loading portfolio data
    portfolio_data = portfolio_manager.load_portfolio(selected_portfolio)

    # Display basic information about the portfolio
    st.subheader(f"Portfolio: {portfolio_data['name']}")

    # Simulation parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        years = st.slider(
            "Forecast horizon (years)",
            min_value=1,
            max_value=30,
            value=10,
            step=1
        )

    with col2:
        simulations = st.slider(
            "Number of simulations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )

    with col3:
        initial_investment = st.number_input(
            "Initial investment ($)",
            min_value=1000,
            max_value=10000000,
            value=10000,
            step=1000
        )

    # Additional parameters
    col1, col2 = st.columns(2)

    with col1:
        annual_contribution = st.number_input(
            "Annual fee ($)",
            min_value=0,
            max_value=1000000,
            value=0,
            step=1000
        )

    with col2:
        rebalance_frequency = st.selectbox(
            "Rebalancing frequency",
            ["No", "Annually", "Quarterly", "Monthly"],
            index=1
        )

    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)

        with col1:
            confidence_level = st.slider(
                "Confidence level for calculating VaR",
                min_value=80,
                max_value=99,
                value=95,
                step=1,
                format="%d%%"
            ) / 100

        with col2:
            return_method = st.selectbox(
                "Method of calculating profitability",
                ["Historical", "Parametric"],
                index=0
            )

        st.info(
            "The parametric method assumes a normal distribution of returns, while the historical method uses an empirical distribution.")

    # Button to start the simulation
    if st.button("Run Monte Carlo simulation"):
        with st.spinner('Running simulation... This may take some time.'):

            tickers = [asset['ticker'] for asset in portfolio_data['assets']]
            weights = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

            # Loading historical data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')  # 5 лет истории

            prices_data = data_fetcher.get_batch_data(tickers, start_date, end_date)

            # Checking for data availability
            if not prices_data or all(df.empty for df in prices_data.values()):
                st.error("Failed to load historical data. Please check tickers or change period..")
                return

            # Create a DataFrame with closing prices
            close_prices = pd.DataFrame()
            for ticker, df in prices_data.items():
                if not df.empty:
                    if 'Adj Close' in df.columns:
                        close_prices[ticker] = df['Adj Close']
                    elif 'Close' in df.columns:
                        close_prices[ticker] = df['Close']

            # Calculating profitability
            returns = close_prices.pct_change().dropna()

            # Calculating portfolio yield
            portfolio_returns = PortfolioAnalytics.calculate_portfolio_return(returns, weights)

            # Perform Monte Carlo simulation
            mc_results = RiskManagement.perform_monte_carlo_simulation(
                portfolio_returns,
                initial_value=initial_investment,
                years=years,
                simulations=simulations,
                annual_contribution=annual_contribution
            )

            simulation_data = mc_results['simulation_data']
            percentiles = mc_results['percentiles']

            median_value = percentiles['median']
            p10_value = percentiles['p10']
            p90_value = percentiles['p90']

            st.subheader("Simulation results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Median cost",
                    f"${median_value:,.2f}",
                    f"{(median_value / initial_investment - 1) * 100:.1f}%"
                )

            with col2:
                st.metric(
                    f"Optimistic scenario({int(90)}%)",
                    f"${p90_value:,.2f}",
                    f"{(p90_value / initial_investment - 1) * 100:.1f}%"
                )

            with col3:
                st.metric(
                    f"Pessimistic scenario ({int(10)}%)",
                    f"${p10_value:,.2f}",
                    f"{(p10_value / initial_investment - 1) * 100:.1f}%"
                )

            # Calculate additional metrics
            probability_double = mc_results['probabilities']['prob_reaching_double']
            var = percentiles['p10'] - initial_investment
            var_percent = -var / initial_investment if var < 0 else 0

            st.write(f"**Probability of doubling your investment:** {probability_double * 100:.1f}%")
            st.write(
                f"**Value at Risk (VaR) at {confidence_level * 100:.0f}% confidence level:** ${abs(var):,.2f} ({var_percent * 100:.1f}%)")

            st.subheader("Visualization of the simulation")

            years_arr = np.linspace(0, years, simulation_data.shape[1])

            p10_line = np.percentile(simulation_data, 10, axis=0)
            p25_line = np.percentile(simulation_data, 25, axis=0)
            p50_line = np.percentile(simulation_data, 50, axis=0)
            p75_line = np.percentile(simulation_data, 75, axis=0)
            p90_line = np.percentile(simulation_data, 90, axis=0)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=years_arr,
                y=p90_line,
                mode='lines',
                name='90th percentile',
                line=dict(color='lightgreen', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=years_arr,
                y=p75_line,
                mode='lines',
                name='75th percentile',
                line=dict(color='rgba(0, 176, 246, 0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(144, 238, 144, 0.3)'
            ))

            fig.add_trace(go.Scatter(
                x=years_arr,
                y=p50_line,
                mode='lines',
                name='Median',
                line=dict(color='blue', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=years_arr,
                y=p25_line,
                mode='lines',
                name='25th percentile',
                line=dict(color='rgba(255, 165, 0, 0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(173, 216, 230, 0.3)'
            ))

            fig.add_trace(go.Scatter(
                x=years_arr,
                y=p10_line,
                mode='lines',
                name='10th percentile',
                line=dict(color='salmon', width=2),
                fill='tonexty',
                fillcolor='rgba(255, 165, 0, 0.3)'
            ))

            fig.add_trace(go.Scatter(
                x=[0, years],
                y=[initial_investment, initial_investment],
                mode='lines',
                name='Initial investment',
                line=dict(color='red', width=2, dash='dash')
            ))

            # Add trajectories of several random simulations
            np.random.seed(42)
            sample_indices = np.random.choice(simulation_data.shape[0], min(20, simulation_data.shape[0]),
                                              replace=False)

            for idx in sample_indices:
                fig.add_trace(go.Scatter(
                    x=years_arr,
                    y=simulation_data[idx, :],
                    mode='lines',
                    name=f'Simulation {idx}',
                    line=dict(color='rgba(128, 128, 128, 0.2)', width=1),
                    showlegend=False
                ))

            fig.update_layout(
                title='Portfolio Value Forecast',
                xaxis_title='Years',
                yaxis_title='Portfolio value ($)',
                legend_title='Percentiles',
                hovermode='x',
                yaxis=dict(type='log' if st.checkbox("Logarithmic scale", value=False) else 'linear')
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Distribution of the final value of the portfolio")

            final_values = simulation_data[:, -1]

            fig_hist = go.Figure()

            fig_hist.add_trace(go.Histogram(
                x=final_values,
                nbinsx=50,
                marker_color='rgba(0, 0, 255, 0.5)'
            ))

            # Add vertical lines for percentiles
            fig_hist.add_vline(x=p10_line[-1], line_dash="dash", line_color="red", annotation_text="10%")
            fig_hist.add_vline(x=p50_line[-1], line_dash="dash", line_color="green", annotation_text="50%")
            fig_hist.add_vline(x=p90_line[-1], line_dash="dash", line_color="blue", annotation_text="90%")

            # Add initial investment
            fig_hist.add_vline(x=initial_investment, line_dash="solid", line_color="black",
                               annotation_text="Initial investment")

            fig_hist.update_layout(
                title='Distribution of the final value of the portfolio after {0} years'.format(years),
                xaxis_title='Portfolio value ($)',
                yaxis_title='Frequency',
                showlegend=False
            )

            st.plotly_chart(fig_hist, use_container_width=True)

            # Table with detailed results
            st.subheader("Detailed results by year")

            # Uniformly select values from percentile arrays for each year
            years_range = range(years + 1)
            indices = np.linspace(0, len(p10_line) - 1, years + 1, dtype=int)

            yearly_results = pd.DataFrame({
                'Year': years_range,
                'Initial investment': [initial_investment] * (years + 1),
                '10th percentile': p10_line[indices],
                '25th percentile': p25_line[indices],
                'Median': p50_line[indices],
                '75th percentile': p75_line[indices],
                '90th percentile': p90_line[indices]
            })

            # Format data for display
            for col in yearly_results.columns:
                if col != 'Year':
                    yearly_results[col] = yearly_results[col].map('${:,.2f}'.format)

            st.dataframe(yearly_results, use_container_width=True)