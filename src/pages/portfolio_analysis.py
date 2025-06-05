import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from datetime import datetime, timedelta
import calendar
import logging
logger = logging.getLogger(__name__)


# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Using absolute imports
from src.utils.calculations import PortfolioAnalytics
from src.utils.risk_management import RiskManagement
from src.utils.visualization import PortfolioVisualization
import src.config as config
from src.utils.historical_context import historical_crisis_context, display_historical_context
from src.utils.advanced_visualizations import create_stress_impact_heatmap, create_interactive_stress_impact_chart, create_risk_tree_visualization


def get_delta_color(metric_name: str, portfolio_value: float, benchmark_value: float) -> str:
    """
    Defines the color of the change (delta) depending on the direction and type of metric.

    Returns:
        "normal" — up arrow (green) — metric is better than benchmark.
        "inverse" — down arrow (red) — metric is worse than benchmark.
    """
    difference = portfolio_value - benchmark_value

    lower_is_better = ['Volatility', 'VAR (95%)', 'CVAR (95%)', 'Down capture']

    if metric_name in lower_is_better:
        return "inverse"  # Red color, arrow down

    else:
        return "normal"  # Green color, arrow up


def style_dataframe(df, precision_dict=None):
    """
    Formats a DataFrame for display in Streamlit.

    Args:
        df: DataFrame for formatting
        precision_dict: Formatting dictionary for columns, e.g. {'column': '{:.2f}%'}

    Returns:
        Styled DataFrame
    """
    if precision_dict is None:
        precision_dict = {}
        for col in df.columns:
            if df[col].dtype in [np.float64, float, np.int64, int]:
                if '%' in col or 'percent' in col.lower() or 'return' in col.lower():
                    precision_dict[col] = '{:.2f}%'
                else:
                    precision_dict[col] = '{:.2f}'

    styled_df = df.style

    styled_df = styled_df.format(precision_dict)

    styled_df = styled_df.set_properties(**{
        'text-align': 'center',
        'vertical-align': 'middle'
    })

    return styled_df


def load_portfolio_data(data_fetcher, tickers, start_date_str, end_date_str, benchmark=None):
    """
    Loads historical data for a list of tickers and a benchmark

    Args:
        data_fetcher: DataFetcher instance
        tickers: List of tickers
        start_date_str: Start date as a string
        end_date_str: End date as a string
        benchmark: Benchmark ticker (optional)

    Returns:
        Tuple with (close_prices, returns, valid_tickers, benchmark_returns, prices_data)
    """

    # Add benchmark to the list of tickers for download
    if benchmark and benchmark not in tickers:
        tickers_to_load = tickers + [benchmark]
    else:
        tickers_to_load = tickers

    # Loading historical prices for all assets
    prices_data = data_fetcher.get_batch_data(tickers_to_load, start_date_str, end_date_str)

    # Check that the data was loaded successfully
    if not prices_data or all(df.empty for df in prices_data.values()):
        raise ValueError("Failed to load historical data. Please check tickers or change period..")

    # Create a DataFrame with closing prices
    close_prices = pd.DataFrame()

    for ticker, df in prices_data.items():
        if not df.empty:
            if 'Adj Close' in df.columns:
                close_prices[ticker] = df['Adj Close']
            elif 'Close' in df.columns:
                close_prices[ticker] = df['Close']

    # Calculating return
    returns = PortfolioAnalytics.calculate_returns(close_prices)

    # Filter tickers that are in the data
    valid_tickers = [t for t in tickers if t in returns.columns]

    # If not all tickers are found, we issue a warning
    if len(valid_tickers) < len(tickers):
        missing_tickers = set(tickers) - set(valid_tickers)
        logger.warning(f"No data found for the following tickers: {missing_tickers}")

    # Calculate benchmark yield
    if benchmark and benchmark in returns.columns:
        benchmark_returns = returns[benchmark]
        benchmark_returns = benchmark_returns.dropna()
    else:
        logger.warning(f"Benchmark {benchmark} is missing from the return data.")
        benchmark_returns = pd.Series(dtype=float)

    return close_prices, returns, valid_tickers, benchmark_returns, prices_data

def run(data_fetcher, portfolio_manager):
    """
    Function to display the portfolio analysis page

    Args:
        data_fetcher: DataFetcher instance for loading data
        portfolio_manager: PortfolioDataManager instance for working with portfolios
    """
    st.title("Portfolio analysis")

    # Get a list of portfolios
    portfolios = portfolio_manager.list_portfolios()

    if not portfolios:
        st.info("No portfolios found. Create a portfolio in the 'Create portfolio' section.")
        return

    # Selecting a portfolio for analysis
    portfolio_names = [p['name'] for p in portfolios]
    selected_portfolio = st.selectbox("Select a portfolio for analysis", portfolio_names)

    if not selected_portfolio:
        return

    #Loading portfolio data
    portfolio_data = portfolio_manager.load_portfolio(selected_portfolio)

    # Display basic information about the portfolio
    st.header(f"Portfolio: {portfolio_data['name']}")

    if 'description' in portfolio_data and portfolio_data['description']:
        st.write(portfolio_data['description'])

    st.write(f"Last update: {portfolio_data.get('last_updated', 'Unknown')}")

    # Analysis parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        benchmark = st.selectbox(
            "Benchmark",
            ["SPY", "QQQ", "DIA", "IWM", "VTI"],
            index=0
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

    # Portfolio price update
    update_col, refresh_col = st.columns([3, 1])
    with update_col:
        if st.button("Update asset prices"):
            portfolio_data = portfolio_manager.update_portfolio_prices(portfolio_data)
            portfolio_manager.save_portfolio(portfolio_data)
            st.success("Asset prices updated!")

    with refresh_col:
        if st.button("Download historical data", help="Download historical data for the selected period"):
            st.info("Loading historical data...")

    tabs = st.tabs([
        "Portfolio Overview",
        "Return",
        "Risk",
        "Assets",
        "Correlations",
        "Stress Testing",
        "Rolling Metrics",
        "Advanced Analysis"
    ])

    # We get tickers from the portfolio
    tickers = [asset['ticker'] for asset in portfolio_data['assets']]

    # Convert dates to strings
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    with st.spinner('Loading historical data...'):
        try:
            # Use the new function to load data
            close_prices, returns, valid_tickers, benchmark_returns, prices_data = load_portfolio_data(
                data_fetcher, tickers, start_date_str, end_date_str, benchmark
            )

            # If not all tickers are found, we display a warning
            if len(valid_tickers) < len(tickers):
                missing_tickers = set(tickers) - set(valid_tickers)
                st.warning(f"No data found for the following tickers: {', '.join(missing_tickers)}")

            # Calculate weights from the portfolio
            weights = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

            # Calculating portfolio yield
            portfolio_returns = PortfolioAnalytics.calculate_portfolio_return(returns, weights)

        except ValueError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"There was an error loading data: {e}")
            return

    # Portfolio Overview Tab
    with tabs[0]:
        st.subheader("Portfolio overview")

        # Calculating portfolio metrics
        with st.spinner('Calculating portfolio metrics...'):
            portfolio_metrics = PortfolioAnalytics.calculate_portfolio_metrics(
                portfolio_returns,
                benchmark_returns,
                risk_free_rate=config.RISK_FREE_RATE
            )

        # Create multiple metrics lines
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_name = "Total Return"
            value = portfolio_metrics.get('total_return', 0) * 100
            benchmark_value = portfolio_metrics.get('benchmark_return',
                                                    0) * 100 if 'benchmark_return' in portfolio_metrics else None

            if benchmark_value is not None:
                difference = value - benchmark_value
                delta_color = get_delta_color(metric_name, value, benchmark_value)

                st.metric(
                    metric_name,
                    f"{value:.2f}%",
                    f"{difference:+.2f}%",
                    delta_color=delta_color
                )
            else:
                st.metric(metric_name, f"{value:.2f}%")

        with col2:
            metric_name = "Annual Return"
            value = portfolio_metrics.get('annualized_return', 0) * 100
            benchmark_value = portfolio_metrics.get('benchmark_annualized_return',
                                                    0) * 100 if 'benchmark_annualized_return' in portfolio_metrics else None

            if benchmark_value is not None:
                difference = value - benchmark_value
                delta_color = get_delta_color(metric_name, value, benchmark_value)

                st.metric(
                    metric_name,
                    f"{value:.2f}%",
                    f"{difference:+.2f}%",
                    delta_color=delta_color
                )
            else:
                st.metric(metric_name, f"{value:.2f}%")

        with col3:
            metric_name = "Volatility"
            value = portfolio_metrics.get('volatility', 0) * 100
            benchmark_value = portfolio_metrics.get('benchmark_volatility',
                                                    0) * 100 if 'benchmark_volatility' in portfolio_metrics else None

            if benchmark_value is not None:
                difference = value - benchmark_value
                delta_color = get_delta_color(metric_name, value, benchmark_value)

                st.metric(
                    metric_name,
                    f"{value:.2f}%",
                    f"{difference:.2f}%",
                    delta_color=delta_color
                )
            else:
                st.metric(metric_name, f"{value:.2f}%")

        with col4:
            metric_name = "Sharpe ratio"
            value = portfolio_metrics.get('sharpe_ratio', 0)
            benchmark_value = portfolio_metrics.get('benchmark_sharpe_ratio',
                                                    0) if 'benchmark_sharpe_ratio' in portfolio_metrics else None

            if benchmark_value is not None:
                difference = value - benchmark_value
                delta_color = get_delta_color(metric_name, value, benchmark_value)

                st.metric(
                    metric_name,
                    f"{value:.2f}%",
                    f"{difference:+.2f}%",
                    delta_color=delta_color
                )
            else:
                st.metric(metric_name, f"{value:.2f}%")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_name = "Max drawdown"
            value = -abs(portfolio_metrics.get('max_drawdown', 0)) * 100
            benchmark_value = -abs(portfolio_metrics.get('benchmark_max_drawdown', 0)) * 100 if 'benchmark_max_drawdown' in portfolio_metrics else None


            if benchmark_value is not None:
                difference = value - benchmark_value
                delta_color = get_delta_color(metric_name, value, benchmark_value)

                st.metric(
                    metric_name,
                    f"{value:.2f}%",
                    f"{difference:+.2f}%",
                    delta_color=delta_color
                )
            else:
                st.metric(metric_name, f"{value:.2f}%")

        with col2:
            metric_name = "Sortino ratio"
            value = portfolio_metrics.get('sortino_ratio', 0)
            benchmark_value = portfolio_metrics.get('benchmark_sortino_ratio',
                                                    0) if 'benchmark_sortino_ratio' in portfolio_metrics else None

            if benchmark_value is not None:
                difference = value - benchmark_value
                delta_color = get_delta_color(metric_name, value, benchmark_value)

                st.metric(
                    metric_name,
                    f"{value:.2f}%",
                    f"{difference:+.2f}%",
                    delta_color=delta_color
                )
            else:
                st.metric(metric_name, f"{value:.2f}%")

        with col3:
            metric_name = "Beta"
            value = portfolio_metrics.get('beta', 0)

            st.metric(
                metric_name,
                f"{value:.2f}"
            )

        with col4:
            metric_name = "Alpha"
            value = portfolio_metrics.get('alpha', 0) * 100

            st.metric(
                metric_name,
                f"{value:.2f}"
            )

        # Combined Performance Graph
        st.subheader("Portfolio dynamics")

        # Create a graph with subgraphs
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=("Cumulative Return", "Drawdowns", "Daily Return")
        )

        # 1. Cumulative Return (top chart)
        cumulative_portfolio_returns = (1 + portfolio_returns).cumprod() - 1
        fig.add_trace(
            go.Scatter(
                x=cumulative_portfolio_returns.index,
                y=cumulative_portfolio_returns.values * 100,
                mode='lines',
                line=dict(color='#1f77b4', width=2),
                name='Portfolio'
            ),
            row=1, col=1
        )

        if benchmark_returns is not None:
            cumulative_benchmark_returns = (1 + benchmark_returns).cumprod() - 1
            fig.add_trace(
                go.Scatter(
                    x=cumulative_benchmark_returns.index,
                    y=cumulative_benchmark_returns.values * 100,
                    mode='lines',
                    line=dict(color='#ff7f0e', width=1.5, dash='dash'),
                    name=benchmark
                ),
                row=1, col=1
            )

        # 2. Drawdowns (middle chart)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / peak - 1) * 100

        drawdowns = np.clip(drawdowns, -100, 0)

        fig.add_trace(
            go.Scatter(
                x=drawdowns.index,
                y=drawdowns.values,
                mode='lines',
                line=dict(color='#d62728', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(214, 39, 40, 0.2)',
                name='Drawdowns'
            ),
            row=2, col=1
        )

        if benchmark_returns is not None:
            cumulative_benchmark = (1 + benchmark_returns).cumprod()
            peak_benchmark = cumulative_benchmark.cummax()
            benchmark_drawdowns = (cumulative_benchmark / peak_benchmark - 1) * 100

            fig.add_trace(
                go.Scatter(
                    x=benchmark_drawdowns.index,
                    y=benchmark_drawdowns.values,
                    mode='lines',
                    line=dict(color='#ff7f0e', dash='dash'),
                    name=benchmark
                ),
                row=2, col=1
            )

        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        #3. Daily yield (bottom chart)
        fig.add_trace(
            go.Bar(
                x=portfolio_returns.index,
                y=portfolio_returns.values * 100,
                marker=dict(color='#2ca02c'),
                name='Daily yield'
            ),
            row=3, col=1
        )

        # Updating the chart settings
        fig.update_layout(
            height=800,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode="x unified"
        )

        # Y Axis Titles
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=3, col=1)

        # Display the graph
        st.plotly_chart(fig, use_container_width=True)

        # Display the portfolio structure
        st.subheader("Portfolio structure")

        col1, col2 = st.columns(2)

        with col1:
            # Improved Asset Allocation Chart
            fig_weights = px.pie(
                values=[asset['weight'] for asset in portfolio_data['assets']],
                names=[asset['ticker'] for asset in portfolio_data['assets']],
                title="Distribution by assets",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_weights.update_traces(textposition='inside', textinfo='percent+label')
            fig_weights.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_weights, use_container_width=True)

        with col2:
            # Sector distribution if available
            sectors = {}
            for asset in portfolio_data['assets']:
                if 'sector' in asset and asset['sector'] != 'N/A':
                    sector = asset['sector']
                    if sector in sectors:
                        sectors[sector] += asset['weight']
                    else:
                        sectors[sector] = asset['weight']

            if sectors:
                fig_sectors = px.pie(
                    values=list(sectors.values()),
                    names=list(sectors.keys()),
                    title="Distribution by sectors",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_sectors.update_traces(textposition='inside', textinfo='percent+label')
                fig_sectors.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_sectors, use_container_width=True)

    # Tab "Return"
    with tabs[1]:
        st.subheader("Return Analysis")

        # Create sub-tabs for different types of Return analysis
        return_tabs = st.tabs([
            "Cumulative Return",
            "Periodic Analysis",
            "Distribution of Return",
            "Return by Periods"
        ])

        with return_tabs[0]:

            st.subheader("Cumulative Return with Benchmark")

            log_scale = st.checkbox("Logarithmic scale", value=False)

            cumulative_portfolio_returns = (1 + portfolio_returns).cumprod() - 1

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=cumulative_portfolio_returns.index,
                y=cumulative_portfolio_returns.values * 100,
                mode='lines',
                name='Portfolio',
                line=dict(width=2, color='#1f77b4')
            ))

            if benchmark_returns is not None:
                cumulative_benchmark_returns = (1 + benchmark_returns).cumprod() - 1
                fig.add_trace(go.Scatter(
                    x=cumulative_benchmark_returns.index,
                    y=cumulative_benchmark_returns.values * 100,
                    mode='lines',
                    name=benchmark,
                    line=dict(width=1.5, color='#ff7f0e', dash='dash')
                ))

            fig.update_layout(
                title='Cumulative Return',
                xaxis_title='Date',
                yaxis_title='Return (%)',
                yaxis=dict(type='log' if log_scale else 'linear'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        with return_tabs[1]:
            st.subheader("Periodic return analysis")

            # Calculate annual yield with DatetimeIndex
            if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                annual_returns = portfolio_returns.resample('Y').apply(
                    lambda x: (1 + x).prod() - 1
                ) * 100

                benchmark_annual_returns = None
                if benchmark_returns is not None:
                    benchmark_annual_returns = benchmark_returns.resample('Y').apply(
                        lambda x: (1 + x).prod() - 1
                    ) * 100

                # Annual Return Graph
                fig_annual = go.Figure()

                fig_annual.add_trace(go.Bar(
                    x=annual_returns.index.year,
                    y=annual_returns.values,
                    name='Portfolio',
                    marker_color='#1f77b4'
                ))

                if benchmark_annual_returns is not None:
                    fig_annual.add_trace(go.Bar(
                        x=benchmark_annual_returns.index.year,
                        y=benchmark_annual_returns.values,
                        name=benchmark,
                        marker_color='#ff7f0e'
                    ))

                fig_annual.update_layout(
                    title='Annual Return',
                    xaxis_title='Year',
                    yaxis_title='Return (%)',
                    barmode='group',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode='x unified'
                )

                st.plotly_chart(fig_annual, use_container_width=True)

                # Heatmap of monthly returns
                monthly_returns = PortfolioVisualization.create_monthly_returns_heatmap(portfolio_returns)

                if not monthly_returns.empty:
                    # Convert to percentage for display
                    monthly_returns_pct = monthly_returns * 100

                    fig_heatmap = px.imshow(
                        monthly_returns_pct,
                        labels=dict(x="Month", y="Year", color="Return (%)"),
                        x=monthly_returns_pct.columns,
                        y=monthly_returns_pct.index,
                        color_continuous_scale='RdYlGn',
                        aspect="auto",
                        text_auto='.1f'
                    )

                    fig_heatmap.update_layout(
                        title='Monthly Return Calendar (%)',
                        height=400
                    )

                    st.plotly_chart(fig_heatmap, use_container_width=True)

        with return_tabs[2]:
            st.subheader("Distribution of returns")

            from scipy import stats as scipy_stats

            col1, col2 = st.columns(2)

            with col1:

                fig_daily_dist = px.histogram(
                    portfolio_returns * 100,
                    nbins=40,
                    title="Distribution of daily returns",
                    labels={'value': 'Return (%)', 'count': 'Frequency'},
                    histnorm='probability density',
                    marginal='box',
                    color_discrete_sequence=['#1f77b4']
                )

                # Add a normal distribution curve
                x = np.linspace(min(portfolio_returns * 100), max(portfolio_returns * 100), 100)
                mean = (portfolio_returns * 100).mean()
                std = (portfolio_returns * 100).std()
                y = scipy_stats.norm.pdf(x, mean, std)

                fig_daily_dist.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='lines',
                        name='Normal distribution',
                        line=dict(color='red', dash='dash')
                    )
                )

                st.plotly_chart(fig_daily_dist, use_container_width=True)

            with col2:

                if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                    monthly_returns_series = portfolio_returns.resample('M').apply(
                        lambda x: (1 + x).prod() - 1
                    ) * 100

                    fig_monthly_dist = px.histogram(
                        monthly_returns_series,
                        nbins=20,
                        title="Distribution of monthly returns",
                        labels={'value': 'Return (%)', 'count': 'Frequency'},
                        histnorm='probability density',
                        marginal='box',
                        color_discrete_sequence=['#2ca02c']
                    )

                    # Add a normal distribution curve for monthly returns
                    x_monthly = np.linspace(min(monthly_returns_series), max(monthly_returns_series), 100)
                    mean_monthly = monthly_returns_series.mean()
                    std_monthly = monthly_returns_series.std()
                    y_monthly = scipy_stats.norm.pdf(x_monthly, mean_monthly,
                                                     std_monthly)

                    fig_monthly_dist.add_trace(
                        go.Scatter(
                            x=x_monthly,
                            y=y_monthly,
                            mode='lines',
                            name='Normal distribution',
                            line=dict(color='red', dash='dash')
                        )
                    )

                    st.plotly_chart(fig_monthly_dist, use_container_width=True)

        with return_tabs[3]:
            st.subheader("Return by periods")

            if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                period_returns = PortfolioAnalytics.calculate_period_performance(portfolio_returns)

                if benchmark_returns is not None:
                    benchmark_period_returns = PortfolioAnalytics.calculate_period_performance(benchmark_returns)

                    periods_data = []
                    for period in period_returns:
                        periods_data.append({
                            'Period': period,
                            'Portfolio (%)': period_returns[period] * 100,
                            'Benchmark (%)': benchmark_period_returns.get(period, 0) * 100,
                            'Difference (%)': (period_returns[period] - benchmark_period_returns.get(period, 0)) * 100
                        })

                    periods_df = pd.DataFrame(periods_data)

                    def highlight_diff(val):
                        if isinstance(val, float):
                            if val > 0:
                                return 'background-color: rgba(75, 192, 192, 0.2); color: green'
                            elif val < 0:
                                return 'background-color: rgba(255, 99, 132, 0.2); color: red'
                        return ''

                    styled_df = periods_df.style.format({
                        'Portfolio (%)': '{:.2f}%',
                        'Benchmark (%)': '{:.2f}%',
                        'Difference (%)': '{:.2f}%'
                    }).applymap(highlight_diff, subset=['Difference (%)'])

                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        height=min(35 * (len(periods_df) + 1), 300)
                    )

                    # Visual comparison
                    fig_periods = px.bar(
                        periods_df,
                        x='Period',
                        y=['Portfolio (%)', 'Benchmark (%)'],
                        barmode='group',
                        title='Comparison of returns by periods',
                        labels={'value': 'Return (%)', 'variable': ''}
                    )

                    st.plotly_chart(fig_periods, use_container_width=True)

                    st.subheader("The best and the worst periods")

                    worst_periods = PortfolioVisualization.create_worst_periods_table(
                        portfolio_returns, benchmark_returns
                    )

                    if not worst_periods.empty:
                        # Format numeric columns
                        for col in ['Return', 'Benchmark', 'Difference']:
                            worst_periods[col] = worst_periods[col].apply(lambda x: f"{x * 100:.2f}%")

                        st.dataframe(
                            worst_periods.style.set_properties(**{
                                'text-align': 'center',
                                'vertical-align': 'middle'
                            }),
                            use_container_width=True,
                            height=min(35 * (len(worst_periods) + 1), 300)
                        )

    # Risk tab
    with tabs[2]:
        st.subheader("Risk analysis")

        risk_tabs = st.tabs([
            "Key metrics",
            "Drawdown Analysis",
            "VaR and CVaR",
            "Impact on risk"
        ])

        with risk_tabs[0]:
            st.subheader("Key risk metrics")

            # Comparison of risk metrics with benchmark
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                metric_name = "Volatility"
                value = portfolio_metrics.get('volatility', 0) * 100
                benchmark_value = portfolio_metrics.get('benchmark_volatility',
                                                        0) * 100 if 'benchmark_volatility' in portfolio_metrics else None

                if benchmark_value is not None:
                    difference = value - benchmark_value
                    delta_color = get_delta_color(metric_name, value, benchmark_value)

                    st.metric(
                        metric_name,
                        f"{value:.2f}%",
                        f"{difference:.2f}%",
                        delta_color=delta_color
                    )
                else:
                    st.metric(metric_name, f"{value:.2f}%")

            with col2:
                metric_name = "Max drawdown"
                value = -abs(portfolio_metrics.get('max_drawdown', 0)) * 100
                benchmark_value = -abs(portfolio_metrics.get('benchmark_max_drawdown', 0)) * 100 if 'benchmark_max_drawdown' in portfolio_metrics else None

                if benchmark_value is not None:
                    difference = value - benchmark_value
                    delta_color = get_delta_color(metric_name, value, benchmark_value)

                    st.metric(
                        metric_name,
                        f"{value:.2f}%",
                        f"{difference:+.2f}%",
                        delta_color=delta_color
                    )
                else:
                    st.metric(metric_name, f"{value:.2f}%")

            with col3:
                metric_name = "Sortino ratio"
                value = portfolio_metrics.get('sortino_ratio', 0)
                benchmark_value = portfolio_metrics.get('benchmark_sortino_ratio',
                                                        0) if 'benchmark_sortino_ratio' in portfolio_metrics else None

                if benchmark_value is not None:
                    difference = value - benchmark_value
                    delta_color = get_delta_color(metric_name, value, benchmark_value)

                    st.metric(
                        metric_name,
                        f"{value:.2f}%",
                        f"{difference:+.2f}%",
                        delta_color=delta_color
                    )
                else:
                    st.metric(metric_name, f"{value:.2f}%")

            with col4:
                metric_name = "Calmar Ratio"
                value = portfolio_metrics.get('calmar_ratio', 0)
                benchmark_value = portfolio_metrics.get('bbenchmark_calmar_ratio',
                                                        0) if 'benchmark_calmar_ratio' in portfolio_metrics else None

                if benchmark_value is not None:
                    difference = value - benchmark_value
                    delta_color = get_delta_color(metric_name, value, benchmark_value)

                    st.metric(
                        metric_name,
                        f"{value:.2f}%",
                        f"{difference:+.2f}%",
                        delta_color=delta_color
                    )
                else:
                    st.metric(metric_name, f"{value:.2f}%")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                metric_name = "VAR (95%)"
                value = portfolio_metrics.get('var_95', 0) * 100
                benchmark_value = portfolio_metrics.get('benchmark_var_95',
                                                        0) * 100 if 'benchmark_var_95' in portfolio_metrics else None

                if benchmark_value is not None:
                    difference = value - benchmark_value
                    delta_color = get_delta_color(metric_name, value, benchmark_value)

                    st.metric(
                        metric_name,
                        f"{value:.2f}%",
                        f"{difference:+.2f}%",
                        delta_color=delta_color
                    )
                else:
                    st.metric(metric_name, f"{value:.2f}%")

            with col2:
                metric_name = "CVAR (95%)"
                value = portfolio_metrics.get('cvar_95', 0) * 100
                benchmark_value = portfolio_metrics.get('benchmark_cvar_95',
                                                        0) * 100 if 'benchmark_cvar_95' in portfolio_metrics else None

                if benchmark_value is not None:
                    difference = value - benchmark_value
                    delta_color = get_delta_color(metric_name, value, benchmark_value)

                    st.metric(
                        metric_name,
                        f"{value:.2f}%",
                        f"{difference:+.2f}%",
                        delta_color=delta_color
                    )
                else:
                    st.metric(metric_name, f"{value:.2f}%")

            with col3:
                metric_name = "Up capture"
                value = portfolio_metrics.get('up_capture', 0)

                st.metric(
                    metric_name,
                    f"{value:.2f}",
                    f"{value - 1:.2f}" if 'up_capture' in portfolio_metrics else None
                )

            with col4:
                metric_name = "Down capture"
                value = portfolio_metrics.get('down_capture', 0)

                st.metric(
                    metric_name,
                    f"{value:.2f}",
                    f"{1 - value:+.2f}" if 'down_capture' in portfolio_metrics else None,
                    delta_color="inverse"
                )

            # Risk/reward chart
            st.subheader("Risk/reward ratio")

            fig_risk_return = go.Figure()

            fig_risk_return.add_trace(go.Scatter(
                x=[portfolio_metrics.get('volatility', 0) * 100],
                y=[portfolio_metrics.get('annualized_return', 0) * 100],
                mode='markers',
                marker=dict(size=15, color='blue'),
                name='Portfolio'
            ))

            if 'benchmark_volatility' in portfolio_metrics and 'benchmark_annualized_return' in portfolio_metrics:
                fig_risk_return.add_trace(go.Scatter(
                    x=[portfolio_metrics.get('benchmark_volatility', 0) * 100],
                    y=[portfolio_metrics.get('benchmark_annualized_return', 0) * 100],
                    mode='markers',
                    marker=dict(size=15, color='orange'),
                    name=benchmark
                ))

            # Add a risk-free rate
            risk_free = config.RISK_FREE_RATE * 100

            # Add a Capital Market Line
            x_range = np.linspace(0, max(portfolio_metrics.get('volatility', 0) * 100 * 1.5,
                                         portfolio_metrics.get('benchmark_volatility',
                                                               0) * 100 * 1.5 if 'benchmark_volatility' in portfolio_metrics else 0),
                                  100)
            if 'sharpe_ratio' in portfolio_metrics:
                slope = portfolio_metrics.get('sharpe_ratio', 0)
                y_range = risk_free + slope * x_range

                fig_risk_return.add_trace(go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    line=dict(color='green', dash='dash'),
                    name='CML (Capital Market Line)'
                ))

            fig_risk_return.update_layout(
                title='Risk/Return',
                xaxis_title='Risk (Volatility, %)',
                yaxis_title='Return (%)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500
            )

            st.plotly_chart(fig_risk_return, use_container_width=True)

        with risk_tabs[1]:
            st.subheader("Drawdown Analysis")

            cumulative_returns = (1 + portfolio_returns).cumprod()
            peak = cumulative_returns.cummax()
            drawdowns = (cumulative_returns / peak - 1) * 100

            fig_drawdown = go.Figure()

            fig_drawdown.add_trace(go.Scatter(
                x=drawdowns.index,
                y=drawdowns.values,
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(214, 39, 40, 0.2)',
                line=dict(color='#d62728'),
                name='Portfolio'
            ))

            if benchmark_returns is not None:
                cumulative_benchmark = (1 + benchmark_returns).cumprod()
                peak_benchmark = cumulative_benchmark.cummax()
                benchmark_drawdowns = (cumulative_benchmark / peak_benchmark - 1) * 100

                fig_drawdown.add_trace(go.Scatter(
                    x=benchmark_drawdowns.index,
                    y=benchmark_drawdowns.values,
                    mode='lines',
                    line=dict(color='#ff7f0e', dash='dash'),
                    name=benchmark
                ))

            fig_drawdown.update_layout(
                title='Drawdowns',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500
            )

            st.plotly_chart(fig_drawdown, use_container_width=True)

            # Detailed analysis of drawdowns
            st.subheader("Top 5 Drawdowns")

            drawdown_analysis = RiskManagement.analyze_drawdowns(portfolio_returns)

            if not drawdown_analysis.empty:
                # Sort by depth (biggest drawdowns first) and take the top 5
                top_drawdowns = drawdown_analysis.sort_values(['depth', 'start_date'], ascending=[True, False]).head(5)

                display_drawdowns = pd.DataFrame({
                    'Start': top_drawdowns['start_date'].dt.strftime('%Y-%m-%d'),
                    'Bottom': top_drawdowns['valley_date'].dt.strftime('%Y-%m-%d'),
                    'Recovery': top_drawdowns['recovery_date'].fillna('In progress').apply(
                        lambda x: x.strftime('%Y-%m-%d') if x != 'In progress' else x
                    ),
                    'Depth (%)': top_drawdowns['depth'] * 100,
                    'Duration (days)': top_drawdowns['length'],
                    'Recovery (days)': top_drawdowns['recovery']
                })

                st.dataframe(
                    style_dataframe(display_drawdowns, {
                        'Depth (%)': '{:.2f}%',
                        'Duration (days)': '{:.0f}',
                        'Recovery (days)': '{:.0f}'
                    }),
                    use_container_width=True,
                    height=min(35 * (len(display_drawdowns) + 1), 300)
                )

            with risk_tabs[2]:
                st.subheader("Value at Risk (VaR) and Conditional VaR (CVaR)")

                col1, col2 = st.columns(2)

                with col1:
                    confidence_level = st.slider(
                        "Trust level",
                        min_value=90,
                        max_value=99,
                        value=95,
                        step=1,
                        format="%d%%"
                    ) / 100

                    # Calculating VaR using different methods
                    var_hist = RiskManagement.calculate_var_historical(portfolio_returns,
                                                                       confidence_level=confidence_level)
                    var_param = RiskManagement.calculate_var_parametric(portfolio_returns,
                                                                        confidence_level=confidence_level)
                    var_mc = RiskManagement.calculate_var_monte_carlo(portfolio_returns,
                                                                      confidence_level=confidence_level)
                    cvar = RiskManagement.calculate_cvar(portfolio_returns, confidence_level=confidence_level)

                    # Create a table with VaR
                    var_data = [
                        {"Method": "Historical", "Meaning (%)": var_hist * 100},
                        {"Method": "Parametric", "Meaning (%)": var_param * 100},
                        {"Method": "Monte Carlo", "Meaning (%)": var_mc * 100},
                        {"Method": "CVaR", "Meaning (%)": cvar * 100}
                    ]

                    var_df = pd.DataFrame(var_data)

                    st.dataframe(
                        style_dataframe(var_df, {
                            'Meaning (%)': '{:.2f}%'
                        }),
                        use_container_width=True,
                        height=min(35 * (len(var_df) + 1), 200)
                    )

                with col2:
                    # Visualization of VaR on the distribution of returns
                    fig_var = px.histogram(
                        portfolio_returns * 100,
                        nbins=50,
                        title=f"Value at Risk at {confidence_level * 100:.0f}% confidence level",
                        labels={'value': 'Daily return (%)', 'count': 'Frequency'},
                        color_discrete_sequence=['lightskyblue']
                    )

                    # Add VaR line
                    fig_var.add_vline(
                        x=-var_hist * 100,
                        line_width=2,
                        line_color='red',
                        line_dash='dash',
                        annotation_text=f'VaR {confidence_level * 100:.0f}%: {var_hist * 100:.2f}%',
                        annotation_position="top right"
                    )

                    # Add the CVaR line
                    fig_var.add_vline(
                        x=-cvar * 100,
                        line_width=2,
                        line_color='darkred',
                        line_dash='dash',
                        annotation_text=f'CVaR {confidence_level * 100:.0f}%: {cvar * 100:.2f}%',
                        annotation_position="bottom right"
                    )

                    fig_var.update_layout(
                        xaxis_title='Daily return (%)',
                        yaxis_title='Frequency',
                        showlegend=False
                    )

                    st.plotly_chart(fig_var, use_container_width=True)

            with risk_tabs[3]:
                st.subheader("Impact on risk")

                # Calculation of Impact on risk
                risk_contribution = RiskManagement.calculate_risk_contribution(returns, weights)

                if risk_contribution:
                    # Sort by Impact on risk (descending)
                    risk_contrib_sorted = {k: v for k, v in
                                           sorted(risk_contribution.items(), key=lambda item: item[1], reverse=True)}

                    # Create a DataFrame
                    risk_contrib_df = pd.DataFrame({
                        'Asset': list(risk_contrib_sorted.keys()),
                        'Impact on risk (%)': [v * 100 for v in risk_contrib_sorted.values()]
                    })

                    # Visualization of risk impact
                    fig_risk_contrib = px.bar(
                        risk_contrib_df,
                        x='Asset',
                        y='Impact on risk (%)',
                        title='Impact on assets to the overall portfolio risk',
                        color='Impact on risk (%)',
                        color_continuous_scale='viridis'
                    )

                    st.plotly_chart(fig_risk_contrib, use_container_width=True)

                    # Comparison of risk Impact with weights
                    compare_df = pd.DataFrame({
                        'Asset': list(risk_contrib_sorted.keys()),
                        'Impact on risk (%)': [risk_contrib_sorted[t] * 100 for t in risk_contrib_sorted],
                        'Weight (%)': [weights.get(t, 0) * 100 for t in risk_contrib_sorted]
                    })

                    fig_risk_weight = go.Figure()

                    fig_risk_weight.add_trace(go.Bar(
                        x=compare_df['Asset'],
                        y=compare_df['Impact on risk (%)'],
                        name='Impact on risk (%)',
                        marker_color='#1f77b4'
                    ))

                    fig_risk_weight.add_trace(go.Bar(
                        x=compare_df['Asset'],
                        y=compare_df['Weight (%)'],
                        name='Weight (%)',
                        marker_color='#ff7f0e'
                    ))

                    fig_risk_weight.update_layout(
                        title='Comparison of risk impact and asset weighting',
                        xaxis_title='Asset',
                        yaxis_title='Percent (%)',
                        barmode='group',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )

                    st.plotly_chart(fig_risk_weight, use_container_width=True)

                    # Portfolio diversification
                    st.subheader("Diversification Assessment")

                    # Calculate the diversification coefficient
                    portfolio_vol = portfolio_metrics.get('volatility', 0)
                    weighted_vol = sum([weights.get(ticker, 0) * returns[ticker].std() * np.sqrt(252)
                                        for ticker in weights if ticker in returns.columns])

                    diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 0

                    st.metric(
                        "Diversification coefficient",
                        f"{diversification_ratio:.2f}",
                        "Higher is better"
                    )

                    st.info("""
                        **The diversification coefficient** shows the ratio of the weighted sum of the volatilities 
                        of individual assets to the total volatility of the portfolio. A value above 1 indicates 
                        a positive diversification effect.
                        """)

    with tabs[3]:
        st.subheader("Assets Analysis")

        # Create sub-tabs for different types of asset analysis
        assets_tabs = st.tabs([
            "Asset overview",
            "Asset impact",
            "Price dynamics",
            "Asset Analysis"
        ])

        with assets_tabs[0]:
            # Displaying a list of assets
            assets_data = []
            for asset in portfolio_data['assets']:
                asset_row = {
                    'Ticker': asset['ticker'],
                    'Weight (%)': asset['weight'] * 100
                }

                # Adding accessible information
                for field in ['name', 'sector', 'industry', 'asset_class', 'currency']:
                    if field in asset:
                        asset_row[field.capitalize()] = asset[field]

                if 'current_price' in asset:
                    asset_row['Current price'] = asset['current_price']

                if 'price_change_pct' in asset:
                    asset_row['Price change (%)'] = asset['price_change_pct']

                assets_data.append(asset_row)

            assets_df = pd.DataFrame(assets_data)

            # Define the formatting dictionary
            format_dict = {
                'Weight (%)': '{:.2f}%',
                'Current price': '{:.2f}',
                'Price change (%)': '{:.2f}%'
            }

            st.dataframe(
                style_dataframe(assets_df, format_dict),
                use_container_width=True,
                height=min(35 * (len(assets_df) + 1), 400)
            )

            if len(portfolio_data['assets']) > 10:
                st.warning(
                    f"The portfolio contains {len(portfolio_data['assets'])} assets. For better diversification, no more than 20-25 assets are recommended.")

        with assets_tabs[1]:
            st.subheader("Impact on assets to return and risk")


            cumulative_returns = {}
            weights_dict = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

            for ticker in returns.columns:
                if ticker in weights_dict:
                    cumulative_returns[ticker] = (1 + returns[ticker]).cumprod().iloc[-1] - 1

            # Создаем DataFrame
            contrib_df = pd.DataFrame({
                'Asset': list(cumulative_returns.keys()),
                'Return (%)': [cumulative_returns[ticker] * 100 for ticker in cumulative_returns],
                'Weighted Return (%)': [cumulative_returns[ticker] * weights_dict[ticker] * 100 for ticker in
                                              cumulative_returns],
                'Weight (%)': [weights_dict[ticker] * 100 for ticker in cumulative_returns]
            })

            # Sort by weighted yield
            contrib_df = contrib_df.sort_values('Weighted Return (%)', ascending=False)

            # Visualization of Impact on return
            fig_contrib = px.bar(
                contrib_df,
                x='Asset',
                y='Weighted Return (%)',
                title='Impact on assets to total return',
                color='Weighted Return (%)',
                color_continuous_scale='RdYlGn',
                text='Weighted Return (%)'
            )

            fig_contrib.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_contrib.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

            st.plotly_chart(fig_contrib, use_container_width=True)

            # Comparison of Impact on return and risk
            if risk_contribution:
                compare_contrib_df = pd.DataFrame({
                    'Asset': list(cumulative_returns.keys()),
                    'Impact on return (%)': [cumulative_returns[ticker] * weights_dict[ticker] * 100 /
                                               sum([cumulative_returns[t] * weights_dict[t] for t in
                                                    cumulative_returns])
                                               for ticker in cumulative_returns],
                    'Impact on risk (%)': [risk_contribution.get(ticker, 0) * 100 for ticker in cumulative_returns]
                })

                # Sort by Impact on return
                compare_contrib_df = compare_contrib_df.sort_values('Impact on return (%)', ascending=False)

                # Visual comparison
                fig_compare = go.Figure()

                fig_compare.add_trace(go.Bar(
                    x=compare_contrib_df['Asset'],
                    y=compare_contrib_df['Impact on return (%)'],
                    name='Impact on return (%)',
                    marker_color='#2ca02c'
                ))

                fig_compare.add_trace(go.Bar(
                    x=compare_contrib_df['Asset'],
                    y=compare_contrib_df['Impact on risk (%)'],
                    name='Impact on risk (%)',
                    marker_color='#d62728'
                ))

                fig_compare.update_layout(
                    title='Comparison of Impact on return and risk',
                    xaxis_title='Asset',
                    yaxis_title='Percent (%)',
                    barmode='group',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig_compare, use_container_width=True)

        with assets_tabs[2]:
            st.subheader("Asset price dynamics")

            # Normalize prices for comparison
            normalized_prices = close_prices.copy()
            for ticker in close_prices.columns:
                if ticker in weights:
                    first_price = normalized_prices[ticker].iloc[0]
                    if first_price > 0:
                        normalized_prices[ticker] = normalized_prices[ticker] / first_price

            selected_assets = st.multiselect(
                "Select assets to display",
                options=list(weights.keys()),
                default=list(weights.keys())[:5] if len(weights) > 5 else list(weights.keys())
            )

            show_percentage = st.checkbox("Show change as a percentage", value=True)

            # Plotting a graph
            if selected_assets:

                display_prices = close_prices.copy()


                for ticker in display_prices.columns:
                    first_price = display_prices[ticker].iloc[0]
                    if first_price > 0:
                        if show_percentage:

                            display_prices[ticker] = (display_prices[ticker] / first_price - 1) * 100
                        else:

                            display_prices[ticker] = display_prices[ticker] / first_price

                fig_norm_prices = go.Figure()

                colors = px.colors.qualitative.Plotly

                for i, ticker in enumerate(selected_assets):
                    if ticker in display_prices.columns:
                        fig_norm_prices.add_trace(go.Scatter(
                            x=display_prices.index,
                            y=display_prices[ticker],
                            mode='lines',
                            name=ticker,
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))

                if benchmark in display_prices.columns:
                    fig_norm_prices.add_trace(go.Scatter(
                        x=display_prices.index,
                        y=display_prices[benchmark],
                        mode='lines',
                        line=dict(color='#FFFFFF', width=2, dash='dash'),
                        name=f"{benchmark} (Benchmark)"
                    ))

                # Set up the header depending on the mode
                if show_percentage:
                    title = 'Asset Price Change (% from Start Date)'
                    y_axis_title = 'Price change (%)'
                else:
                    title = 'Normalized asset prices (from initial date)'
                    y_axis_title = 'Normalized price'

                fig_norm_prices.update_layout(
                    title=title,
                    xaxis_title='Date',
                    yaxis_title=y_axis_title,
                    legend_title='Assets',
                    hovermode='x unified',
                    # Добавляем темную тему
                    template='plotly_dark',
                    # Улучшаем внешний вид
                    height=500,
                    margin=dict(l=40, r=40, t=50, b=40)
                )

                st.plotly_chart(fig_norm_prices, use_container_width=True)

        with assets_tabs[3]:
            st.subheader("Detailed analysis of a single asset")

            selected_asset = st.selectbox(
                "Select an asset for detailed analysis",
                options=list(weights.keys())
            )

            if selected_asset in close_prices.columns:
                asset_price = close_prices[selected_asset]
                asset_return = returns[selected_asset]

                # Asset statistics
                st.write("### Asset statistics")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    metric_name = "Total Return"
                    value = portfolio_metrics.get('total_return', 0) * 100
                    benchmark_value = portfolio_metrics.get('benchmark_return',
                                                            0) * 100 if 'benchmark_return' in portfolio_metrics else None

                    if benchmark_value is not None:
                        difference = value - benchmark_value
                        delta_color = get_delta_color(metric_name, value, benchmark_value)

                        st.metric(
                            metric_name,
                            f"{value:.2f}%",
                            f"{difference:+.2f}%",
                            delta_color=delta_color
                        )
                    else:
                        st.metric(metric_name, f"{value:.2f}%")

                with col2:
                    metric_name = "Annual Return"
                    value = portfolio_metrics.get('annualized_return', 0) * 100
                    benchmark_value = portfolio_metrics.get('benchmark_annualized_return',
                                                            0) * 100 if 'benchmark_annualized_return' in portfolio_metrics else None

                    if benchmark_value is not None:
                        difference = value - benchmark_value
                        delta_color = get_delta_color(metric_name, value, benchmark_value)

                        st.metric(
                            metric_name,
                            f"{value:.2f}%",
                            f"{difference:+.2f}%",
                            delta_color=delta_color
                        )
                    else:
                        st.metric(metric_name, f"{value:.2f}%")

                with col3:
                    metric_name = "Volatility"
                    value = portfolio_metrics.get('volatility', 0) * 100
                    benchmark_value = portfolio_metrics.get('benchmark_volatility',
                                                            0) * 100 if 'benchmark_volatility' in portfolio_metrics else None

                    if benchmark_value is not None:
                        difference = value - benchmark_value
                        delta_color = get_delta_color(metric_name, value, benchmark_value)

                        st.metric(
                            metric_name,
                            f"{value:.2f}%",
                            f"{difference:+.2f}%",
                            delta_color=delta_color
                        )
                    else:
                        st.metric(metric_name, f"{value:.2f}%")

                with col4:
                    metric_name = "Sharpe ratio"
                    value = portfolio_metrics.get('sharpe_ratio', 0)
                    benchmark_value = portfolio_metrics.get('benchmark_sharpe_ratio',
                                                            0) if 'benchmark_sharpe_ratio' in portfolio_metrics else None

                    if benchmark_value is not None:
                        difference = value - benchmark_value
                        delta_color = get_delta_color(metric_name, value, benchmark_value)

                        st.metric(
                            metric_name,
                            f"{value:.2f}%",
                            f"{difference:+.2f}%",
                            delta_color=delta_color
                        )
                    else:
                        st.metric(metric_name, f"{value:.2f}%")

                with st.spinner('Loading volume data...'):

                    asset_data = data_fetcher.get_historical_prices(
                        selected_asset,
                        start_date_str,
                        end_date_str
                    )

                    has_volume = not asset_data.empty and 'Volume' in asset_data.columns


                if has_volume:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])

                    fig.add_trace(
                        go.Scatter(
                            x=asset_price.index,
                            y=asset_price.values,
                            mode='lines',
                            name='Price',
                            line=dict(color='#1f77b4')
                        ),
                        secondary_y=False
                    )

                    fig.add_trace(
                        go.Bar(
                            x=asset_data.index,
                            y=asset_data['Volume'],
                            name='Volume',
                            marker=dict(color='#2ca02c', opacity=0.3)
                        ),
                        secondary_y=True
                    )


                    fig.update_layout(
                        title=f'{selected_asset} - Price and volume',
                        xaxis_title='Date',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode='x unified'
                    )

                    fig.update_yaxes(title_text="Price", secondary_y=False)
                    fig.update_yaxes(title_text="Volume", secondary_y=True)

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # If volume data is not available, just show price
                    fig = px.line(
                        asset_price,
                        title=f'{selected_asset} - Price',
                        labels={'value': 'Price', 'index': 'Date'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Comparison with portfolio and benchmark
                st.subheader("Comparison of return")

                cumulative_asset = (1 + asset_return).cumprod() - 1
                cumulative_portfolio = (1 + portfolio_returns).cumprod() - 1

                fig_compare = go.Figure()

                fig_compare.add_trace(go.Scatter(
                    x=cumulative_asset.index,
                    y=cumulative_asset.values * 100,
                    mode='lines',
                    name=selected_asset,
                    line=dict(color='#1f77b4')
                ))

                fig_compare.add_trace(go.Scatter(
                    x=cumulative_portfolio.index,
                    y=cumulative_portfolio.values * 100,
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='#2ca02c')
                ))

                if benchmark_returns is not None:
                    cumulative_benchmark = (1 + benchmark_returns).cumprod() - 1
                    fig_compare.add_trace(go.Scatter(
                        x=cumulative_benchmark.index,
                        y=cumulative_benchmark.values * 100,
                        mode='lines',
                        name=benchmark,
                        line=dict(color='#ff7f0e', dash='dash')
                    ))

                fig_compare.update_layout(
                    title=f'Comparing the return of {selected_asset} with a portfolio and a benchmark',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return (%)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode='x unified'
                )

                st.plotly_chart(fig_compare, use_container_width=True)

                # Asset Correlations
                st.subheader("Correlations")

                correlations = returns.corr()[selected_asset].drop(selected_asset).sort_values(ascending=False)

                fig_corr = px.bar(
                    x=correlations.index,
                    y=correlations.values,
                    title=f'Correlation of {selected_asset} with other assets',
                    labels={'x': 'Asset', 'y': 'Correlation'},
                    color=correlations.values,
                    color_continuous_scale='RdBu_r',
                    range_color=[-1, 1]
                )

                st.plotly_chart(fig_corr, use_container_width=True)

    # Correlations tab
    with tabs[4]:
        st.subheader("Correlation analysis")

        # Create sub-tabs for different types of correlation analysis
        corr_tabs = st.tabs([
            "Correlation matrix",
            "Correlation with benchmark",
            "Cluster analysis",
            "Dynamics of correlations"
        ])

        with corr_tabs[0]:
            # Correlation matrix
            correlation_matrix = returns.corr()

            # Create a correlation heat map
            fig_corr = px.imshow(
                correlation_matrix,
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1,
                title='Correlation matrix'
            )

            fig_corr.update_layout(
                height=600,
                xaxis_title='Asset',
                yaxis_title='Asset'
            )

            st.plotly_chart(fig_corr, use_container_width=True)

            # Correlation statistics
            st.subheader("Correlation statistics")

            # Create a mask for the upper triangle of the matrix (without the diagonal)
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

            # Get all correlations from the upper triangle
            all_correlations = correlation_matrix.values[mask]

            # Display statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Average correlation",
                    f"{np.mean(all_correlations):.2f}"
                )

            with col2:
                st.metric(
                    "Median correlation",
                    f"{np.median(all_correlations):.2f}"
                )

            with col3:
                st.metric(
                    "Min. correlation",
                    f"{np.min(all_correlations):.2f}"
                )

            with col4:
                st.metric(
                    "Max correlation",
                    f"{np.max(all_correlations):.2f}"
                )

        with corr_tabs[1]:
            # Correlation with benchmark
            if benchmark in returns.columns:
                correlations_with_benchmark = returns.corr()[benchmark].drop(benchmark).sort_values(ascending=False)

                fig_bench_corr = px.bar(
                    x=correlations_with_benchmark.index,
                    y=correlations_with_benchmark.values,
                    title=f'Asset correlation with {benchmark}',
                    labels={'x': 'Asset', 'y': 'Correlation'},
                    color=correlations_with_benchmark.values,
                    color_continuous_scale='RdBu_r',
                    range_color=[-1, 1]
                )

                st.plotly_chart(fig_bench_corr, use_container_width=True)

                corr_df = pd.DataFrame({
                    'Asset': correlations_with_benchmark.index,
                    'Correlation with benchmark': correlations_with_benchmark.values,
                    'Beta': [PortfolioAnalytics.calculate_beta(returns[ticker], returns[benchmark])
                             for ticker in correlations_with_benchmark.index]
                })

                def color_correlation(val):
                    if val > 0.8:
                        return 'background-color: rgba(255, 0, 0, 0.2)'
                    elif val > 0.5:
                        return 'background-color: rgba(255, 165, 0, 0.2)'
                    elif val < -0.5:
                        return 'background-color: rgba(0, 128, 0, 0.2)'
                    return ''

                styled_corr_df = corr_df.style.format({
                    'Correlation with benchmark': '{:.2f}',
                    'Beta': '{:.2f}'
                }).applymap(color_correlation, subset=['Correlation with benchmark']).set_properties(**{
                    'text-align': 'center',
                    'vertical-align': 'middle'
                })

                st.dataframe(
                    styled_corr_df,
                    use_container_width=True,
                    height=min(35 * (len(corr_df) + 1), 350)
                )

        with corr_tabs[2]:
            st.subheader("Cluster analysis of correlations")

            try:
                from scipy.cluster import hierarchy
                from scipy.spatial import distance

                dist = distance.squareform(1 - np.abs(correlation_matrix))

                linkage = hierarchy.linkage(dist, method='average')

                dendrogram_result = hierarchy.dendrogram(linkage, no_plot=True)

                order = dendrogram_result['leaves']

                reordered_corr = correlation_matrix.iloc[order, order]

                fig_cluster = px.imshow(
                    reordered_corr,
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    title='Clustered Correlation Matrix'
                )

                fig_cluster.update_layout(
                    height=600,
                    xaxis_title='Asset',
                    yaxis_title='Asset'
                )

                st.plotly_chart(fig_cluster, use_container_width=True)

                fig_dendro = go.Figure()

                dendro = hierarchy.dendrogram(linkage, labels=correlation_matrix.index, orientation='bottom',
                                              no_plot=True)

                for i, d in enumerate(dendro['dcoord']):
                    fig_dendro.add_trace(go.Scatter(
                        x=dendro['icoord'][i],
                        y=dendro['dcoord'][i],
                        mode='lines',
                        line=dict(color='blue', width=1),
                        hoverinfo='none',
                        showlegend=False
                    ))

                if 'ivl_positions' not in dendro:

                    leaf_count = len(dendro['ivl'])
                    leaf_positions = []
                    for i in range(leaf_count):

                        leaf_x = []
                        for j, (xx, yy) in enumerate(zip(dendro['icoord'], dendro['dcoord'])):
                            if yy[0] == 0.0 and yy[-1] == 0.0:
                                leaf_x.append(xx[0])
                        leaf_positions = sorted(leaf_x)
                        if len(leaf_positions) == leaf_count:
                            break


                    if len(leaf_positions) != leaf_count:
                        leaf_positions = [10 * i for i in range(leaf_count)]
                else:
                    leaf_positions = dendro['ivl_positions']

                fig_dendro.update_layout(
                    title='Asset Clustering Dendrogram',
                    xaxis=dict(
                        tickmode='array',
                        tickvals=leaf_positions,
                        ticktext=dendro['ivl'],
                        tickangle=45
                    ),
                    yaxis=dict(title='Distance'),
                    height=500
                )

                st.plotly_chart(fig_dendro, use_container_width=True)

            except ImportError:
                st.error("Cluster analysis requires the scipy package to be installed.")
                st.info("Install scipy with the command: pip install scipy")
            except Exception as e:
                st.error(f"Failed to perform cluster analysis: {str(e)}")
                st.info("Cluster analysis requires the scipy package to be installed.")

        with corr_tabs[3]:
            st.subheader("Dynamics of correlations")

            if isinstance(returns.index, pd.DatetimeIndex) and benchmark in returns.columns:
                # Selecting assets for dynamics analysis
                selected_corr_assets = st.multiselect(
                    "Select assets for correlation dynamics analysis",
                    options=[t for t in weights.keys() if t != benchmark],
                    default=list(weights.keys())[:3] if len(weights) > 3 else list(weights.keys())
                )

                window_size = st.slider(
                    "Window size (days) for calculating moving correlation",
                    min_value=30,
                    max_value=252,
                    value=60,
                    step=10
                )

                if selected_corr_assets:

                    rolling_corrs = pd.DataFrame(index=returns.index)

                    for ticker in selected_corr_assets:
                        if ticker in returns.columns:

                            rolling_corrs[ticker] = returns[ticker].rolling(window=window_size).corr(returns[benchmark])

                    rolling_corrs = rolling_corrs.dropna()

                    if not rolling_corrs.empty:

                        fig_rolling_corrs = go.Figure()

                        for ticker in rolling_corrs.columns:
                            fig_rolling_corrs.add_trace(go.Scatter(
                                x=rolling_corrs.index,
                                y=rolling_corrs[ticker],
                                mode='lines',
                                name=f"{ticker} with {benchmark}"
                            ))

                        fig_rolling_corrs.update_layout(
                            title=f'Rolling correlation of assets with {benchmark} (window {window_size} days)',
                            xaxis_title='Date',
                            yaxis_title='Correlation',
                            yaxis=dict(range=[-1, 1]),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            hovermode='x unified'
                        )

                        # Add lines for correlation levels
                        fig_rolling_corrs.add_hline(y=0.8, line_dash="dash", line_color="red",
                                                    annotation_text="Strong correlation (0.8)")
                        fig_rolling_corrs.add_hline(y=0.5, line_dash="dash", line_color="orange",
                                                    annotation_text="Average correlation (0.5)")
                        fig_rolling_corrs.add_hline(y=0, line_dash="dash", line_color="green",
                                                    annotation_text="No correlation (0)")
                        fig_rolling_corrs.add_hline(y=-0.5, line_dash="dash", line_color="blue",
                                                    annotation_text="Average negative (-0.5)")

                        st.plotly_chart(fig_rolling_corrs, use_container_width=True)

                    # Calculate the average correlation of the portfolio with the benchmark
                    weighted_corr = pd.Series(index=returns.index)

                    valid_dates = returns.index

                    window_indices = range(len(valid_dates) - window_size + 1)

                    for i in window_indices:
                        window_start = i
                        window_end = i + window_size
                        window_dates = valid_dates[window_start:window_end]

                        window_corrs = {}
                        for ticker in weights:
                            if ticker in returns.columns and ticker != benchmark:
                                ticker_returns = returns.loc[window_dates, ticker]
                                bench_returns = returns.loc[window_dates, benchmark]
                                if not ticker_returns.empty and not bench_returns.empty:
                                    window_corrs[ticker] = ticker_returns.corr(bench_returns)

                        if window_corrs:
                            total_weight = sum(weights[t] for t in window_corrs)
                            if total_weight > 0:
                                avg_corr = sum(window_corrs[t] * weights[t] for t in window_corrs) / total_weight
                                if i + window_size <= len(valid_dates):
                                    weighted_corr[valid_dates[i + window_size - 1]] = avg_corr

                    weighted_corr = weighted_corr.dropna()

                    if not weighted_corr.empty:

                        fig_avg_corr = go.Figure()

                        fig_avg_corr.add_trace(go.Scatter(
                            x=weighted_corr.index,
                            y=weighted_corr.values,
                            mode='lines',
                            line=dict(color='purple', width=2),
                            name='Average portfolio correlation'
                        ))

                        fig_avg_corr.update_layout(
                            title=f'Average weighted portfolio correlation with {benchmark} (window {window_size} days)',
                            xaxis_title='Date',
                            yaxis_title='Correlation',
                            yaxis=dict(range=[-1, 1]),
                            hovermode='x unified'
                        )

                        fig_avg_corr.add_hline(y=0.8, line_dash="dash", line_color="red",
                                               annotation_text="Strong correlation (0.8)")
                        fig_avg_corr.add_hline(y=0.5, line_dash="dash", line_color="orange",
                                               annotation_text="Average correlation (0.5)")
                        fig_avg_corr.add_hline(y=0, line_dash="dash", line_color="green",
                                               annotation_text="No correlation (0)")

                        st.plotly_chart(fig_avg_corr, use_container_width=True)

    with tabs[5]:
        st.subheader("Stress testing")

        # Create sub-tabs for different types of stress tests
        stress_tabs = st.tabs([
            "Historical Scenarios",
            "Custom Scenarios",
            "Sensitivity Analysis",
            "Extreme Scenarios"
        ])

        with stress_tabs[0]:
            st.subheader("Historical Stress Testing Scenarios")

            scenarios = [
                "financial_crisis_2008",
                "covid_2020",
                "tech_bubble_2000",
                "black_monday_1987",
                "inflation_shock",
                "rate_hike_2018",
                "moderate_recession",
                "severe_recession"
            ]

            scenario_names = {
                "financial_crisis_2008": "Financial Crisis 2008",
                "covid_2020": "COVID-19 Pandemic (2020)",
                "tech_bubble_2000": "Dot-com Crash (2000-2002)",
                "black_monday_1987": "Black Monday (1987)",
                "inflation_shock": "Inflation Shock (2021-2022)",
                "rate_hike_2018": "Fed Rate Hike (2018)",
                "moderate_recession": "Moderate Recession",
                "severe_recession": "Severe Recession"
            }

            col1, col2 = st.columns(2)

            with col1:
                selected_scenario = st.selectbox(
                    "Select a stress testing scenario",
                    options=scenarios,
                    format_func=lambda x: scenario_names.get(x, x)
                )

            with col2:
                portfolio_value = st.number_input(
                    "Portfolio value ($)",
                    min_value=1000,
                    value=10000,
                    step=1000
                )

            # Conducting improved stress testing with historical data
            if st.button("Conduct a historical stress test"):
                with st.spinner("Performing historical stress test..."):

                    tickers = [asset['ticker'] for asset in portfolio_data['assets']]

                    weights = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

                    stress_test_result = RiskManagement.perform_historical_stress_test(
                        data_fetcher, tickers, weights, selected_scenario, portfolio_value, portfolio_data
                    )

                if 'error' in stress_test_result:
                    st.error(f"Error while running stress test: {stress_test_result['error']}")
                else:
                    st.subheader(f"Stress test results: {stress_test_result['scenario_name']}")
                    st.write(f"**Period:** {stress_test_result['period']}")
                    st.write(f"**Description:** {stress_test_result['scenario_description']}")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Change in value",
                            f"${stress_test_result['portfolio_loss']:.2f}",
                            f"{stress_test_result['shock_percentage'] * 100:.1f}%",
                            delta_color="inverse"
                        )

                    with col2:
                        st.metric(
                            "Cost after shock",
                            f"${stress_test_result['portfolio_after_shock']:.2f}"
                        )

                    with col3:
                        st.metric(
                            "Expected recovery time",
                            f"{stress_test_result['recovery_months']:.1f} мес."
                        )

                    position_data = []
                    for ticker, impact in stress_test_result['position_impacts'].items():
                        position_data.append({
                            'Ticker': ticker,
                            'Weight (%)': impact['weight'] * 100,
                            'Price change (%)': impact['price_change'] * 100,
                            'Position cost ($)': impact['position_value'],
                            'Losses ($)': impact['position_loss']
                        })

                    position_df = pd.DataFrame(position_data)

                    # Sort by losses
                    position_df = position_df.sort_values('Losses ($)', ascending=True)

                    st.subheader("Impact on individual assets")
                    st.dataframe(
                        style_dataframe(position_df, {
                            'Weight (%)': '{:.2f}%',
                            'Price change (%)': '{:.2f}%',
                            'Position cost ($)': '${:.2f}',
                            'Losses ($)': '${:.2f}'
                        }),
                        use_container_width=True,
                        height=min(35 * (len(position_df) + 1), 350)
                    )

                    # Visualization of losses by assets
                    fig_pos_loss = px.bar(
                        position_df,
                        x='Ticker',
                        y='Losses ($)',
                        title='Asset losses',
                        color='Price change (%)',
                        color_continuous_scale='RdYlGn'
                    )

                    st.plotly_chart(fig_pos_loss, use_container_width=True)

                    st.subheader("Stress Test Visualization")

                    fig_stress = go.Figure()

                    months = list(range(-1, int(stress_test_result['recovery_months']) + 2))
                    values = []

                    values.append(portfolio_value)

                    values.append(stress_test_result['portfolio_after_shock'])

                    recovery_rate = (portfolio_value - stress_test_result['portfolio_after_shock']) / \
                                    stress_test_result[
                                        'recovery_months'] if stress_test_result['recovery_months'] > 0 else 0

                    for i in range(1, len(months) - 1):
                        values.append(stress_test_result['portfolio_after_shock'] + recovery_rate * i)

                    fig_stress.add_trace(go.Scatter(
                        x=months,
                        y=values,
                        mode='lines+markers',
                        name='Portfolio value'
                    ))

                    fig_stress.add_shape(
                        type="line",
                        x0=months[0],
                        y0=portfolio_value,
                        x1=months[-1],
                        y1=portfolio_value,
                        line=dict(color="green", width=2, dash="dot"),
                        name="Original cost"
                    )

                    fig_stress.update_layout(
                        title=f"Stress test: {stress_test_result['scenario_name']}",
                        xaxis_title="Months",
                        yaxis_title="Portfolio value ($)",
                        hovermode="x"
                    )

                    st.plotly_chart(fig_stress, use_container_width=True)

                    # Compare impact at asset class or sector level where data is available
                    sectors = {}
                    for asset in portfolio_data['assets']:
                        if 'sector' in asset and asset['sector'] != 'N/A':
                            sector = asset['sector']
                            ticker = asset['ticker']
                            if sector not in sectors:
                                sectors[sector] = {
                                    'weight': 0,
                                    'impact': 0,
                                    'tickers': []
                                }

                            # Add ticker information
                            sectors[sector]['tickers'].append(ticker)
                            sectors[sector]['weight'] += weights[ticker]

                            # Adding to the overall impact
                            if ticker in stress_test_result['position_impacts']:
                                impact = stress_test_result['position_impacts'][ticker]['price_change'] * weights[
                                    ticker]
                                sectors[sector]['impact'] += impact

                    if sectors:

                        sector_data = []
                        for sector, data in sectors.items():
                            sector_data.append({
                                'Sector': sector,
                                'Weight (%)': data['weight'] * 100,
                                'Price change (%)': (data['impact'] / data['weight']) * 100 if data['weight'] > 0 else 0,
                                'Total impact (%)': data['impact'] * 100,
                                'Number of assets': len(data['tickers'])
                            })

                        sector_df = pd.DataFrame(sector_data)
                        sector_df = sector_df.sort_values('Total impact (%)', ascending=True)

                        st.subheader("Impact at the sector level")
                        st.dataframe(sector_df.style.format({
                            'Weight (%)': '{:.2f}%',
                            'Price change (%)': '{:.2f}%',
                            'Total impact (%)': '{:.2f}%'
                        }), use_container_width=True)

                        # Visualize Impact by Sector
                        fig_sector = px.bar(
                            sector_df,
                            x='Sector',
                            y='Total impact (%)',
                            title='Impact of stress scenario by sector',
                            color='Price change (%)',
                            color_continuous_scale='RdYlGn'
                        )

                        st.plotly_chart(fig_sector, use_container_width=True)

                    st.subheader("Advanced Stress Test Visualization")

                    if 'scenario' in stress_test_result and stress_test_result['scenario'] in historical_crisis_context:
                        st.subheader("Historical context")
                        display_historical_context(stress_test_result['scenario'])

                    try:

                        interactive_chart = create_interactive_stress_impact_chart(
                            {stress_test_result['scenario']: stress_test_result},
                            portfolio_value
                        )
                        st.plotly_chart(interactive_chart, use_container_width=True)

                        # Тепловые карты влияния
                        fig_assets, fig_sectors = create_stress_impact_heatmap(
                            portfolio_data,
                            {stress_test_result['scenario']: stress_test_result}
                        )
                        st.plotly_chart(fig_assets, use_container_width=True)
                        st.plotly_chart(fig_sectors, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating visualizations: {e}")

        with stress_tabs[1]:
            st.subheader("User Stress Test")

            st.write("""
            Create your own stress testing scenario by specifying shock changes 
            for individual assets or asset classes. Historical correlations and asset betas are taken into account.
            """)

            # Entering the portfolio value
            custom_portfolio_value = st.number_input(
                "Portfolio value ($)",
                min_value=1000,
                value=10000,
                step=1000,
                key="custom_portfolio_value"
            )

            shock_type = st.radio(
                "User scenario type",
                ["By individual assets", "By sectors", "Combined shock"]
            )

            with st.expander("Advanced Analysis Settings"):
                col1, col2 = st.columns(2)

                with col1:
                    use_correlations = st.checkbox("Take into account correlations between assets", value=True,
                                                   help="Considers how a shock to one asset affects other assets based on historical correlations")

                with col2:
                    use_beta = st.checkbox("Consider the beta of assets", value=True,
                                           help="Takes into account the sensitivity of each asset to market movements")

            custom_shocks = {}
            asset_sectors = {}

            if shock_type == "By individual assets":

                st.subheader("Set shocks for individual assets")

                market_shock = st.slider(
                    "General market shock (%)",
                    min_value=-90,
                    max_value=50,
                    value=0,
                    step=5,
                    key="market_shock"
                ) / 100

                custom_shocks['market'] = market_shock
                custom_shocks['assets'] = {}

                col1, col2 = st.columns(2)

                assets_list = list(portfolio_data['assets'])
                mid_point = len(assets_list) // 2

                with col1:
                    for i, asset in enumerate(assets_list[:mid_point]):
                        ticker = asset['ticker']

                        sector = asset.get('sector', 'N/A')
                        if sector != 'N/A':
                            asset_sectors[ticker] = sector

                        asset_shock = st.slider(
                            f"{ticker} - {asset.get('name', '')} (%)",
                            min_value=-90,
                            max_value=50,
                            value=0,
                            step=5,
                            key=f"shock_asset_{ticker}"
                        ) / 100

                        if asset_shock != 0:
                            custom_shocks['assets'][ticker] = asset_shock

                with col2:
                    for i, asset in enumerate(assets_list[mid_point:]):
                        ticker = asset['ticker']

                        sector = asset.get('sector', 'N/A')
                        if sector != 'N/A':
                            asset_sectors[ticker] = sector

                        asset_shock = st.slider(
                            f"{ticker} - {asset.get('name', '')} (%)",
                            min_value=-90,
                            max_value=50,
                            value=0,
                            step=5,
                            key=f"shock_asset_{ticker}_col2"
                        ) / 100

                        if asset_shock != 0:
                            custom_shocks['assets'][ticker] = asset_shock

            elif shock_type == "By sectors":

                sectors = {}
                for asset in portfolio_data['assets']:
                    if 'sector' in asset and asset['sector'] != 'N/A':
                        sector = asset['sector']
                        ticker = asset['ticker']
                        if sector not in sectors:
                            sectors[sector] = []
                        sectors[sector].append(ticker)

                        asset_sectors[ticker] = sector

                st.subheader("Set shocks for sectors")

                market_shock = st.slider(
                    "Total Market Shock (%)",
                    min_value=-90,
                    max_value=50,
                    value=0,
                    step=5,
                    key="market_shock_sectors"
                ) / 100

                custom_shocks['market'] = market_shock

                if sectors:
                    for sector, tickers in sectors.items():
                        sector_shock = st.slider(
                            f"Shock for sector {sector} ({len(tickers)} assets) (%)",
                            min_value=-90,
                            max_value=50,
                            value=0,
                            step=5,
                            key=f"shock_sector_{sector}"
                        ) / 100

                        if sector_shock != 0:
                            custom_shocks[sector] = sector_shock
                else:
                    st.warning("Sector information for assets is not available. Use individual asset shock.")

            else:
                st.subheader("Combined shock to market, sectors and assets")

                market_shock = st.slider(
                    "Total Market Shock (%)",
                    min_value=-90,
                    max_value=50,
                    value=-25,
                    step=5,
                    key="market_shock_combined"
                ) / 100

                custom_shocks['market'] = market_shock

                sectors = {}
                for asset in portfolio_data['assets']:
                    if 'sector' in asset and asset['sector'] != 'N/A':
                        sector = asset['sector']
                        ticker = asset['ticker']
                        if sector not in sectors:
                            sectors[sector] = []
                        sectors[sector].append(ticker)

                        asset_sectors[ticker] = sector

                if sectors:
                    st.subheader("Sector shocks:")

                    key_sectors = list(sectors.keys())[:min(5, len(sectors))]

                    for sector in key_sectors:
                        sector_shock = st.slider(
                            f"Additional shock for sector {sector} (%)",
                            min_value=-50,
                            max_value=30,
                            value=0,
                            step=5,
                            key=f"shock_sector_combined_{sector}"
                        ) / 100

                        if sector_shock != 0:
                            custom_shocks[sector] = sector_shock

                st.subheader("Shocks for individual assets (optional):")

                custom_assets = st.multiselect(
                    "Select assets to define individual shocks",
                    options=[asset['ticker'] for asset in portfolio_data['assets']],
                    default=[]
                )

                custom_shocks['assets'] = {}

                if custom_assets:
                    for ticker in custom_assets:
                        asset_name = next((asset.get('name', ticker) for asset in portfolio_data['assets']
                                           if asset['ticker'] == ticker), ticker)

                        asset_shock = st.slider(
                            f"Additional shock for {ticker} - {asset_name} (%)",
                            min_value=-50,
                            max_value=30,
                            value=0,
                            step=5,
                            key=f"shock_asset_combined_{ticker}"
                        ) / 100

                        if asset_shock != 0:
                            custom_shocks['assets'][ticker] = asset_shock

            # Button to run custom stress test
            if st.button("Run a custom stress test"):
                with st.spinner("Performing a custom stress test..."):

                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                    end_date = datetime.now().strftime('%Y-%m-%d')

                    tickers = [asset['ticker'] for asset in portfolio_data['assets']]
                    weights = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

                    if 'SPY' not in tickers:
                        tickers.append('SPY')

                    price_data = data_fetcher.get_batch_data(tickers, start_date, end_date)

                    if not price_data or all(df.empty for df in price_data.values()):
                        st.error(
                            "Failed to load historical data. Please check tickers or change period.")
                    else:

                        close_prices = pd.DataFrame()

                        for ticker, df in price_data.items():
                            if not df.empty:
                                if 'Adj Close' in df.columns:
                                    close_prices[ticker] = df['Adj Close']
                                elif 'Close' in df.columns:
                                    close_prices[ticker] = df['Close']

                        returns = PortfolioAnalytics.calculate_returns(close_prices)

                        stress_test_result = RiskManagement.perform_advanced_custom_stress_test(
                            returns=returns,
                            weights=weights,
                            custom_shocks=custom_shocks,
                            asset_sectors=asset_sectors,
                            portfolio_value=custom_portfolio_value,
                            correlation_adjusted=use_correlations,
                            use_beta=use_beta
                        )

                        if 'error' in stress_test_result:
                            st.error(f"Error while running stress test: {stress_test_result['error']}")
                        else:
                            st.subheader("User Stress Test Results")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    "Change in value",
                                    f"${stress_test_result['portfolio_loss']:.2f}",
                                    f"{stress_test_result['loss_percentage'] * 100:.1f}%",
                                    delta_color="inverse"
                                )

                            with col2:
                                st.metric(
                                    "Cost after shock",
                                    f"${stress_test_result['portfolio_after_shock']:.2f}"
                                )

                            with col3:
                                st.metric(
                                    "Expected recovery time",
                                    f"{stress_test_result['recovery_months']:.1f} month"

                                )
                            # Create a DataFrame for detailed asset analysis
                            detailed_data = []
                            for ticker, impact in stress_test_result['detailed_impacts'].items():
                                detailed_data.append({
                                    'Ticker': ticker,
                                    'Weight (%)': impact['weight'] * 100,
                                    'Beta': impact['beta'],
                                    'Price change (%)': impact['shock_percentage'] * 100,
                                    'Position cost ($)': impact['position_value'],
                                    'Losses ($)': impact['position_loss']
                                })

                            detailed_df = pd.DataFrame(detailed_data)

                            # Sort by losses
                            detailed_df = detailed_df.sort_values('Losses ($)', ascending=True)

                            st.subheader("Detailed analysis of the impact on assets")
                            st.dataframe(detailed_df.style.format({
                                'Weight (%)': '{:.2f}%',
                                'Beta': '{:.2f}',
                                'Price change (%)': '{:.2f}%',
                                'Position cost ($)': '${:.2f}',
                                'Losses ($)': '${:.2f}'
                            }), use_container_width=True)

                            # Visualization of losses by assets
                            fig_pos_loss = px.bar(
                                detailed_df,
                                x='Ticker',
                                y='Losses ($)',
                                title='Asset losses',
                                color='Price change (%)',
                                color_continuous_scale='RdYlGn',
                                hover_data=['Weight (%)', 'Beta']
                            )

                            st.plotly_chart(fig_pos_loss, use_container_width=True)

                            if asset_sectors:
                                sector_impacts = {}
                                for ticker, impact in stress_test_result['detailed_impacts'].items():
                                    if ticker in asset_sectors:
                                        sector = asset_sectors[ticker]
                                        if sector not in sector_impacts:
                                            sector_impacts[sector] = {
                                                'weight': 0,
                                                'loss': 0,
                                                'value': 0,
                                                'tickers_count': 0
                                            }

                                        sector_impacts[sector]['weight'] += impact['weight']
                                        sector_impacts[sector]['loss'] += impact['position_loss']
                                        sector_impacts[sector]['value'] += impact['position_value']
                                        sector_impacts[sector]['tickers_count'] += 1

                                sector_data = []
                                for sector, data in sector_impacts.items():
                                    sector_data.append({
                                        'Sector': sector,
                                        'Weight (%)': data['weight'] * 100,
                                        'Number of assets': data['tickers_count'],
                                        'Price ($)': data['value'],
                                        'Losses ($)': data['loss'],
                                        'Price change (%)': (data['loss'] / data['value'] * 100) if data[
                                                                                                     'value'] > 0 else 0
                                    })

                                sector_df = pd.DataFrame(sector_data)
                                sector_df = sector_df.sort_values('Losses ($)', ascending=True)

                                st.subheader("Sector analysis")
                                st.dataframe(sector_df.style.format({
                                    'Weight (%)': '{:.2f}%',
                                    'Price ($)': '${:.2f}',
                                    'Losses ($)': '${:.2f}',
                                    'Price change (%)': '{:.2f}%'
                                }), use_container_width=True)

                                # Visualization of losses by sectors
                                fig_sector_loss = px.bar(
                                    sector_df,
                                    x='Sector',
                                    y='Losses ($)',
                                    title='Losses by sector',
                                    color='Price change (%)',
                                    color_continuous_scale='RdYlGn',
                                    hover_data=['Weight (%)', 'Number of assets']
                                )

                                st.plotly_chart(fig_sector_loss, use_container_width=True)

                            # Visualization of recovery
                            st.subheader("Recovery forecast")

                            months = list(range(-1, int(stress_test_result['recovery_months']) + 2))
                            values = []

                            values.append(custom_portfolio_value)

                            values.append(stress_test_result['portfolio_after_shock'])

                            recovery_rate = (custom_portfolio_value - stress_test_result['portfolio_after_shock']) / \
                                            stress_test_result[
                                                'recovery_months'] if stress_test_result['recovery_months'] > 0 else 0

                            for i in range(1, len(months) - 1):
                                values.append(stress_test_result['portfolio_after_shock'] + recovery_rate * i)

                            fig_recovery = go.Figure()

                            fig_recovery.add_trace(go.Scatter(
                                x=months,
                                y=values,
                                mode='lines+markers',
                                name='Cost forecast'
                            ))

                            fig_recovery.add_shape(
                                type="line",
                                x0=months[0],
                                y0=custom_portfolio_value,
                                x1=months[-1],
                                y1=custom_portfolio_value,
                                line=dict(color="green", width=2, dash="dot"),
                                name="Original cost"
                            )

                            fig_recovery.update_layout(
                                title="Portfolio Recovery Forecast",
                                xaxis_title="Months",
                                yaxis_title="Portfolio value ($)",
                                hovermode="x unified"
                            )

                            st.plotly_chart(fig_recovery, use_container_width=True)

                            st.subheader("Stress test parameters")

                            shock_summary = []

                            if 'market' in custom_shocks:
                                shock_summary.append({
                                    'Type': 'Market Shock',
                                    'Object': 'The whole market',
                                    'The Shock Given (%)': custom_shocks['market'] * 100
                                })

                            # Adding sector shocks
                            for key, value in custom_shocks.items():
                                if key != 'market' and key != 'assets':
                                    shock_summary.append({
                                        'Type': 'Sector shock',
                                        'Object': key,
                                        'The Shock Given (%)': value * 100
                                    })

                            # Add shocks for individual assets
                            if 'assets' in custom_shocks:
                                for ticker, shock in custom_shocks['assets'].items():
                                    shock_summary.append({
                                        'Type': 'Asset',
                                        'Object': ticker,
                                        'The Shock Given (%)': shock * 100
                                    })

                            if shock_summary:
                                shock_df = pd.DataFrame(shock_summary)
                                st.dataframe(shock_df.style.format({
                                    'The Shock Given (%)': '{:.1f}%'
                                }), use_container_width=True)

                                st.info("""
                                **How to interpret the results:**

                                1. **Correlations and Betas**: The calculation takes into account historical correlations between assets and their betas relative to the market, which gives a more realistic picture than simply summing individual shocks.

                                2. **Recovery Time**: Estimate based on average annual market return of 7%. Actual recovery time may vary depending on market conditions.

                                3. **Asset Impact**: Assets with high beta and/or strong correlation to negatively shocked sectors will experience a greater impact.
                                """)
                            else:
                                st.warning("No shock values ​​specified. Results may be uninformative.")

        with stress_tabs[2]:
            st.subheader("Sensitivity analysis")

            st.write("""
            Sensitivity analysis shows how the value of a portfolio will change when key risk factors change.
            """)

            factors = st.multiselect(
                "Select factors for sensitivity analysis",
                options=["Interest Rates", "Inflation", "Oil Prices", "Dollar Rate", "Recession"],
                default=["Interest Rates", "Inflation"]
            )

            if factors:
                # Create a dictionary of factors and their impact on different asset classes
                factor_impacts = {
                    "Interest Rates": {
                        "Stocks": -0.05,  # -5% for every 1% increase in rates
                        "Bonds": -0.1,  # -10% for every 1% increase in rates
                        "Real Estate": -0.08,  # -8% for every 1% increase in rates
                        "Gold": -0.03,  # -3% for every 1% increase in rates
                        "Money Market": 0.01  # +1% for every 1% increase in rates
                    },
                    "Inflation": {
                        "Stocks": -0.02,  # -2% with inflation growth of 1%
                        "Bonds": -0.05,  # -5% with inflation growth of 1%
                        "Real Estate": 0.02,  # +2% with inflation growth of 1%
                        "Gold": 0.05,  # +5% with inflation growth of 1%
                        "Money Market": -0.01  # -1% with inflation growth of 1%
                    },
                    "Oil Prices": {
                        "Stocks": 0.01,  # +1% if oil prices rise by 10%
                        "Bonds": -0.01,  # -1% if oil prices rise by 10%
                        "Real Estate": 0.01,  # +1% if oil prices rise by 10%
                        "Gold": 0.02,  # +2% if oil prices rise by 10%
                        "Money Market": 0  # 0% if oil prices rise by 10%
                    },
                    "Dollar Rate": {
                        "Stocks": -0.02,  # -2% if the dollar strengthens by 5%
                        "Bonds": -0.01,  # -1% if the dollar strengthens by 5%
                        "Real Estate": -0.02,  # -2% if the dollar strengthens by 5%
                        "Gold": -0.03,  # -3% if the dollar strengthens by 5%
                        "Money Market": 0.01  # +1% if the dollar strengthens by 5%
                    },
                    "Recession": {
                        "Stocks": -0.3,  # -30% during recession
                        "Bonds": -0.05,  # -5% during recession
                        "Real Estate": -0.2,  # -20% during recession
                        "Gold": 0.1,  # +10% during recession
                        "Money Market": 0.01  # +1% during recession
                    }
                }

                # Determine the asset class for each ticker
                asset_classes = {}
                for asset in portfolio_data['assets']:
                    ticker = asset['ticker']
                    if 'asset_class' in asset and asset['asset_class'] != 'N/A':
                        class_name = asset['asset_class']
                    elif 'sector' in asset and asset['sector'] in ['Financial Services', 'Financial', 'Financials']:
                        class_name = "Stocks"
                    else:

                        class_name = "Stocks"

                    asset_classes[ticker] = class_name

                # Create a sensitivity results table
                sensitivity_results = []

                for factor in factors:
                    # For each factor we determine the range of changes
                    if factor == "Interest Rates":
                        changes = [0.25, 0.5, 1.0, 1.5, 2.0]
                        change_label = "p.p."
                    elif factor == "Inflation":
                        changes = [1.0, 2.0, 3.0, 4.0, 5.0]
                        change_label = "p.p."
                    elif factor == "Oil prices":
                        changes = [10, 20, 30, 40, 50]
                        change_label = "%"
                    elif factor == "Dollar Rate":
                        changes = [5, 10, 15, 20, 25]
                        change_label = "%"
                    elif factor == "Recession":
                        changes = [1]
                        change_label = "x"
                    else:
                        changes = [1]
                        change_label = ""

                    # We calculate the impact of each change in the factor
                    for change in changes:
                        portfolio_impact = 0

                        for ticker, weight in weights.items():
                            asset_class = asset_classes.get(ticker, "Stocks")

                            if asset_class in ["Equity", "Stock", "Stocks"]:
                                asset_class = "Stocks"
                            elif asset_class in ["Bond", "Bonds", "Fixed Income"]:
                                asset_class = "Bonds"
                            elif asset_class in ["Real Estate", "REIT"]:
                                asset_class = "Real Estate"
                            elif asset_class in ["Gold", "Precious Metals"]:
                                asset_class = "Gold"
                            elif asset_class in ["Cash", "Money Market"]:
                                asset_class = "Money Market"

                            # Find the corresponding influence coefficient
                            impact_coef = factor_impacts[factor].get(asset_class, 0)

                            # Calculate the impact on the asset
                            asset_impact = impact_coef * change

                            # Add the weighted impact to the overall portfolio impact
                            portfolio_impact += asset_impact * weight

                        sensitivity_results.append({
                            'Factor': factor,
                            'Price change': f"+{change} {change_label}" if change_label else "Occurrence",
                            'Impact on portfolio (%)': portfolio_impact * 100
                        })

                sensitivity_df = pd.DataFrame(sensitivity_results)
                sensitivity_df = sensitivity_df.sort_values('Impact on portfolio (%)', ascending=True)

                # Visualization of sensitivity results
                fig_sensitivity = px.bar(
                    sensitivity_df,
                    x='Impact on portfolio (%)',
                    y='Factor',
                    color='Impact on portfolio (%)',
                    color_continuous_scale='RdYlGn',
                    text='Price change',
                    orientation='h',
                    title='Portfolio sensitivity to risk factors'
                )

                fig_sensitivity.update_layout(
                    xaxis_title='Impact on portfolio performance (%)',
                    yaxis_title='Risk factor',
                    height=500
                )

                st.plotly_chart(fig_sensitivity, use_container_width=True)

                # Display the table with the results
                st.dataframe(sensitivity_df.style.format({
                    'Impact on portfolio (%)': '{:.2f}%'
                }), use_container_width=True)

        with stress_tabs[3]:
            st.subheader("Extreme scenarios")

            st.write("""
            Extreme scenario analysis assesses the impact of unlikely but possible events on portfolio value using statistical modeling.
            """)

            # Entering the portfolio value
            extreme_portfolio_value = st.number_input(
                "Portfolio value ($)",
                min_value=1000,
                value=10000,
                step=1000,
                key="extreme_portfolio_value"
            )

            # We define extreme scenarios with more detailed descriptions
            extreme_scenarios = {
                "market_crash_50": {
                    "name": "Market Crash (-50%)",
                    "description": "A large-scale market crash similar to the 2008 crisis, leading to a 50% drop in major indices.",
                    "impact": -0.5,
                    "details": "Similar to the period September 2008 - March 2009, when the S&P 500 lost more than 50%. Events of this severity happen approximately once every 50-80 years."
                },
                "severe_recession_35": {
                    "name": "Severe Recession (-35%)",
                    "description": "A severe economic recession with a prolonged contraction in GDP and high unemployment.",
                    "impact": -0.35,
                    "details": "Similar to the start of the COVID-19 pandemic (February-March 2020), when markets fell by about 35%. The likelihood of such events is about once every 10-15 years."
                },
                "inflation_shock_25": {
                    "name": "Inflation shock (+8%)",
                    "description": "A sharp jump in inflation above 8%, forcing central banks to raise interest rates aggressively.",
                    "impact": -0.25,
                    "details": "Historical examples: the inflation shock of the 1970s and the period 2021-2022. Particularly negative for long-term bonds and growth stocks."
                },
                "geopolitical_crisis_20": {
                    "name": "Geopolitical Crisis",
                    "description": "A major international conflict or crisis that affects global trade and energy markets.",
                    "impact": -0.20,
                    "details": "Such as the 1973 oil crisis or tensions between major powers. Often causes commodity and gold prices to rise while most stocks fall."
                },
                "tech_bubble_burst_40": {
                    "name": "Tech Bubble Burst",
                    "description": "A sharp correction in overvalued tech stocks, similar to the dot-com crash of 2000.",
                    "impact": -0.40,
                    "details": "Tech and growth stocks could lose 60-80% of their value, while value stocks and defensive sectors will do better. Likely about once every 20-25 years."
                },
                "currency_crisis_15": {
                    "name": "Currency Crisis",
                    "description": "A major devaluation of one or more global currencies that causes a chain reaction across markets.",
                    "impact": -0.15,
                    "details": "Examples include the Asian financial crisis of 1997 or the European currency crisis of 1992. Usually regional in nature, but can spread globally."
                }
            }

            # Selecting scenarios for analysis
            selected_extreme_scenarios = st.multiselect(
                "Select extreme scenarios for analysis",
                options=list(extreme_scenarios.keys()),
                default=list(extreme_scenarios.keys())[:3],
                format_func=lambda x: extreme_scenarios[x]["name"]
            )

            # Optional setting of simulation parameters
            with st.expander("Simulation parameters"):
                col1, col2 = st.columns(2)

                with col1:
                    confidence_level = st.slider(
                        "Modeling confidence level (%)",
                        min_value=80,
                        max_value=99,
                        value=95,
                        format="%d%%"
                    )

                    monte_carlo_sims = st.slider(
                        "Number of Monte Carlo simulations",
                        min_value=100,
                        max_value=5000,
                        value=1000,
                        step=100
                    )

                with col2:
                    recovery_annual_return = st.slider(
                        "Expected annual return for recovery (%)",
                        min_value=3,
                        max_value=12,
                        value=7,
                        step=1
                    ) / 100

                    fat_tail_factor = st.slider(
                        "Heavy tail ratio",
                        min_value=1.0,
                        max_value=3.0,
                        value=1.5,
                        step=0.1,
                        help="Increases the probability of extreme events. A value of 1.0 corresponds to a normal distribution."
                    )

            if selected_extreme_scenarios:

                if st.button("Conduct an extreme scenario analysis"):
                    with st.spinner("Performing extreme scenario analysis..."):

                        start_date = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')  # 5 years of history
                        end_date = datetime.now().strftime('%Y-%m-%d')

                        tickers = [asset['ticker'] for asset in portfolio_data['assets']]
                        weights = {asset['ticker']: asset['weight'] for asset in portfolio_data['assets']}

                        if 'SPY' not in tickers:
                            tickers.append('SPY')

                        prices_data = data_fetcher.get_batch_data(tickers, start_date, end_date)

                        if not prices_data or all(df.empty for df in prices_data.values()):
                            st.error(
                                "Failed to load historical data. Please check tickers or change period.")
                        else:

                            close_prices = pd.DataFrame()

                            for ticker, df in prices_data.items():
                                if not df.empty:
                                    if 'Adj Close' in df.columns:
                                        close_prices[ticker] = df['Adj Close']
                                    elif 'Close' in df.columns:
                                        close_prices[ticker] = df['Close']

                            returns = PortfolioAnalytics.calculate_returns(close_prices)
                            portfolio_returns = PortfolioAnalytics.calculate_portfolio_return(returns, weights)

                            # Create a DataFrame to display the results
                            extreme_results = []

                            for scenario_key in selected_extreme_scenarios:
                                scenario = extreme_scenarios[scenario_key]

                                # Basic information about the scenario
                                impact = scenario["impact"]
                                portfolio_loss = extreme_portfolio_value * impact
                                portfolio_after_shock = extreme_portfolio_value + portfolio_loss

                                if not portfolio_returns.empty:

                                    hist_std = portfolio_returns.std()

                                    np.random.seed(42)
                                    sim_returns = []

                                    for _ in range(monte_carlo_sims):

                                        degrees_of_freedom = 4
                                        t_random = np.random.standard_t(degrees_of_freedom)

                                        shock = t_random * hist_std * fat_tail_factor * (
                                                    -impact / 0.5)
                                        sim_returns.append(shock)

                                    var_level = np.percentile(sim_returns, 100 - confidence_level)

                                    cvar_values = [r for r in sim_returns if r <= var_level]
                                    cvar = np.mean(cvar_values) if cvar_values else var_level

                                    var_amount = extreme_portfolio_value * var_level
                                    cvar_amount = extreme_portfolio_value * cvar
                                else:

                                    var_amount = portfolio_loss * 1.2
                                    cvar_amount = portfolio_loss * 1.4

                                daily_return = (1 + recovery_annual_return) ** (1 / 252) - 1

                                if impact < 0:

                                    recovery_days = -np.log(1 + impact) / np.log(1 + daily_return)
                                    recovery_months = recovery_days / 21
                                else:
                                    recovery_days = 0
                                    recovery_months = 0

                                extreme_results.append({
                                    'Scenario': scenario["name"],
                                    'Description': scenario["description"],
                                    'Shock (%)': impact * 100,
                                    'Loss ($)': portfolio_loss,
                                    'Value after shock ($)': portfolio_after_shock,
                                    'VaR at {0}% ($)'.format(confidence_level): var_amount,
                                    'CVaR at {0}% ($)'.format(confidence_level): cvar_amount,
                                    'Recovery (months)': recovery_months,
                                    'key': scenario_key,
                                    'details': scenario["details"]
                                })

                            st.subheader("Results of the analysis of extreme scenarios")

                            result_df = pd.DataFrame(extreme_results)
                            result_df_display = result_df[[
                                'Scenario', 'Shock (%)', 'Loss ($)', 'Value after shock ($)', 'Recovery (months)'
                            ]]

                            # Display the results as a table
                            st.dataframe(result_df_display.style.format({
                                'Shock (%)': '{:.1f}%',
                                'Loss ($)': '${:.2f}',
                                'Value after shock ($)': '${:.2f}',
                                'Recovery (months)': '{:.1f}'
                            }), use_container_width=True)

                            # Visualization of losses by scenarios
                            fig_extreme = px.bar(
                                result_df,
                                x='Scenario',
                                y='Loss ($)',
                                color='Shock (%)',
                                color_continuous_scale='RdYlGn',
                                title='Impact of extreme scenarios on portfolio'
                            )

                            fig_extreme.update_layout(
                                xaxis_title='Scenario',
                                yaxis_title='Loss of Value ($)',
                                height=500
                            )

                            st.plotly_chart(fig_extreme, use_container_width=True)

                            st.subheader("Detailed description of scenarios")

                            for scenario_key in selected_extreme_scenarios:
                                scenario = extreme_scenarios[scenario_key]
                                scenario_result = next((r for r in extreme_results if r['key'] == scenario_key), None)

                                if scenario_result:
                                    with st.expander(f"{scenario['name']} - Details"):
                                        st.write(f"**Description**: {scenario['description']}")
                                        st.write(f"**Details**: {scenario['details']}")

                                        col1, col2, col3 = st.columns(3)

                                        with col1:
                                            st.metric(
                                                "Expected loss",
                                                f"${abs(scenario_result['Loss ($)']):.2f}",
                                                f"{scenario_result['Shock (%)']}%",
                                                delta_color="inverse"
                                            )

                                        with col2:
                                            st.metric(
                                                f"VaR at {confidence_level}%",
                                                f"${abs(scenario_result['VaR at {0}% ($)'.format(confidence_level)]):.2f}"
                                            )

                                        with col3:
                                            st.metric(
                                                "Recovery time",
                                                f"{scenario_result['Recovery (months)']:.1f} мес."
                                            )

                            # Additional information on the distribution of potential losses
                            st.subheader("Distribution of potential losses")

                            sim_losses = []
                            for _ in range(monte_carlo_sims):

                                t_random = np.random.standard_t(4)

                                portfolio_volatility = portfolio_returns.std() * np.sqrt(252)

                                extreme_loss = -abs(
                                    t_random) * portfolio_volatility * fat_tail_factor * extreme_portfolio_value * 0.2
                                sim_losses.append(extreme_loss)

                            fig_dist = px.histogram(
                                sim_losses,
                                nbins=50,
                                title='Distribution of potential extreme losses',
                                labels={'value': 'Loss ($)', 'count': 'Frequency'},
                                color_discrete_sequence=['rgba(255, 0, 0, 0.6)']
                            )

                            # Добавляем вертикальные линии для потерь по сценариям
                            for scenario_result in extreme_results:
                                fig_dist.add_vline(
                                    x=scenario_result['Loss ($)'],
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=scenario_result['Scenario'],
                                    annotation_position="top right"
                                )

                            fig_dist.update_layout(
                                xaxis_title='Potential loss ($)',
                                yaxis_title='Frequency',
                                showlegend=False
                            )

                            st.plotly_chart(fig_dist, use_container_width=True)

                            # Добавление предупреждения об интерпретации
                            st.info("""
                                **Important note about extreme event modeling:**
                                
                                Extreme scenarios represent rare and unlikely events that are difficult to accurately model.
                                Actual losses may differ from those projected. This analysis should be considered illustrative,
                                
                                not an exact forecast. In addition, the impact of extreme events on different asset classes may differ significantly
                                
                                and change over time.
                            """)

    with tabs[6]:
        st.subheader("Rolling metrics")

        rolling_tabs = st.tabs([
            "Volatility",
            "Ratios",
            "Rolling Beta/Alpha",
            "Split Analysis"
        ])

        with rolling_tabs[0]:
            st.subheader("Sliding volatility")

            window_size = st.slider(
                "Window size (days) for volatility",
                min_value=21,
                max_value=252,
                value=63,
                step=21
            )

            if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                rolling_vol = portfolio_returns.rolling(window=window_size).std() * np.sqrt(252) * 100

                if benchmark_returns is not None:
                    benchmark_rolling_vol = benchmark_returns.rolling(window=window_size).std() * np.sqrt(252) * 100

                    fig_rolling_vol = go.Figure()

                    fig_rolling_vol.add_trace(go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol.values,
                        mode='lines',
                        name='Portfolio',
                        line=dict(color='blue', width=2)
                    ))

                    fig_rolling_vol.add_trace(go.Scatter(
                        x=benchmark_rolling_vol.index,
                        y=benchmark_rolling_vol.values,
                        mode='lines',
                        name=benchmark,
                        line=dict(color='orange', width=2, dash='dash')
                    ))

                    fig_rolling_vol.update_layout(
                        title=f'Moving volatility ({window_size} days)',
                        xaxis_title='Date',
                        yaxis_title='Volatility (%)',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig_rolling_vol, use_container_width=True)

                    st.subheader("Volatility statistics")

                    vol_stats = pd.DataFrame({
                        'Metric': ['Avg Volatility (%)', 'Median Volatility (%)', 'Min Volatility (%)',
                                   'Max Volatility (%)'],
                        'Portfolio': [
                            rolling_vol.mean(),
                            rolling_vol.median(),
                            rolling_vol.min(),
                            rolling_vol.max()
                        ],
                        'Benchmark': [
                            benchmark_rolling_vol.mean(),
                            benchmark_rolling_vol.median(),
                            benchmark_rolling_vol.min(),
                            benchmark_rolling_vol.max()
                        ]
                    })

                    st.dataframe(vol_stats.style.format({
                        'Portfolio': '{:.2f}',
                        'Benchmark': '{:.2f}'
                    }), use_container_width=True)

        with rolling_tabs[1]:
            st.subheader("Sliding risk coefficients")

            # Параметры расчета
            coef_window_size = st.slider(
                "Window size (days) for coefficients",
                min_value=63,
                max_value=252,
                value=126,
                step=21
            )

            if isinstance(portfolio_returns.index, pd.DatetimeIndex):

                def rolling_sharpe_sortino(returns, window, risk_free=0):
                    rolling_sharpe = []
                    rolling_sortino = []

                    for i in range(window, len(returns) + 1):
                        window_returns = returns.iloc[i - window:i]

                        # Sharpe
                        excess_returns = window_returns - risk_free / 252
                        sharpe = excess_returns.mean() / window_returns.std() * np.sqrt(252)
                        rolling_sharpe.append(sharpe)

                        # Sortino
                        negative_returns = window_returns[window_returns < risk_free / 252]
                        if len(negative_returns) > 0:
                            downside_deviation = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(252)
                            sortino = excess_returns.mean() * 252 / downside_deviation if downside_deviation > 0 else np.nan
                        else:
                            sortino = np.nan

                        rolling_sortino.append(sortino)

                    return pd.Series(rolling_sharpe, index=returns.index[window - 1:]), pd.Series(rolling_sortino,
                                                                                                  index=returns.index[
                                                                                                        window - 1:])

                # Calculate sliding coefficients
                rolling_sharpe, rolling_sortino = rolling_sharpe_sortino(
                    portfolio_returns, coef_window_size, config.RISK_FREE_RATE
                )

                # Calculate for benchmark if available
                if benchmark_returns is not None:
                    benchmark_rolling_sharpe, benchmark_rolling_sortino = rolling_sharpe_sortino(
                        benchmark_returns, coef_window_size, config.RISK_FREE_RATE
                    )

                    fig_rolling_sharpe = go.Figure()

                    fig_rolling_sharpe.add_trace(go.Scatter(
                        x=rolling_sharpe.index,
                        y=rolling_sharpe.values,
                        mode='lines',
                        name='Portfolio',
                        line=dict(color='blue', width=2)
                    ))

                    fig_rolling_sharpe.add_trace(go.Scatter(
                        x=benchmark_rolling_sharpe.index,
                        y=benchmark_rolling_sharpe.values,
                        mode='lines',
                        name=benchmark,
                        line=dict(color='orange', width=2, dash='dash')
                    ))


                    fig_rolling_sharpe.add_hline(y=0, line_dash="dash", line_color="red",
                                                 annotation_text="Zero Sharpe")

                    fig_rolling_sharpe.update_layout(
                        title=f'Rolling Sharpe Ratio ({coef_window_size} days)',
                        xaxis_title='Date',
                        yaxis_title='Sharpe ratio',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig_rolling_sharpe, use_container_width=True)

                    fig_rolling_sortino = go.Figure()

                    fig_rolling_sortino.add_trace(go.Scatter(
                        x=rolling_sortino.index,
                        y=rolling_sortino.values,
                        mode='lines',
                        name='Portfolio',
                        line=dict(color='blue', width=2)
                    ))

                    fig_rolling_sortino.add_trace(go.Scatter(
                        x=benchmark_rolling_sortino.index,
                        y=benchmark_rolling_sortino.values,
                        mode='lines',
                        name=benchmark,
                        line=dict(color='orange', width=2, dash='dash')
                    ))


                    fig_rolling_sortino.add_hline(y=0, line_dash="dash", line_color="red",
                                                  annotation_text="Zero Sortino")

                    fig_rolling_sortino.update_layout(
                        title=f'Rolling Sortino Ratio ({coef_window_size} days)',
                        xaxis_title='Date',
                        yaxis_title='Sortino ratio',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig_rolling_sortino, use_container_width=True)

        with rolling_tabs[2]:
            st.subheader("Rolling beta and alpha")

            beta_window_size = st.slider(
                "Window size (days) for beta and alpha",
                min_value=63,
                max_value=252,
                value=126,
                step=21,
                key="beta_window_size"
            )

            if isinstance(portfolio_returns.index, pd.DatetimeIndex) and benchmark_returns is not None:

                rolling_cov = portfolio_returns.rolling(window=beta_window_size).cov(benchmark_returns)
                rolling_var = benchmark_returns.rolling(window=beta_window_size).var()
                rolling_beta = rolling_cov / rolling_var

                rolling_portfolio_return = portfolio_returns.rolling(window=beta_window_size).mean() * 252
                rolling_benchmark_return = benchmark_returns.rolling(window=beta_window_size).mean() * 252
                rolling_alpha = rolling_portfolio_return - (
                            config.RISK_FREE_RATE + rolling_beta * (rolling_benchmark_return - config.RISK_FREE_RATE))

                fig_rolling_beta = go.Figure()

                fig_rolling_beta.add_trace(go.Scatter(
                    x=rolling_beta.index,
                    y=rolling_beta.values,
                    mode='lines',
                    name='Beta',
                    line=dict(color='purple', width=2)
                ))

                fig_rolling_beta.add_hline(y=1, line_dash="dash", line_color="grey",
                                           annotation_text="Beta = 1")
                fig_rolling_beta.add_hline(y=0, line_dash="dash", line_color="green",
                                           annotation_text="Beta = 0")

                fig_rolling_beta.update_layout(
                    title=f'Rolling beta ({beta_window_size} days)',
                    xaxis_title='Date',
                    yaxis_title='Beta',
                    hovermode='x unified'
                )

                st.plotly_chart(fig_rolling_beta, use_container_width=True)

                fig_rolling_alpha = go.Figure()

                fig_rolling_alpha.add_trace(go.Scatter(
                    x=rolling_alpha.index,
                    y=rolling_alpha.values * 100,
                    mode='lines',
                    name='Alpha',
                    line=dict(color='green', width=2)
                ))

                fig_rolling_alpha.add_hline(y=0, line_dash="dash", line_color="red",
                                            annotation_text="Alpha = 0")

                fig_rolling_alpha.update_layout(
                    title=f'Rolling alpha ({beta_window_size} days)',
                    xaxis_title='Date',
                    yaxis_title='Alpha (%)',
                    hovermode='x unified'
                )

                st.plotly_chart(fig_rolling_alpha, use_container_width=True)

        with rolling_tabs[3]:
            st.subheader("Separate analysis of bullish and bearish periods")

            if isinstance(portfolio_returns.index, pd.DatetimeIndex) and benchmark_returns is not None:

                common_index = portfolio_returns.index.intersection(benchmark_returns.index)

                portfolio_returns_aligned = portfolio_returns.loc[common_index]
                benchmark_returns_aligned = benchmark_returns.loc[common_index]

                bull_market = benchmark_returns_aligned > 0
                bear_market = benchmark_returns_aligned < 0

                bull_portfolio_return = portfolio_returns_aligned[bull_market].mean() * 252 * 100
                bear_portfolio_return = portfolio_returns_aligned[bear_market].mean() * 252 * 100

                bull_benchmark_return = benchmark_returns_aligned[bull_market].mean() * 252 * 100
                bear_benchmark_return = benchmark_returns_aligned[bear_market].mean() * 252 * 100

                if bull_market.sum() > 0:
                    bull_beta = portfolio_returns_aligned[bull_market].cov(benchmark_returns_aligned[bull_market]) / \
                                benchmark_returns_aligned[bull_market].var()
                else:
                    bull_beta = 0

                if bear_market.sum() > 0:
                    bear_beta = portfolio_returns_aligned[bear_market].cov(benchmark_returns_aligned[bear_market]) / \
                                benchmark_returns_aligned[bear_market].var()
                else:
                    bear_beta = 0

                market_conditions_df = pd.DataFrame({
                    'Metric': ['Portfolio Return (%)', 'Benchmark Return (%)', 'Beta', 'Difference (%)'],
                    'Bullish market': [
                        bull_portfolio_return,
                        bull_benchmark_return,
                        bull_beta,
                        bull_portfolio_return - bull_benchmark_return
                    ],
                    'Bearish market': [
                        bear_portfolio_return,
                        bear_benchmark_return,
                        bear_beta,
                        bear_portfolio_return - bear_benchmark_return
                    ]
                })

                st.dataframe(market_conditions_df.style.format({
                    'Bullish market': '{:.2f}',
                    'Bearish market': '{:.2f}'
                }), use_container_width=True)

                fig_market_conditions = go.Figure()

                fig_market_conditions.add_trace(go.Bar(
                    x=['Bullish market', 'Bearish market'],
                    y=[bull_portfolio_return, bear_portfolio_return],
                    name='Portfolio',
                    marker_color='blue'
                ))

                fig_market_conditions.add_trace(go.Bar(
                    x=['Bullish market', 'Bearish market'],
                    y=[bull_benchmark_return, bear_benchmark_return],
                    name=benchmark,
                    marker_color='orange'
                ))

                fig_market_conditions.update_layout(
                    title='Comparison of returns in different periods of the market',
                    xaxis_title='Market status',
                    yaxis_title='Annual Return (%)',
                    barmode='group',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig_market_conditions, use_container_width=True)

                # Rolling performance over time
                bull_window_size = st.slider(
                    "Window size (days) for period analysis",
                    min_value=63,
                    max_value=252,
                    value=126,
                    step=21,
                    key="bull_window_size"
                )

                #Calculate the rolling beta in bullish/bearish periods
                # For each window, determine whether the market is bullish or bearish
                rolling_bull_beta = []
                rolling_bear_beta = []
                dates = []

                for i in range(bull_window_size, len(benchmark_returns)):

                    window_start_idx = i - bull_window_size
                    window_end_idx = i

                    window_dates = benchmark_returns.index[window_start_idx:window_end_idx]
                    window_date = benchmark_returns.index[i]

                    window_benchmark = benchmark_returns.loc[window_dates]
                    window_portfolio = portfolio_returns.loc[window_dates]

                    common_dates = window_benchmark.index.intersection(window_portfolio.index)
                    window_benchmark = window_benchmark.loc[common_dates]
                    window_portfolio = window_portfolio.loc[common_dates]

                    window_bull = window_benchmark > 0
                    window_bear = window_benchmark < 0

                    if window_bull.sum() > 10:
                        bull_beta_val = window_portfolio[window_bull].cov(window_benchmark[window_bull]) / \
                                        window_benchmark[window_bull].var()
                        rolling_bull_beta.append(bull_beta_val)
                    else:
                        rolling_bull_beta.append(np.nan)

                    if window_bear.sum() > 10:
                        bear_beta_val = window_portfolio[window_bear].cov(window_benchmark[window_bear]) / \
                                        window_benchmark[window_bear].var()
                        rolling_bear_beta.append(bear_beta_val)
                    else:
                        rolling_bear_beta.append(np.nan)

                    dates.append(window_date)

                rolling_bull_beta_series = pd.Series(rolling_bull_beta, index=dates)
                rolling_bear_beta_series = pd.Series(rolling_bear_beta, index=dates)

                fig_bull_bear_beta = go.Figure()

                fig_bull_bear_beta.add_trace(go.Scatter(
                    x=rolling_bull_beta_series.index,
                    y=rolling_bull_beta_series.values,
                    mode='lines',
                    name='Beta in a bullish market',
                    line=dict(color='green', width=2)
                ))

                fig_bull_bear_beta.add_trace(go.Scatter(
                    x=rolling_bear_beta_series.index,
                    y=rolling_bear_beta_series.values,
                    mode='lines',
                    name='Beta in a bearish market',
                    line=dict(color='red', width=2)
                ))

                fig_bull_bear_beta.add_hline(y=1, line_dash="dash", line_color="grey",
                                             annotation_text="Beta = 1")

                fig_bull_bear_beta.update_layout(
                    title=f'Rolling beta in different market periods ({bull_window_size} days)',
                    xaxis_title='Date',
                    yaxis_title='Beta',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode='x unified'
                )

                st.plotly_chart(fig_bull_bear_beta, use_container_width=True)

    with tabs[7]:
        st.subheader("Advanced Analysis")

        advanced_tabs = st.tabs([
            "Yield Calendar",
            "Seasonal Analysis",
            "Distribution Quantiles",
            "Multiple Metrics"
        ])

        with advanced_tabs[0]:
            st.subheader("Monthly Return Calendar")

            if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

                monthly_df = pd.DataFrame({
                    'year': monthly_returns.index.year,
                    'month': monthly_returns.index.month,
                    'return': monthly_returns.values * 100
                })

                heatmap_data = monthly_df.pivot(index='year', columns='month', values='return')

                month_names = {
                    1: 'January', 2: 'February', 3: 'March', 4: 'Apr', 5: 'May', 6: 'June', 7: 'July', 8: 'Aug',
                    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                }
                heatmap_data = heatmap_data.rename(columns=month_names)

                # Visualize the heat map
                fig_heatmap = px.imshow(
                    heatmap_data,
                    labels=dict(x="Month", y="Year", color="Return (%)"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale='RdYlGn',
                    aspect="auto",
                    text_auto='.1f'
                )

                fig_heatmap.update_layout(
                    title='Monthly Return Calendar (%)'
                )

                st.plotly_chart(fig_heatmap, use_container_width=True)

                # Add a column with annual yield
                if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                    annual_returns = portfolio_returns.resample('A').apply(
                        lambda x: (1 + x).prod() - 1
                    ) * 100

                    annual_returns.index = annual_returns.index.year

                    years_in_heatmap = heatmap_data.index.tolist()
                    annual_returns = annual_returns[annual_returns.index.isin(years_in_heatmap)]

                    if not annual_returns.empty:

                        heatmap_data['Year'] = annual_returns.values

                        st.dataframe(heatmap_data.style.format('{:.2f}%').background_gradient(
                            cmap='RdYlGn', axis=None
                        ), use_container_width=True)

        with advanced_tabs[1]:
            st.subheader("Seasonal analysis")

            if isinstance(portfolio_returns.index, pd.DatetimeIndex):

                seasonal_returns = portfolio_returns.copy()
                seasonal_returns = pd.DataFrame(seasonal_returns)
                seasonal_returns.columns = ['returns']

                seasonal_returns['day_of_week'] = seasonal_returns.index.day_name()
                seasonal_returns['month'] = seasonal_returns.index.month_name()
                seasonal_returns['year'] = seasonal_returns.index.year
                seasonal_returns['quarter'] = seasonal_returns.index.quarter

                col1, col2 = st.columns(2)

                with col1:

                    day_of_week_returns = seasonal_returns.groupby('day_of_week')['returns'].mean() * 100

                    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                    day_of_week_returns = day_of_week_returns.reindex(days_order)

                    fig_days = px.bar(
                        x=day_of_week_returns.index,
                        y=day_of_week_returns.values,
                        title='Average return by day of the week (%)',
                        labels={'x': 'Day of the week', 'y': 'Average return (%)'},
                        color=day_of_week_returns.values,
                        color_continuous_scale='RdYlGn'
                    )

                    st.plotly_chart(fig_days, use_container_width=True)

                with col2:

                    month_returns = seasonal_returns.groupby('month')['returns'].mean() * 100

                    months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                                    'July', 'August', 'September', 'October', 'November', 'December']
                    month_returns = month_returns.reindex(months_order)

                    fig_months = px.bar(
                        x=month_returns.index,
                        y=month_returns.values,
                        title='Average return by month (%)',
                        labels={'x': 'Month', 'y': 'Average return (%)'},
                        color=month_returns.values,
                        color_continuous_scale='RdYlGn'
                    )

                    st.plotly_chart(fig_months, use_container_width=True)

                quarter_returns = seasonal_returns.groupby('quarter')['returns'].mean() * 100

                fig_quarters = px.bar(
                    x=quarter_returns.index,
                    y=quarter_returns.values,
                    title='Average return by quarter (%)',
                    labels={'x': 'Quarter', 'y': 'Average return (%)'},
                    color=quarter_returns.values,
                    color_continuous_scale='RdYlGn'
                )

                st.plotly_chart(fig_quarters, use_container_width=True)

            with advanced_tabs[2]:
                st.subheader("Quantiles of the Return Distribution")

                daily_quantiles = portfolio_returns.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]) * 100

                quantiles_df = pd.DataFrame({
                    'quantile': ['1%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '99%'],
                    'Daily return (%)': daily_quantiles.values.round(2)
                })

                st.dataframe(quantiles_df, use_container_width=True)

                fig_quantiles = go.Figure()

                fig_quantiles.add_trace(go.Box(
                    y=portfolio_returns * 100,
                    name='Portfolio',
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8,
                    marker=dict(color='blue')
                ))

                if benchmark_returns is not None:
                    fig_quantiles.add_trace(go.Box(
                        y=benchmark_returns * 100,
                        name=benchmark,
                        boxpoints='outliers',
                        jitter=0.3,
                        pointpos=-1.8,
                        marker=dict(color='orange')
                    ))

                fig_quantiles.update_layout(
                    title='Distribution of daily returns (%)',
                    yaxis_title='Return (%)',
                    boxmode='group',
                    showlegend=True
                )

                st.plotly_chart(fig_quantiles, use_container_width=True)

                st.subheader("Comparison with normal distribution")

                # Q-Q plot
                fig_qq = go.Figure()

                sorted_returns = sorted(portfolio_returns * 100)
                n = len(sorted_returns)

                from scipy import stats
                theoretical_quantiles = [stats.norm.ppf((i + 0.5) / n) for i in range(n)]
                theoretical_quantiles = np.array(
                    theoretical_quantiles) * portfolio_returns.std() * 100 + portfolio_returns.mean() * 100

                fig_qq.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_returns,
                    mode='markers',
                    name='Q-Q Plot',
                    marker=dict(color='blue', size=5)
                ))

                min_val = min(min(theoretical_quantiles), min(sorted_returns))
                max_val = max(max(theoretical_quantiles), max(sorted_returns))

                fig_qq.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect match',
                    line=dict(color='red', dash='dash')
                ))

                fig_qq.update_layout(
                    title='Q-Q Plot (Comparison with normal distribution)',
                    xaxis_title='Theoretical quantiles',
                    yaxis_title='Actual quantiles',
                    hovermode='closest'
                )

                st.plotly_chart(fig_qq, use_container_width=True)

                st.subheader("Statistical tests")

                try:

                    from scipy import stats as scipy_stats

                    shapiro_stat, shapiro_p = scipy_stats.shapiro(portfolio_returns)

                    jb_stat, jb_p = scipy_stats.jarque_bera(portfolio_returns)

                    skewness = portfolio_returns.skew()
                    kurtosis = portfolio_returns.kurtosis()

                    stats_results = pd.DataFrame({
                        'Test/Metric': ['Shapiro-Wilk test (p-value)', 'Jarque-Bera test (p-value)', 'Skewness',
                                        'Kurtosis'],
                        'Meaning': [shapiro_p, jb_p, skewness, kurtosis]
                    })

                    interpretations = []

                    if shapiro_p < 0.05:
                        interpretations.append("Shapiro-Wilk test: Distribution is not normal (p < 0.05)")
                    else:
                        interpretations.append(
                            "Shapiro-Wilk test: Failure to reject the hypothesis of normality (p >= 0.05)")

                    if jb_p < 0.05:
                        interpretations.append("Jarque-Bera test: Distribution is not normal (p < 0.05)")
                    else:
                        interpretations.append(
                            "Jarque-Bera test: It is impossible to reject the hypothesis of normality (p >= 0.05)")

                    if abs(skewness) > 0.5:
                        direction = "positive" if skewness > 0 else "negative"
                        interpretations.append(
                            f"The distribution has {direction} asymmetry (fat tail in the {'right' if skewness > 0 else 'left'} part)")
                    else:
                        interpretations.append("The distribution is approximately symmetrical.")

                    if kurtosis > 0.5:
                        interpretations.append("The distribution has heavy tails (leptokurtosis)")
                    elif kurtosis < -0.5:
                        interpretations.append("The distribution has light tails (platykurtosis)")
                    else:
                        interpretations.append("The kurtosis is close to the normal distribution")

                    st.dataframe(stats_results.style.format({
                        'Meaning': '{:.4f}'
                    }), use_container_width=True)

                    for interpretation in interpretations:
                        st.write(f"• {interpretation}")

                except Exception as e:
                    st.error(f"Statistical tests failed: {e}")

            with advanced_tabs[3]:
                st.subheader("Multiple performance metrics")

                metrics_data = []

                metrics_data.append({
                    'Metrics': 'Total Return (%)',
                    'Portfolio': portfolio_metrics.get('total_return', 0) * 100,
                    'Benchmark': portfolio_metrics.get('benchmark_return', 0) * 100,
                    'Difference': (portfolio_metrics.get('total_return', 0) - portfolio_metrics.get('benchmark_return',
                                                                                                 0)) * 100
                })

                metrics_data.append({
                    'Metrics': 'Annual Return (%)',
                    'Portfolio': portfolio_metrics.get('annualized_return', 0) * 100,
                    'Benchmark': portfolio_metrics.get('benchmark_annualized_return',
                                                      0) * 100 if 'benchmark_annualized_return' in portfolio_metrics else 0,
                    'Difference': (portfolio_metrics.get('annualized_return', 0) - portfolio_metrics.get(
                        'benchmark_annualized_return',
                        0)) * 100 if 'benchmark_annualized_return' in portfolio_metrics else 0
                })


                metrics_data.append({
                    'Metrics': 'Volatility (%)',
                    'Portfolio': portfolio_metrics.get('volatility', 0) * 100,
                    'Benchmark': portfolio_metrics.get('benchmark_volatility',
                                                      0) * 100 if 'benchmark_volatility' in portfolio_metrics else 0,
                    'Difference': (portfolio_metrics.get('benchmark_volatility', 0) - portfolio_metrics.get('volatility',
                                                                                                         0)) * 100 if 'benchmark_volatility' in portfolio_metrics else 0
                })

                metrics_data.append({
                    'Metrics': 'Maximum drawdown (%)',
                    'Portfolio': portfolio_metrics.get('max_drawdown', 0) * 100,
                    'Benchmark': portfolio_metrics.get('benchmark_max_drawdown',
                                                      0) * 100 if 'benchmark_max_drawdown' in portfolio_metrics else 0,
                    'Difference': (portfolio_metrics.get('benchmark_max_drawdown', 0) - portfolio_metrics.get(
                        'max_drawdown', 0)) * 100 if 'benchmark_max_drawdown' in portfolio_metrics else 0
                })

                metrics_data.append({
                    'Metrics': 'Sharpe ratio',
                    'Portfolio': portfolio_metrics.get('sharpe_ratio', 0),
                    'Benchmark': portfolio_metrics.get('benchmark_sharpe_ratio',
                                                      0) if 'benchmark_sharpe_ratio' in portfolio_metrics else 0,
                    'Difference': portfolio_metrics.get('sharpe_ratio', 0) - portfolio_metrics.get(
                        'benchmark_sharpe_ratio', 0) if 'benchmark_sharpe_ratio' in portfolio_metrics else 0
                })

                metrics_data.append({
                    'Metrics': 'Sortino ratio',
                    'Portfolio': portfolio_metrics.get('sortino_ratio', 0),
                    'Benchmark': portfolio_metrics.get('benchmark_sortino_ratio',
                                                      0) if 'benchmark_sortino_ratio' in portfolio_metrics else 0,
                    'Difference': portfolio_metrics.get('sortino_ratio', 0) - portfolio_metrics.get(
                        'benchmark_sortino_ratio', 0) if 'benchmark_sortino_ratio' in portfolio_metrics else 0
                })

                metrics_data.append({
                    'Metrics': 'Calmar Ratio',
                    'Portfolio': portfolio_metrics.get('calmar_ratio', 0),
                    'Benchmark': portfolio_metrics.get('benchmark_calmar_ratio',
                                                      0) if 'benchmark_calmar_ratio' in portfolio_metrics else 0,
                    'Difference': portfolio_metrics.get('calmar_ratio', 0) - portfolio_metrics.get(
                        'benchmark_calmar_ratio', 0) if 'benchmark_calmar_ratio' in portfolio_metrics else 0
                })

                metrics_data.append({
                    'Metrics': 'Beta',
                    'Portfolio': portfolio_metrics.get('beta', 0),
                    'Benchmark': 1.0,
                    'Difference': portfolio_metrics.get('beta', 0) - 1.0
                })

                metrics_data.append({
                    'Metrics': 'Alpha (%)',
                    'Portfolio': portfolio_metrics.get('alpha', 0) * 100,
                    'Benchmark': 0.0,
                    'Difference': portfolio_metrics.get('alpha', 0) * 100
                })

                metrics_data.append({
                    'Metrics': 'Information coefficient',
                    'Portfolio': portfolio_metrics.get('information_ratio', 0),
                    'Benchmark': 0.0,
                    'Difference': portfolio_metrics.get('information_ratio', 0)
                })

                metrics_data.append({
                    'Metrics': 'Win Rate (%)',
                    'Portfolio': portfolio_metrics.get('win_rate', 0) * 100,
                    'Benchmark': 0.0,
                    'Difference': 0.0
                })

                metrics_data.append({
                    'Metrics': 'Payout ratio',
                    'Portfolio': portfolio_metrics.get('payoff_ratio', 0),
                    'Benchmark': 0.0,
                    'Difference': 0.0
                })

                metrics_data.append({
                    'Metrics': 'Profit factor',
                    'Portfolio': portfolio_metrics.get('profit_factor', 0),
                    'Benchmark': 0.0,
                    'Difference': 0.0
                })


                metrics_df = pd.DataFrame(metrics_data)

                def color_diff(val):
                    if isinstance(val, float):
                        if val > 0:
                            return 'background-color: rgba(75, 192, 192, 0.2); color: green'
                        elif val < 0:
                            return 'background-color: rgba(255, 99, 132, 0.2); color: red'
                    return ''

                st.dataframe(metrics_df.style.format({
                    'Portfolio': '{:.2f}',
                    'Benchmark': '{:.2f}',
                    'Difference': '{:.2f}'
                }).applymap(color_diff, subset=['Difference']), use_container_width=True)

                st.subheader("Visual comparison with benchmark")

                metrics_to_plot = ['Total Return (%)', 'Annual Return (%)', 'Volatility (%)',
                                   'Maximum drawdown (%)', 'Sharpe ratio', 'Sortino ratio']

                plot_df = metrics_df[metrics_df['Metrics'].isin(metrics_to_plot)].copy()


                fig_metrics = go.Figure()

                fig_metrics.add_trace(go.Bar(
                    x=plot_df['Metrics'],
                    y=plot_df['Portfolio'],
                    name='Portfolio',
                    marker_color='blue'
                ))

                fig_metrics.add_trace(go.Bar(
                    x=plot_df['Metrics'],
                    y=plot_df['Benchmark'],
                    name='Benchmark',
                    marker_color='orange'
                ))

                fig_metrics.update_layout(
                    title='Comparison of key metrics with the benchmark',
                    xaxis_title='Metrics',
                    yaxis_title='Meaning',
                    barmode='group',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig_metrics, use_container_width=True)
