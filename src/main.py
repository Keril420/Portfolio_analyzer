import streamlit as st
from datetime import datetime
import pandas as pd
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.utils.data_fetcher import DataFetcher, PortfolioDataManager
from src.utils.calculations import PortfolioAnalytics
from src.utils.risk_management import RiskManagement
from src.utils.visualization import PortfolioVisualization
import src.config as config
from src.pages import portfolio_creation, portfolio_analysis, portfolio_optimization
from src.utils.scenario_chaining import scenario_chaining_page
from src.utils.advanced_visualizations import create_stress_impact_heatmap, create_interactive_stress_impact_chart, create_risk_tree_visualization
from src.utils.historical_context import display_historical_context, historical_analogy_page

# Page setup
st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Главная"


def main():
    # Initializing data
    data_fetcher = DataFetcher(cache_dir=str(config.CACHE_DIR), cache_expiry_days=config.CACHE_EXPIRY_DAYS)
    portfolio_manager = PortfolioDataManager(data_fetcher, storage_dir=str(config.PORTFOLIO_DIR))

    # Set API keys from configuration
    if config.ALPHA_VANTAGE_API_KEY:
        data_fetcher.api_keys['alpha_vantage'] = config.ALPHA_VANTAGE_API_KEY

    # Side navigation bar
    st.sidebar.title("Navigation")

    # List of pages
    pages = [
        "Home",
        "Portfolio creation",
        "Portfolio analysis",
        "Portfolio optimization",
        "Chains of stressful events",
        "Historical analogies"
    ]

    # Page selection
    selected_page = st.sidebar.radio("Select section:", pages)

    # Refresh the current page in session state
    st.session_state.current_page = selected_page

    # Display the selected page
    if selected_page == "Home":
        show_home_page(data_fetcher, portfolio_manager)
    elif selected_page == "Portfolio creation":
        portfolio_creation.run(data_fetcher, portfolio_manager)
    elif selected_page == "Portfolio analysis":
        portfolio_analysis.run(data_fetcher, portfolio_manager)
    elif selected_page == "Portfolio optimization":
        portfolio_optimization.run(data_fetcher, portfolio_manager)
    elif selected_page == "Chains of stressful events":
        scenario_chaining_page()
    elif selected_page == "Historical analogies":
        historical_analogy_page()

    # Additional information in the sidebar
    with st.sidebar.expander("About the program"):
        st.write("""
        The investment portfolio management system allows you to create,
        analyze and optimize investment portfolios
        using various strategies and models.


        Author: Wild Market Capital (@imnotkeril)
        Version: 1.0.0
        """)

    # API Status
    with st.sidebar.expander("API Status"):
        # Checking the API key
        if not config.ALPHA_VANTAGE_API_KEY:
            st.warning("Alpha Vantage API: Not configured")
        else:
            st.success("Alpha Vantage API: Connected")

        # Displaying API call counters
        st.write("API Call Counters:")
        st.write(f"- yFinance: {data_fetcher.api_call_counts['yfinance']}")
        st.write(f"- Alpha Vantage: {data_fetcher.api_call_counts['alpha_vantage']}")

        # Clear cache button
        if st.button("Clear data cache"):
            data_fetcher.clear_cache()
            st.success("Data cache cleared!")


def add_api_key_section(data_fetcher):
    """
    Add a section for API key management on the home page

    Args:
        data_fetcher: DataFetcher instance for API key management
    """
    st.subheader("🔑 API Key Configuration")

    # Input for API key
    api_key = st.text_input(
        "Enter your Alpha Vantage API Key",
        value=os.environ.get('ALPHA_VANTAGE_API_KEY', ''),
        type="password",
        help="Your API key will be used for enhanced data fetching"
    )

    # Buttons for API key management
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save API Key"):
            if api_key:
                # Save to environment variable
                os.environ['ALPHA_VANTAGE_API_KEY'] = api_key

                # Update DataFetcher with new API key
                data_fetcher.api_keys['alpha_vantage'] = api_key

                # Update config
                import src.config as config
                config.ALPHA_VANTAGE_API_KEY = api_key

                st.success("API Key saved successfully!")
            else:
                st.warning("Please enter a valid API key.")

    with col2:
        if st.button("Clear API Key"):
            # Remove from environment variable
            if 'ALPHA_VANTAGE_API_KEY' in os.environ:
                del os.environ['ALPHA_VANTAGE_API_KEY']

            # Clear from DataFetcher
            data_fetcher.api_keys['alpha_vantage'] = ''

            # Update config
            import src.config as config
            config.ALPHA_VANTAGE_API_KEY = ''

            st.info("API Key cleared.")

    # API Key Status
    st.markdown("### Current API Key Status")
    if data_fetcher.api_keys.get('alpha_vantage'):
        st.success("✅ API Key is configured and active")
    else:
        st.warning("❌ No API Key configured")

    # Explanation about API keys
    st.markdown("""
    To enhance the application's data fetching capabilities, you can provide an API key from Alpha Vantage.

    ### How to Get an API Key:
    1. Visit [Alpha Vantage Website](https://www.alphavantage.co/)
    2. Click on "Get Your Free API Key Today"
    3. Fill out the registration form
    4. Copy the API key provided after registration

    ### Benefits of Adding an API Key:
    - Access to more detailed financial data
    - Increased data fetch limits
    - More comprehensive market information
    - Enhanced search and lookup capabilities
    """)






def show_home_page(data_fetcher, portfolio_manager):
    """
    Function for displaying the main page

    Args:
        data_fetcher: DataFetcher instance for loading data
        portfolio_manager: Portfolio DataManager instance for working with portfolios
    """
    # Application Title
    st.title("Investment Portfolio Management System")

    # Welcome text
    st.write("""
    Welcome to our advanced portfolio management system! This application helps investors create, analyze, optimize,
    and monitor investment portfolios using sophisticated financial models and interactive visualizations.
    """)

    # Split the screen into two columns
    col1, col2 = st.columns(2)

    with col1:
        # Brief overview of functions
        st.subheader("Key Capabilities")

        st.markdown("""
        #### 📈 Portfolio Management
        - Create and track multiple investment portfolios
        - Import/export portfolio data from/to CSV and Excel
        - Real-time data fetching for market information
        - Customizable views for different analysis needs

        #### 📊 Advanced Analytics
        - Comprehensive performance metrics (returns, volatility, drawdowns)
        - Risk-adjusted measurements (Sharpe, Sortino, Calmar)
        - Benchmark comparison against major indices
        - Calendar-based analysis and seasonal patterns

        #### 🔍 Risk Assessment
        - Multi-dimensional risk analysis (VaR, CVaR, drawdowns)
        - Stress testing against historical and hypothetical scenarios
        - Correlation analysis to identify portfolio vulnerabilities
        - Risk contribution breakdown by asset and sector

        #### 🧮 Portfolio Optimization
        - Multiple optimization methodologies (Markowitz, Risk Parity)
        - Efficient frontier visualization with interactive selection
        - Custom constraint implementation for real-world limitations
        - Tactical asset allocation recommendations
        """)

    with col2:
        # Get a list of portfolios
        portfolios = portfolio_manager.list_portfolios()

        st.subheader("Your Portfolios")

        if portfolios:
            # Create a table with portfolios
            portfolios_df = pd.DataFrame({
                'Name': [p['name'] for p in portfolios],
                'Assets': [p['asset_count'] for p in portfolios],
                'Last update': [p['last_updated'] for p in portfolios]
            })

            st.dataframe(portfolios_df, use_container_width=True)

            # Quick Action Buttons
            st.subheader("Quick Actions")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Create New Portfolio"):
                    st.session_state.current_page = "Portfolio creation"
                    st.rerun()

            with col2:
                if st.button("Analyze Existing Portfolio"):
                    st.session_state.current_page = "Portfolio analysis"
                    st.rerun()
        else:
            st.info("You don't have any portfolios yet. Start by creating your first portfolio.")

            if st.button("Create First Portfolio"):
                st.session_state.current_page = "Portfolio creation"
                st.rerun()

        st.markdown("---")  # Separator
        add_api_key_section(data_fetcher)



    # Information section
    st.subheader("Getting Started")

    with st.expander("Guide for Beginners"):
        st.write("""
        ### Step-by-Step Guide

        1. **Create or Import a Portfolio**: Navigate to the "Create Portfolio" section to:
           - Add investments manually with ticker search
           - Import from CSV with your existing holdings
           - Use templates for common investment strategies
           - Specify weights or dollar amounts for each position

        2. **Analyze Your Portfolio**: In the "Portfolio Analysis" section, you can:
           - Review key performance metrics and risk measures
           - Compare against benchmarks and historical periods
           - Explore asset correlations and diversification metrics
           - Conduct stress tests and scenario analysis

        3. **Optimize Your Portfolio**: The "Portfolio Optimization" section allows you to:
           - Visualize the efficient frontier for your asset universe
           - Find the optimal portfolio based on your risk preference
           - Apply different optimization methodologies
           - Set constraints on asset allocations

        4. **Explore Advanced Features**: Additional specialized tools include:
           - Stress scenario chains for modeling complex market events
           - Historical analogies for market condition comparison
           - Rolling metrics to observe changing performance characteristics
           - Monte Carlo simulations for future projections
        """)

    # Practical Applications Section
    st.subheader("Practical Applications")

    st.write("""
    - **Long-term investors**: Create and monitor diversified portfolios aligned with your investment goals
    - **Active traders**: Analyze risk exposures and optimize position sizing
    - **Financial advisors**: Demonstrate portfolio characteristics and potential improvements to clients
    - **Students and researchers**: Explore financial theories with real market data

    This system combines modern portfolio theory, quantitative risk management, and interactive data visualization
    to provide powerful insights for investment decision-making.
    """)





if __name__ == "__main__":
    main()
