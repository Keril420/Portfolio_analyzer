import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import src.config as config
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Using absolute imports
from src.utils.calculations import PortfolioAnalytics
from src.utils.risk_management import RiskManagement
from src.utils.visualization import PortfolioVisualization

def run(data_fetcher, portfolio_manager):
    """
    Function to display the portfolio creation page

    Args:
        data_fetcher: DataFetcher instance for loading data
        portfolio_manager: PortfolioDataManager instance for working with portfolios
    """
    st.title("Portfolio creation")

    # Create tabs for different ways to create a portfolio
    tabs = st.tabs([
        "Manual input",
        "Import from CSV",
        "Templates",
        "Portfolio management"
    ])

    with tabs[0]:
        create_portfolio_manually(data_fetcher, portfolio_manager)

    with tabs[1]:
        import_portfolio_from_csv(data_fetcher, portfolio_manager)

    with tabs[2]:
        create_portfolio_from_template(data_fetcher, portfolio_manager)

    with tabs[3]:
        manage_portfolios(data_fetcher, portfolio_manager)


def create_portfolio_manually(data_fetcher, portfolio_manager):
    """
    Function for manual portfolio creation

    Args:
        data_fetcher: DataFetcher instance for loading data
        portfolio_manager: PortfolioDataManager instance for working with portfolios
    """
    st.header("Manual portfolio creation")

    # Basic information about the portfolio
    portfolio_name = st.text_input("Portfolio name", key="manual_portfolio_name")
    portfolio_description = st.text_area("Description (optional)", key="manual_description")

    # Entering tickers and weights
    st.subheader("Adding assets")

    input_method = st.radio(
        "Method of entering assets",
        ["Text input", "Step-by-step adding", "Search for assets"]
    )

    if input_method == "Text input":
        st.write("Enter tickers and weights in the format:")
        st.code("AAPL:0.4, MSFT:0.3, GOOGL:0.3")
        st.write("or")
        st.code("AAPL 0.4, MSFT 0.3, GOOGL 0.3")
        st.write("or one per line:")
        st.code("""
        AAPL:0.4
        MSFT:0.3
        GOOGL:0.3
        """)

        tickers_text = st.text_area("List of tickers and scales", height=200, key="manual_tickers_text")

        if st.button("Check tickers") and tickers_text.strip():
            try:
                parsed_assets = portfolio_manager.parse_ticker_weights_text(tickers_text)

                if not parsed_assets:
                    st.error("No tickers were recognized. Please check your input format.")
                else:
                    st.success(f"Recognized {len(parsed_assets)} assets.")

                    # Checking the validity of tickers
                    tickers = [asset['ticker'] for asset in parsed_assets]
                    valid_tickers, invalid_tickers = data_fetcher.validate_tickers(tickers)

                    if invalid_tickers:
                        st.warning(f"The following tickers were not found.: {', '.join(invalid_tickers)}")

                    # Show a table with recognized assets
                    assets_df = pd.DataFrame({
                        'Ticker': [asset['ticker'] for asset in parsed_assets],
                        'Weight': [f"{asset['weight'] * 100:.2f}%" for asset in parsed_assets],
                        'Status': ['✅ Found' if asset['ticker'] in valid_tickers else '❌ Not found' for asset in
                                   parsed_assets]
                    })

                    st.dataframe(assets_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error parsing tickers: {e}")

        # and add an existence check
        if st.button("Create a portfolio") and portfolio_name and tickers_text.strip():
            try:
                # Check if a portfolio with this name exists
                existing_portfolios = portfolio_manager.list_portfolios()
                existing_names = [p['name'] for p in existing_portfolios]

                if portfolio_name in existing_names:
                    st.warning(
                        f"A portfolio with the name '{portfolio_name}' exists. Please select another name or go to the 'Portfolio Management' section to edit it.")
                else:
                    portfolio = portfolio_manager.create_portfolio_from_text(
                        tickers_text, portfolio_name, portfolio_description
                    )

                    # Save the portfolio
                    saved_path = portfolio_manager.save_portfolio(portfolio)

                    st.success(f"Portfolio '{portfolio_name}' successfully created with {len(portfolio['assets'])} assets!")

                    st.subheader("Structure of the created portfolio")

                    weights_data = {
                        'Ticker': [asset['ticker'] for asset in portfolio['assets']],
                        'Weight': [asset['weight'] for asset in portfolio['assets']]
                    }

                    fig = px.pie(
                        values=[asset['weight'] for asset in portfolio['assets']],
                        names=[asset['ticker'] for asset in portfolio['assets']],
                        title="Asset Allocation"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating portfolio: {e}")

    elif input_method == "Step by step addition":
        # Create or get a list of assets from the state
        if 'stepwise_assets' not in st.session_state:
            st.session_state.stepwise_assets = []

        # Asset Add Form
        with st.form("add_asset_form"):
            st.write("Adding a new asset")
            ticker = st.text_input("Ticker", key="stepwise_ticker").strip().upper()
            weight = st.slider("Weight (%)", 0, 100, 10, key="stepwise_weight") / 100

            submitted = st.form_submit_button("Add asset")

            if submitted and ticker:
                # Checking the validity of the ticker
                valid_tickers, _ = data_fetcher.validate_tickers([ticker])

                if not valid_tickers:
                    st.error(f"Ticker {ticker} not found. Please check the correctness of the ticker.")
                else:
                    # Add asset to the list
                    st.session_state.stepwise_assets.append({
                        'ticker': ticker,
                        'weight': weight
                    })
                    st.success(f"Asset {ticker} successfully added.")

        # Display the current list of assets
        if st.session_state.stepwise_assets:
            st.write("Current assets")

            assets_df = pd.DataFrame({
                'Ticker': [asset['ticker'] for asset in st.session_state.stepwise_assets],
                'Weight': [f"{asset['weight'] * 100:.2f}%" for asset in st.session_state.stepwise_assets]
            })

            st.dataframe(assets_df, use_container_width=True)

            # Visualization of the current distribution
            total_weight = sum(asset['weight'] for asset in st.session_state.stepwise_assets)

            if total_weight > 0:
                # Normalize weights for display
                normalized_weights = [asset['weight'] / total_weight for asset in st.session_state.stepwise_assets]

                fig = px.pie(
                    values=normalized_weights,
                    names=[asset['ticker'] for asset in st.session_state.stepwise_assets],
                    title="Asset Allocation"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Buttons for managing the list
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Clear list"):
                    st.session_state.stepwise_assets = []
                    st.success("The list of assets has been cleared..")

            with col2:
                if st.button("Normalize weights"):
                    total_weight = sum(asset['weight'] for asset in st.session_state.stepwise_assets)

                    if total_weight > 0:
                        for asset in st.session_state.stepwise_assets:
                            asset['weight'] = asset['weight'] / total_weight

                        st.success("Weights are normalized.")

            # Creating a portfolio from step-by-step added assets
            if st.button("Create a portfolio") and portfolio_name and st.session_state.stepwise_assets:
                try:
                    # Create a text view for the portfolio creation function
                    tickers_text = "\n".join(
                        [f"{asset['ticker']}:{asset['weight']}" for asset in st.session_state.stepwise_assets])

                    portfolio = portfolio_manager.create_portfolio_from_text(
                        tickers_text, portfolio_name, portfolio_description
                    )

                    st.success(f"Portfolio '{portfolio_name}' successfully created with {len(portfolio['assets'])} assets!")

                    st.session_state.stepwise_assets = []
                except Exception as e:
                    st.error(f"Error creating portfolio: {e}")

    elif input_method == "Search for assets":

        search_query = st.text_input("Search for assets (enter company name or ticker)", key="search_query")

        if search_query:
            with st.spinner('Search for assets...'):
                search_results = data_fetcher.search_tickers(search_query, limit=10)

                if not search_results:
                    st.info(f"Nothing found for '{search_query}'.")
                else:
                    st.success(f"Found {len(search_results)} assets.")

                    # Display search results
                    results_df = pd.DataFrame(search_results)

                    # Format the table
                    if 'symbol' in results_df.columns and 'name' in results_df.columns:
                        results_df = results_df[['symbol', 'name', 'type', 'region', 'currency']]
                        results_df.columns = ['Ticker', 'Name', 'Type', 'Region', 'Currency']

                    st.dataframe(results_df, use_container_width=True)

                    # Adding the selected asset
                    selected_ticker = st.selectbox("Select an asset to add",
                                                   [f"{result['symbol']} - {result['name']}" for result in
                                                    search_results])

                    # Extract the ticker from the selected row
                    if selected_ticker:
                        ticker = selected_ticker.split(" - ")[0]

                        # Weight for the selected asset
                        weight = st.slider("Weight (%)", 0, 100, 10, key="search_weight") / 100

                        if st.button("Add to portfolio"):
                            # Create or get a list of assets from the state
                            if 'search_assets' not in st.session_state:
                                st.session_state.search_assets = []

                            # Add asset to the list
                            st.session_state.search_assets.append({
                                'ticker': ticker,
                                'weight': weight
                            })

                            st.success(f"Asset {ticker} has been successfully added to the portfolio.")

        # Display the current list of assets from the search
        if 'search_assets' in st.session_state and st.session_state.search_assets:
            st.write("### Current assets")

            assets_df = pd.DataFrame({
                'Ticker': [asset['ticker'] for asset in st.session_state.search_assets],
                'Weight': [f"{asset['weight'] * 100:.2f}%" for asset in st.session_state.search_assets]
            })

            st.dataframe(assets_df, use_container_width=True)

            # Visualization of the current distribution
            total_weight = sum(asset['weight'] for asset in st.session_state.search_assets)

            if total_weight > 0:
                # Normalize weights for display
                normalized_weights = [asset['weight'] / total_weight for asset in st.session_state.search_assets]

                fig = px.pie(
                    values=normalized_weights,
                    names=[asset['ticker'] for asset in st.session_state.search_assets],
                    title="Asset Allocation"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Buttons for managing the list
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Clear list", key="clear_search_list"):
                    st.session_state.search_assets = []
                    st.success("The list of assets has been cleared..")

            with col2:
                if st.button("Normalize weights", key="normalize_search_weights"):

                    total_weight = sum(asset['weight'] for asset in st.session_state.search_assets)

                    if total_weight > 0:
                        for asset in st.session_state.search_assets:
                            asset['weight'] = asset['weight'] / total_weight

                        st.success("Weights are normalized.")

            # Creating a portfolio from found assets
            if st.button("Create a portfolio",
                         key="create_search_portfolio") and portfolio_name and st.session_state.search_assets:
                try:
                    # Create a text view for the portfolio creation function
                    tickers_text = "\n".join(
                        [f"{asset['ticker']}:{asset['weight']}" for asset in st.session_state.search_assets])

                    portfolio = portfolio_manager.create_portfolio_from_text(
                        tickers_text, portfolio_name, portfolio_description
                    )

                    st.success(f"Portfolio '{portfolio_name}' successfully created with {len(portfolio['assets'])} assets!")

                    # Clear the list to create a new portfolio
                    st.session_state.search_assets = []
                except Exception as e:
                    st.error(f"Error creating portfolio: {e}")


def import_portfolio_from_csv(data_fetcher, portfolio_manager):
    """
   Function for importing portfolio from CSV file

    Args:
        data_fetcher: DataFetcher instance to load data
        portfolio manager: Portfolio DataManager instance for working with portfolios
    """
    st.header("Import portfolio from CSV")

    # Basic information about the portfolio
    portfolio_name = st.text_input("Portfolio name", key="csv_portfolio_name")

    # CSV format instructions
    with st.expander("CSV file format"):
        st.write("""
        The CSV file must contain at least a 'ticker' column with asset tickers..

        Additional columns that can be included:
            - 'weight': asset weight in the portfolio (if not specified, equal weights will be used)
            - 'quantity': number of units of the asset
            - 'purchase_price': purchase price
            - 'purchase_date': purchase date in YYYY-MM-DD format
            - 'sector': sector
            - 'asset_class': asset class
            - 'region': region
            - 'currency': currency
            
            Example:
            ```
            ticker,weight,quantity,purchase_price
            AAPL,0.4,10,150.5
            MSFT,0.3,5,250.75
            GOOGL,0.3,2,2500.0
        ```
        """)

    # Upload CSV file
    uploaded_file = st.file_uploader("Select CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # CSV Preview
            df = pd.read_csv(uploaded_file)

            st.write("### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Checking if a required column exists
            if 'ticker' not in df.columns:
                st.error("The CSV file must contain a 'ticker' column with asset tickers.")
            else:
                if st.button("Import portfolio") and portfolio_name:
                    with st.spinner('Import portfolio...'):
                        temp_file = "./data/temp_upload.csv"
                        with open(temp_file, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        portfolio = portfolio_manager.import_from_csv(temp_file, portfolio_name)

                        portfolio_manager.save_portfolio(portfolio)

                        st.success(
                            f"Portfolio '{portfolio_name}' successfully imported with {len(portfolio['assets'])} assets!")

                        st.subheader("Structure of the imported portfolio")

                        fig = px.pie(
                            values=[asset['weight'] for asset in portfolio['assets']],
                            names=[asset['ticker'] for asset in portfolio['assets']],
                            title="Asset Allocation"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")


def create_portfolio_from_template(data_fetcher, portfolio_manager):
    """
    Function for creating a portfolio based on templates

    Args:
        data_fetcher: DataFetcher instance for loading data
        portfolio_manager: PortfolioDataManager instance for working with portfolios
    """
    st.header("Creating a portfolio from a template")

    # Basic information about the portfolio
    portfolio_name = st.text_input("Portfolio name", key="template_portfolio_name")
    portfolio_description = st.text_area("Description (optional)", key="template_description")

    # List of templates
    templates = {
        "S&P 500 Top 10": {
            "description": "The 10 Largest Companies in the S&P 500 Index",
            "assets": [
                {"ticker": "AAPL", "weight": 0.20, "name": "Apple Inc.", "sector": "Technology"},
                {"ticker": "MSFT", "weight": 0.18, "name": "Microsoft Corporation", "sector": "Technology"},
                {"ticker": "AMZN", "weight": 0.12, "name": "Amazon.com Inc.", "sector": "Consumer Discretionary"},
                {"ticker": "NVDA", "weight": 0.10, "name": "NVIDIA Corporation", "sector": "Technology"},
                {"ticker": "GOOGL", "weight": 0.09, "name": "Alphabet Inc. Class A",
                 "sector": "Communication Services"},
                {"ticker": "META", "weight": 0.08, "name": "Meta Platforms Inc.", "sector": "Communication Services"},
                {"ticker": "GOOG", "weight": 0.07, "name": "Alphabet Inc. Class C", "sector": "Communication Services"},
                {"ticker": "BRK.B", "weight": 0.06, "name": "Berkshire Hathaway Inc. Class B", "sector": "Financials"},
                {"ticker": "TSLA", "weight": 0.05, "name": "Tesla, Inc.", "sector": "Consumer Discretionary"},
                {"ticker": "UNH", "weight": 0.05, "name": "UnitedHealth Group Incorporated", "sector": "Healthcare"}
            ]
        },
        "Classic 60/40": {
            "description": "Classic allocation: 60% stocks, 40% bonds",
            "assets": [
                {"ticker": "VOO", "weight": 0.40, "name": "Vanguard S&P 500 ETF", "sector": "Equities",
                 "asset_class": "ETF"},
                {"ticker": "VEA", "weight": 0.20, "name": "Vanguard FTSE Developed Markets ETF", "sector": "Equities",
                 "asset_class": "ETF"},
                {"ticker": "BND", "weight": 0.30, "name": "Vanguard Total Bond Market ETF", "sector": "Fixed Income",
                 "asset_class": "ETF"},
                {"ticker": "BNDX", "weight": 0.10, "name": "Vanguard Total International Bond ETF",
                 "sector": "Fixed Income", "asset_class": "ETF"}
            ]
        },
        "Portfolio of constant weights": {
            "description": "Even distribution between asset classes",
            "assets": [
                {"ticker": "VTI", "weight": 0.25, "name": "Vanguard Total Stock Market ETF", "sector": "Equities",
                 "asset_class": "ETF"},
                {"ticker": "TLT", "weight": 0.25, "name": "iShares 20+ Year Treasury Bond ETF",
                 "sector": "Fixed Income", "asset_class": "ETF"},
                {"ticker": "GLD", "weight": 0.25, "name": "SPDR Gold Shares", "sector": "Commodities",
                 "asset_class": "ETF"},
                {"ticker": "IEF", "weight": 0.25, "name": "iShares 7-10 Year Treasury Bond ETF",
                 "sector": "Fixed Income", "asset_class": "ETF"}
            ]
        },
        "Dividend portfolio": {
            "description": "A portfolio focused on stable dividends",
            "assets": [
                {"ticker": "VYM", "weight": 0.20, "name": "Vanguard High Dividend Yield ETF", "sector": "Equities",
                 "asset_class": "ETF"},
                {"ticker": "SCHD", "weight": 0.20, "name": "Schwab US Dividend Equity ETF", "sector": "Equities",
                 "asset_class": "ETF"},
                {"ticker": "PG", "weight": 0.10, "name": "Procter & Gamble Co", "sector": "Consumer Staples"},
                {"ticker": "JNJ", "weight": 0.10, "name": "Johnson & Johnson", "sector": "Healthcare"},
                {"ticker": "KO", "weight": 0.10, "name": "Coca-Cola Co", "sector": "Consumer Staples"},
                {"ticker": "PEP", "weight": 0.10, "name": "PepsiCo, Inc.", "sector": "Consumer Staples"},
                {"ticker": "MCD", "weight": 0.10, "name": "McDonald's Corp", "sector": "Consumer Discretionary"},
                {"ticker": "MMM", "weight": 0.10, "name": "3M Co", "sector": "Industrials"}
            ]
        },
        "Technology portfolio": {
            "description": "Portfolio focused on the technology sector",
            "assets": [
                {"ticker": "AAPL", "weight": 0.15, "name": "Apple Inc.", "sector": "Technology"},
                {"ticker": "MSFT", "weight": 0.15, "name": "Microsoft Corporation", "sector": "Technology"},
                {"ticker": "AMZN", "weight": 0.10, "name": "Amazon.com Inc.", "sector": "Consumer Discretionary"},
                {"ticker": "NVDA", "weight": 0.10, "name": "NVIDIA Corporation", "sector": "Technology"},
                {"ticker": "GOOGL", "weight": 0.10, "name": "Alphabet Inc. Class A",
                 "sector": "Communication Services"},
                {"ticker": "META", "weight": 0.10, "name": "Meta Platforms Inc.", "sector": "Communication Services"},
                {"ticker": "TSLA", "weight": 0.10, "name": "Tesla, Inc.", "sector": "Consumer Discretionary"},
                {"ticker": "ADBE", "weight": 0.07, "name": "Adobe Inc.", "sector": "Technology"},
                {"ticker": "CRM", "weight": 0.07, "name": "Salesforce, Inc.", "sector": "Technology"},
                {"ticker": "PYPL", "weight": 0.06, "name": "PayPal Holdings, Inc.", "sector": "Financials"}
            ]
        }
    }

    # Select template
    selected_template = st.selectbox(
        "Select a portfolio template",
        list(templates.keys()),
        format_func=lambda x: f"{x} - {templates[x]['description']}",
        key="template_selection"
    )

    if selected_template:
        template = templates[selected_template]

        st.subheader(f"Sample: {selected_template}")
        st.write(template["description"])

        # Display the composition of the template
        template_df = pd.DataFrame({
            'Ticker': [asset['ticker'] for asset in template['assets']],
            'Name': [asset.get('name', '') for asset in template['assets']],
            'Weight': [f"{asset['weight'] * 100:.2f}%" for asset in template['assets']],
            'Sector': [asset.get('sector', 'N/A') for asset in template['assets']]
        })

        st.dataframe(template_df, use_container_width=True)

        # Template rendering
        fig = px.pie(
            values=[asset['weight'] for asset in template['assets']],
            names=[asset['ticker'] for asset in template['assets']],
            title="Asset Allocation in Template"
        )
        st.plotly_chart(fig, use_container_width=True)


        if st.button("Create a portfolio from a template") and portfolio_name:
            with st.spinner('Portfolio creation...'):
                try:
                    # Checking the validity of tickers
                    tickers = [asset['ticker'] for asset in template['assets']]
                    valid_tickers, invalid_tickers = data_fetcher.validate_tickers(tickers)

                    if invalid_tickers:
                        st.warning(
                            f"The following tickers were not found: {', '.join(invalid_tickers)}. They will be excluded from the portfolio.")

                    # Create a portfolio structure
                    assets = []
                    for asset in template['assets']:
                        if asset['ticker'] in valid_tickers:
                            assets.append(asset.copy())

                    portfolio_data = {
                        'name': portfolio_name,
                        'description': portfolio_description or template['description'],
                        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'assets': assets
                    }

                    # Normalize weights if invalid tickers were excluded
                    if invalid_tickers:
                        total_weight = sum(asset['weight'] for asset in portfolio_data['assets'])
                        for asset in portfolio_data['assets']:
                            asset['weight'] = asset['weight'] / total_weight

                    # Enriching portfolio data with additional information
                    portfolio_manager._enrich_portfolio_data(portfolio_data)

                    # Save the portfolio
                    portfolio_manager.save_portfolio(portfolio_data)

                    st.success(
                        f"Portfolio '{portfolio_name}' successfully created with {len(portfolio_data['assets'])} assets!")
                except Exception as e:
                    st.error(f"Error creating portfolio: {e}")


def manage_portfolios(data_fetcher, portfolio_manager):
    """
    Function for managing existing portfolios

    Args:
        data_fetcher: DataFetcher instance for loading data
        portfolio_manager: PortfolioDataManager instance for working with portfolios
    """
    st.header("Portfolio management")

    # Get a list of portfolios
    portfolios = portfolio_manager.list_portfolios()

    if not portfolios:
        st.info("No portfolios found. Please create a portfolio in one of the sections.")
        return

    # Display a list of portfolios
    st.subheader("List of portfolios")

    portfolios_df = pd.DataFrame({
        'Name': [p['name'] for p in portfolios],
        'Assets': [p['asset_count'] for p in portfolios],
        'Last update': [p['last_updated'] for p in portfolios]
    })

    st.dataframe(portfolios_df, use_container_width=True)

    selected_portfolio = st.selectbox(
        "Select a portfolio for action",
        [p['name'] for p in portfolios],
        key="manage_portfolio_selection"
    )

    if selected_portfolio:
        action = st.radio(
            "Select an action",
            ["View", "Export to CSV", "Duplication", "Delete"],
            key="portfolio_action"
        )

        if action == "View":
            # Loading portfolio data
            portfolio_data = portfolio_manager.load_portfolio(selected_portfolio)

            st.subheader(f"Portfolio: {portfolio_data['name']}")

            if 'description' in portfolio_data and portfolio_data['description']:
                st.write(portfolio_data['description'])

            # Display the list of assets
            assets_data = []
            for asset in portfolio_data['assets']:
                asset_row = {
                    'Ticker': asset['ticker'],
                    'Weight (%)': f"{asset['weight'] * 100:.2f}%"
                }

                # Adding accessible information
                for field in ['name', 'sector', 'industry', 'asset_class', 'currency']:
                    if field in asset:
                        asset_row[field.capitalize()] = asset[field]

                if 'current_price' in asset:
                    asset_row['Current price'] = asset['current_price']

                if 'price_change_pct' in asset:
                    asset_row['Price change (%)'] = f"{asset['price_change_pct']:.2f}%"

                assets_data.append(asset_row)

            st.dataframe(pd.DataFrame(assets_data), use_container_width=True)

            # Visualization of asset allocation
            fig = px.pie(
                values=[asset['weight'] for asset in portfolio_data['assets']],
                names=[asset['ticker'] for asset in portfolio_data['assets']],
                title="Asset Allocation"
            )
            st.plotly_chart(fig, use_container_width=True)

            # If there is a sector distribution, display it
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
                    title="Distribution by sectors"
                )
                st.plotly_chart(fig_sectors, use_container_width=True)

        elif action == "Export to CSV":
            # Loading portfolio data
            portfolio_data = portfolio_manager.load_portfolio(selected_portfolio)

            try:
                csv_path = portfolio_manager.export_to_csv(portfolio_data)

                st.success(f"Portfolio '{selected_portfolio}' successfully exported to CSV: {csv_path}")

                # Create a download button
                with open(csv_path, 'r') as f:
                    csv_content = f.read()

                st.download_button(
                    label="Download CSV file",
                    data=csv_content,
                    file_name=f"{selected_portfolio}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error exporting portfolio: {e}")

        elif action == "Duplication":
            new_name = st.text_input("Enter a name for the new portfolio",
                                     value=f"{selected_portfolio} (copy)",
                                     key="duplicate_name")

            if st.button("Duplicate portfolio") and new_name:
                try:
                    # Loading original portfolio data
                    portfolio_data = portfolio_manager.load_portfolio(selected_portfolio)

                    # Create a copy with a new name
                    portfolio_data['name'] = new_name
                    portfolio_data['created'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    portfolio_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    portfolio_manager.save_portfolio(portfolio_data)

                    st.success(f"Portfolio '{selected_portfolio}' successfully duplicated as '{new_name}'!")
                except Exception as e:
                    st.error(f"Error duplicating portfolio: {e}")

        elif action == "Delete":
            st.warning(f"You are about to delete portfolio '{selected_portfolio}'. This action cannot be undone.")

            if st.button("Confirm deletion"):
                try:
                    portfolio_file = next((p['filename'] for p in portfolios if p['name'] == selected_portfolio), None)

                    if portfolio_file:
                        portfolio_manager.delete_portfolio(portfolio_file)

                        st.success(f"Portfolio '{selected_portfolio}' has been successfully deleted!")
                    else:
                        st.error("Unable to find portfolio file.")
                except Exception as e:
                    st.error(f"Error deleting portfolio: {e}")