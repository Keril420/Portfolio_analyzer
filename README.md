# Investment Portfolio Management System

An advanced interactive web application for creating, managing, analyzing, and optimizing investment portfolios with comprehensive analytics and risk management capabilities.

![Portfolio Management System](https://img.shields.io/badge/Portfolio-Management-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0%2B-red)

## Overview

This system provides a comprehensive solution for portfolio management, allowing investors to create, analyze, and optimize their investment portfolios using advanced financial models and analytical tools. The application combines modern portfolio theory with real-time data, interactive visualizations, and machine learning approaches to deliver actionable insights.

## Key Features

### Portfolio Creation and Management
- Create and save multiple investment portfolios
- Add assets through manual entry, CSV import, or text list input
- Access comprehensive asset information (company details, sectors, financial metrics)
- Portfolio templates and duplication capabilities
- Custom tagging and categorization

### Performance Analytics
- Calculate key financial metrics:
  - Returns (CAGR, absolute, annual, period-specific)
  - Volatility at different time intervals
  - Risk-adjusted metrics (Sharpe, Sortino, Calmar, Information Ratio)
  - Alpha/Beta relative to benchmarks
  - Maximum drawdown and recovery analysis
- Historical performance tracking with benchmark comparison
- Sector, regional, and asset class breakdowns
- Customizable time period analysis

### Risk Management
- Advanced risk measurement:
  - Value at Risk (VaR) using parametric, historical, and Monte Carlo methods
  - Conditional VaR (Expected Shortfall)
  - Stress testing under various market scenarios
  - Drawdown analysis and recovery projections
- Diversification assessment and correlation analysis
- Volatility and risk contribution breakdown
- Sector concentration risk analysis

### Portfolio Optimization
- Multiple optimization methodologies:
  - Markowitz Mean-Variance Optimization
  - Maximum Sharpe Ratio
  - Minimum Variance
  - Risk Parity
  - Equal Weight
- Efficient frontier visualization and optimal portfolio selection
- Custom constraints and objective functions
- Rebalancing recommendations

### Advanced Analysis
- Stress scenario modeling with historical analogies
- Monte Carlo simulation for future projections
- Scenario chaining for complex market events
- Rolling metrics for time-varying performance analysis
- Seasonal patterns and calendar-based analytics

### Interactive Visualizations
- Performance charts (cumulative returns, drawdowns, volatility)
- Risk heatmaps and correlation matrices
- Asset allocation breakdowns
- Historical scenario comparison
- Interactive risk-return scatter plots

## Project Structure

```
PythonProject/
├── src/                          # Main application code
│   ├── main.py                   # Streamlit application entry point
│   ├── config.py                 # Configuration parameters
│   ├── utils/                    # Utility functions
│   │   ├── calculations.py       # Portfolio analytics and calculations
│   │   ├── data_fetcher.py       # Data acquisition from APIs
│   │   ├── optimization.py       # Portfolio optimization algorithms
│   │   ├── risk_management.py    # Risk metrics and stress testing
│   │   ├── visualization.py      # Visualization utilities
│   │   ├── advanced_visualizations.py # Advanced visualization tools
│   │   ├── scenario_chaining.py  # Scenario modeling capabilities
│   │   ├── historical_context.py # Historical market data and analogies
│   ├── pages/                    # Streamlit pages
│   │   ├── portfolio_analysis.py # Portfolio analysis page
│   │   ├── portfolio_creation.py # Portfolio creation page
│   │   ├── portfolio_optimization.py # Portfolio optimization page
│   ├── data/                     # Data storage directory
│   │   ├── cache/                # Cached API data
│   │   ├── portfolios/           # Saved portfolio files
├── run_app.py                    # Entry point script
├── setup.py                      # Package installation script
├── requirements.txt              # Dependencies file
├── README.md                     # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/imnotkeril/Investment-Portfolio-Management-System.git
   cd Investment-Portfolio-Management-System
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Install as a development package:
   ```
   pip install -e .
   ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```
python run_app.py
```

Or directly with Streamlit:
```
streamlit run src/main.py
```

The application will be available at http://localhost:8501

## Getting Started Guide

### 1. Create Your First Portfolio

Start by navigating to the "Create Portfolio" section where you can:
- Enter tickers manually with automatic validation
- Import from a CSV file with predefined format
- Use a template portfolio as a starting point
- Paste a list of tickers with weights

The system will automatically fetch additional information about each asset including sector, market cap, and other fundamental data.

### 2. Analyze Your Portfolio

In the "Portfolio Analysis" section, you can:
- View key performance metrics including returns, volatility, and risk-adjusted measures
- Compare your portfolio against benchmarks like S&P 500, NASDAQ, or custom indices
- Analyze portfolio composition by sector, asset class, and other dimensions
- Explore detailed risk metrics including VaR, drawdowns, and stress test results
- Examine correlations between assets and identify diversification opportunities

The analysis section provides various tabs that allow you to dive deep into different aspects of performance, risk, correlation, and stress testing.

### 3. Optimize Your Portfolio

The "Portfolio Optimization" section offers tools to:
- Generate an efficient frontier to visualize risk-return tradeoffs
- Optimize for maximum Sharpe ratio, minimum variance, or specific return targets
- Apply risk parity approaches for balanced risk allocation
- Set constraints on sector exposure, individual asset weights, or other parameters
- Compare optimization results against your current portfolio

### 4. Explore Advanced Features

Additional specialized capabilities include:
- "Stress Scenario Chains" to model complex market events and their cascading effects
- "Historical Analogies" to compare current market conditions with past scenarios
- Rolling metrics analysis to see how performance evolves over time
- Seasonal patterns and calendar effects on portfolio performance

## Development

### Prerequisites
- Python 3.8 or higher
- Knowledge of pandas, numpy, and financial concepts
- API key for Alpha Vantage (optional for enhanced data access)

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is available for private use only.

## Acknowledgments

- Built with Streamlit, pandas, numpy, plotly, and other open-source libraries
- Financial data provided by Yahoo Finance and Alpha Vantage
- Portfolio optimization algorithms based on modern portfolio theory
- Risk management techniques from financial industry practices

---

For questions or support, please open an issue on the GitHub repository.