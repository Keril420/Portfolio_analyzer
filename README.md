# Investment Portfolio Management Application

An interactive web application for creating, managing, and optimizing investment portfolios.

## Features

- Create and manage multiple investment portfolios
- Import/export portfolio data from/to CSV
- Fetch real-time and historical market data
- Calculate key performance metrics
- Visualize portfolio allocation and performance
- Perform risk analysis and stress testing
- Optimize portfolio using various methods:
  - Markowitz Mean-Variance Optimization
  - Maximum Sharpe Ratio
  - Minimum Variance
  - Risk Parity
  - Equal Weight
- Tactical asset allocation strategies
- Monte Carlo simulation for future portfolio projection

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
   git clone https://github.com/Keril420/Portfolio_analyzer.git
   cd Portfolio_analyzer
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

4. Install additional dependencies:
   ```
   pip install pandas-datareader beautifulsoup4
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

## Features in Detail

### Portfolio Creation
- Manual input of tickers and weights
- Import from CSV files
- Selection from predefined templates
- Search for assets by company name or ticker

### Portfolio Analysis
- Key performance metrics calculation (Sharpe, Sortino, Beta, Alpha, etc.)
- Returns analysis
- Risk assessment
- Asset correlation analysis
- Stress testing

### Portfolio Optimization
- Optimize existing portfolios
- Create optimized new portfolios
- Tactical asset allocation
- Monte Carlo simulation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License
```
>_<
```