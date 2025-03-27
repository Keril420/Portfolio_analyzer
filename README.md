# Investment Portfolio Management Application

An interactive web application for creating, managing, and optimizing investment portfolios.

## Features

- Create and manage multiple investment portfolios
- Import/export portfolio data from/to CSV
- Fetch real-time and historical market data
- Calculate key performance metrics
- Visualize portfolio allocation and performance
- Perform risk analysis and stress testing
- Optimize portfolio using various methods

## Project Structure

```
investment_portfolio_app/
├── src/                          # Main application code
│   ├── main.py                   # Streamlit application entry point
│   ├── config.py                 # Configuration parameters
│   ├── utils/                    # Utility functions
│   │   ├── data_loader.py        # Data acquisition from API, CSV, JSON
│   ├── models/                   # Machine learning models
│   ├── components/               # UI components
│   ├── pages/                    # Streamlit pages
├── data/                         # Data storage directory
│   ├── raw/                      # Raw CSV/JSON files
│   ├── processed/                # Processed data files
├── notebooks/                    # Jupyter Notebooks
├── tests/                        # Test files
├── requirements.txt              # Dependencies file
├── README.md                     # Project documentation
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```
streamlit run src/main.py
```

The application will be available at http://localhost:8501

## License

