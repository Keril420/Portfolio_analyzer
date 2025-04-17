from .alphavantage import AlphaVantage as av

from datetime import datetime
search_date = datetime.now().date().strftime('%Y-%m-%d')

class FundamentalData(av):

    """This class implements all the api calls to fundamental data
    """
    def __init__(self, *args, **kwargs):
        """
        Inherit AlphaVantage base class with its default arguments.
        """
        super(FundamentalData, self).__init__(*args, **kwargs)
        self._append_type = False
        if self.output_format.lower() == 'csv':
            raise ValueError("Output format {} is not compatible with the FundamentalData class".format(
                self.output_format.lower()))

    @av._output_format
    @av._call_api_on_func
    def get_company_overview(self, symbol):
        """
        Returns the company information, financial ratios, 
        and other key metrics for the equity specified. 
        Data is generally refreshed on the same day a company reports its latest 
        earnings and financials.

        Keyword Arguments:
            symbol:  the symbol for the equity we want to get its data
        """
        _FUNCTION_KEY = 'OVERVIEW'
        return _FUNCTION_KEY, None, None
    
    @av._output_format
    @av._call_api_on_func
    def get_dividends(self, symbol):
        """
        Returns historical and future (declared) dividend distributions.

        Keyword Arguments:
            symbol:  the symbol for the equity we want to get its data
        """
        _FUNCTION_KEY = 'DIVIDENDS'
        return _FUNCTION_KEY, 'data', 'symbol'
    
    @av._output_format
    @av._call_api_on_func
    def get_splits(self, symbol):
        """
        Returns historical split events.

        Keyword Arguments:
            symbol:  the symbol for the equity we want to get its data
        """
        _FUNCTION_KEY = 'SPLITS'
        return _FUNCTION_KEY, 'data', 'symbol'
    
    @av._output_format
    @av._call_api_on_func
    def get_income_statement_annual(self, symbol):
        """
        Returns the annual and quarterly income statements for the company of interest. 
        Data is generally refreshed on the same day a company reports its latest 
        earnings and financials.

        Keyword Arguments:
            symbol:  the symbol for the equity we want to get its data
        """
        _FUNCTION_KEY = 'INCOME_STATEMENT'
        return _FUNCTION_KEY, 'annualReports', 'symbol'

    @av._output_format
    @av._call_api_on_func
    def get_income_statement_quarterly(self, symbol):
        """
        Returns the annual and quarterly income statements for the company of interest. 
        Data is generally refreshed on the same day a company reports its latest 
        earnings and financials.

        Keyword Arguments:
            symbol:  the symbol for the equity we want to get its data
        """
        _FUNCTION_KEY = 'INCOME_STATEMENT'
        return _FUNCTION_KEY, 'quarterlyReports', 'symbol'

    @av._output_format
    @av._call_api_on_func
    def get_balance_sheet_annual(self, symbol):
        """
        Returns the annual and quarterly balance sheets for the company of interest.
        Data is generally refreshed on the same day a company reports its latest 
        earnings and financials.

        Keyword Arguments:
            symbol:  the symbol for the equity we want to get its data
        """
        _FUNCTION_KEY = 'BALANCE_SHEET'
        return _FUNCTION_KEY, 'annualReports', 'symbol'

    @av._output_format
    @av._call_api_on_func
    def get_balance_sheet_quarterly(self, symbol):
        """
        Returns the annual and quarterly balance sheets for the company of interest.
        Data is generally refreshed on the same day a company reports its latest 
        earnings and financials.

        Keyword Arguments:
            symbol:  the symbol for the equity we want to get its data
        """
        _FUNCTION_KEY = 'BALANCE_SHEET'
        return _FUNCTION_KEY, 'quarterlyReports', 'symbol'

    @av._output_format
    @av._call_api_on_func
    def get_cash_flow_annual(self, symbol):
        """
        Returns the annual and quarterly cash flows for the company of interest.
        Data is generally refreshed on the same day a company reports its latest 
        earnings and financials.

        Keyword Arguments:
            symbol:  the symbol for the equity we want to get its data
        """
        _FUNCTION_KEY = 'CASH_FLOW'
        return _FUNCTION_KEY, 'annualReports', 'symbol'

    @av._output_format
    @av._call_api_on_func
    def get_cash_flow_quarterly(self, symbol):
        """
        Returns the annual and quarterly cash flows for the company of interest.
        Data is generally refreshed on the same day a company reports its latest 
        earnings and financials.

        Keyword Arguments:
            symbol:  the symbol for the equity we want to get its data
        """
        _FUNCTION_KEY = 'CASH_FLOW'
        return _FUNCTION_KEY, 'quarterlyReports', 'symbol'
    
    @av._output_format
    @av._call_api_on_func
    def get_earnings_annual(self, symbol):
        """
        Returns the annual and quarterly earnings (EPS) for the company of interest. 
        Quarterly data also includes analyst estimates and surprise metrics.

        Keyword Arguments:
            symbol:  the symbol for the equity we want to get its data
        """
        _FUNCTION_KEY = 'EARNINGS'
        return _FUNCTION_KEY, 'annualEarnings', 'symbol'
    
    @av._output_format
    @av._call_api_on_func
    def get_earnings_quarterly(self, symbol):
        """
        Returns the annual and quarterly earnings (EPS) for the company of interest. 
        Quarterly data also includes analyst estimates and surprise metrics.

        Keyword Arguments:
            symbol:  the symbol for the equity we want to get its data
        """
        _FUNCTION_KEY = 'EARNINGS'
        return _FUNCTION_KEY, 'quarterlyEarnings', 'symbol'
