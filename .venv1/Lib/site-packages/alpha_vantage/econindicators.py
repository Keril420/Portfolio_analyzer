from .alphavantage import AlphaVantage as av


class EconIndicators(av):
    """This class implements all the economic indicator api calls
    """

    @av._output_format
    @av._call_api_on_func
    def get_real_gdp(self, interval='annual'):
        """ Returns the annual and quarterly Real GDP of the United States

        Keyword Arguments:
            interval:  supported values are 'quarterly', 'annual' (default 'annual')
        """
        _FUNCTION_KEY = "REAL_GDP"
        return _FUNCTION_KEY, 'data', None
    
    @av._output_format
    @av._call_api_on_func
    def get_real_gdp_per_capita(self):
        """ Returns the quarterly Real GDP per Capita data of the United States
        """
        _FUNCTION_KEY = "REAL_GDP_PER_CAPITA"
        return _FUNCTION_KEY, 'data', None
    
    @av._output_format
    @av._call_api_on_func
    def get_treasury_yield(self, interval='monthly', maturity='10year'):
        """ Returns the US treasury yield of a given maturity timeline

        Keyword Arguments:
            interval:  supported values are 'daily', 'weekly', 'monthly' (default 'monthly')
            maturity:  supported values are '3month', '2year', '5year', '7year', 
                '10year', '30year' (default '10year')
        """
        _FUNCTION_KEY = "TREASURY_YIELD"
        return _FUNCTION_KEY, 'data', None
    
    @av._output_format
    @av._call_api_on_func
    def get_ffr(self, interval='monthly'):
        """ Returns the federal funds rate (interest rate) of the United States

        Keyword Arguments:
            interval:  supported values are 'daily', 'weekly', 'monthly' (default 'monthly')
        """
        _FUNCTION_KEY = "FEDERAL_FUNDS_RATE"
        return _FUNCTION_KEY, 'data', None
    
    @av._output_format
    @av._call_api_on_func
    def get_cpi(self, interval='monthly'):
        """ Returns the consumer price index of the United States

        Keyword Arguments:
            interval:  supported values are 'semiannual', 'monthly' (default 'monthly')
        """
        _FUNCTION_KEY = "CPI"
        return _FUNCTION_KEY, 'data', None
    
    @av._output_format
    @av._call_api_on_func
    def get_inflation(self):
        """ Returns the annual inflation rates (consumer prices) of the United States
        """
        _FUNCTION_KEY = "INFLATION"
        return _FUNCTION_KEY, 'data', None
    
    @av._output_format
    @av._call_api_on_func
    def get_retail_sales(self):
        """ Returns the monthly Advance Retail Sales: Retail Trade data of the United States
        """
        _FUNCTION_KEY = "RETAIL_SALES"
        return _FUNCTION_KEY, 'data', None
    
    @av._output_format
    @av._call_api_on_func
    def get_durables(self):
        """ Returns the monthly manufacturers' new orders of durable goods in the United States
        """
        _FUNCTION_KEY = "DURABLES"
        return _FUNCTION_KEY, 'data', None
    
    @av._output_format
    @av._call_api_on_func
    def get_unemployment(self):
        """ Returns the monthly unemployment data of the United States
        """
        _FUNCTION_KEY = "UNEMPLOYMENT"
        return _FUNCTION_KEY, 'data', None
    
    @av._output_format
    @av._call_api_on_func
    def get_nonfarm(self):
        """ Returns the monthly US All Employees: Total Nonfarm
        """
        _FUNCTION_KEY = "NONFARM_PAYROLL"
        return _FUNCTION_KEY, 'data', None
