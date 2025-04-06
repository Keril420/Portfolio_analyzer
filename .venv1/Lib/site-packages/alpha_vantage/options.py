from .alphavantage import AlphaVantage as av


class Options(av):

    """This class implements all the api calls to US options data
    """
    @av._output_format
    @av._call_api_on_func
    def get_realtime_options(self, symbol, contract=None):
        """ Return realtime US options data.
        It raises ValueError when problems arise

        Keyword Arguments:
            symbol:  the symbol for the equity we want to get its data
            contract:  US options contract ID.
                By default, not set and entire option chain is returned
        """
        _FUNCTION_KEY = "REALTIME_OPTIONS"
        return _FUNCTION_KEY, 'data', None
    
    @av._output_format
    @av._call_api_on_func
    def get_historical_options(self, symbol, date=None):
        """ Return historical US options data.
        It raises ValueError when problems arise

        Keyword Arguments:
            symbol:  the symbol for the equity we want to get its data
            date:  By default, not set and data for the previous trading session is returned. 
                Any date later than 2008-01-01 is accepted. 
        """
        _FUNCTION_KEY = "HISTORICAL_OPTIONS"
        return _FUNCTION_KEY, 'data', None