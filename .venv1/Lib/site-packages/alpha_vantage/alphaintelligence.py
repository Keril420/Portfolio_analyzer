from .alphavantage import AlphaVantage as av


class AlphaIntelligence(av):

    """This class implements all the api calls to advanced market intelligence
    """
    @av._output_format
    @av._call_api_on_func
    def get_news_sentiment(self, tickers=None, topics=None, time_from=None, time_to=None,
                           sort='LATEST', limit=50):
        """ Return live and historical market news & sentiment data 
        from news outlets around the world.
        It raises ValueError when problems arise

        Keyword Arguments:
            tickers:  the stock/crypto/forex symbols of your choice
            topics:  news topics of your choice
            time_from and time_to:  time range of the news articles you are targeting, 
                in YYYYMMDDTHHMM format. If time_from is specified but time_to is missing, 
                returns articles published between the time_from value and the current time
            sort:  sort articles returned by API
                supported values are 'LATEST', 'EARLIEST', 'RELEVANCE' (default 'LATEST')
            limit:  number of output results
                supported values are 50, 1000 (default 50)
        """
        _FUNCTION_KEY = "NEWS_SENTIMENT"
        return _FUNCTION_KEY, 'feed', None
    
    @av._output_format
    @av._call_api_on_func
    def get_top_gainers(self):
        """ Returns the top 20 gainers in the US market.
        It raises ValueError when problems arise.
        """
        _FUNCTION_KEY = "TOP_GAINERS_LOSERS"
        return _FUNCTION_KEY, 'top_gainers', None
    
    @av._output_format
    @av._call_api_on_func
    def get_top_losers(self):
        """ Returns the top 20 losers in the US market.
        It raises ValueError when problems arise.
        """
        _FUNCTION_KEY = "TOP_GAINERS_LOSERS"
        return _FUNCTION_KEY, 'top_losers', None
    
    @av._output_format
    @av._call_api_on_func
    def get_most_active(self):
        """ Returns the top 20  most actively traded tickers in the US market.
        It raises ValueError when problems arise.
        """
        _FUNCTION_KEY = "TOP_GAINERS_LOSERS"
        return _FUNCTION_KEY, 'most_actively_traded', None