# Crypto currency SQLite database

Using Coinbase's REST API, a database of historical data can be created and kept
current for any supported coin (Ethereum, Bitcoin, Tether, etc.). This is
particularily useful for time series estimation models, since direct requests to
the API allow for up to 1-minute data intervals--a resolution which is much
higher than most compiled datasets available for download on the web which tend
to only provide daily data intervals.
