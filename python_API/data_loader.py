import yfinance as yf

def load_stock_data(ticker="AAPL", period="1y"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
    df.dropna(inplace=True)
    return df