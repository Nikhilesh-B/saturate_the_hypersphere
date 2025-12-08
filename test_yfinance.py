
import yfinance as yf

def test_ticker(symbol):
    print(f"Testing {symbol}...")
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1mo")
        if hist.empty:
            print(f"Empty history for {symbol}")
        else:
            print(f"Success for {symbol}: {len(hist)} rows")
            print(hist.head(2))
    except Exception as e:
        print(f"Error for {symbol}: {e}")

test_ticker("AAPL")
test_ticker("MSFT")
test_ticker("UAL")

