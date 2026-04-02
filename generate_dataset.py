import yfinance as yf
import pandas as pd

stocks = [
    "AAPL", "TSLA", "MSFT", "AMZN", "GOOGL",
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    "HDFCBANK.NS", "ICICIBANK.NS"
]

all_data = []

#FetchData
for ticker in stocks:
    print(f"Fetching {ticker}...")

    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)

        if df.empty:
            continue

        # Fix columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        #FEATURES
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(10).std()

        # Add stock name
        df['Stock'] = ticker

        all_data.append(df)

    except Exception as e:
        print(f"Error in {ticker}: {e}")

# COMBINE ALL DATA
final_df = pd.concat(all_data, ignore_index=True)

# Clean
final_df.dropna(inplace=True)


#SAVE CSV
final_df.to_csv("stock_dataset.csv", index=False)

print("✅ Dataset saved as stock_dataset.csv")