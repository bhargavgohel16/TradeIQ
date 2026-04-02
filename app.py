import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import warnings
import time

warnings.filterwarnings("ignore")
nltk.download('vader_lexicon')

#STOCK_OPTIONS

stock_options = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Google (GOOGL)": "GOOGL",
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "TCS (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS"
}

#STOCK_DATA

def get_stock_data(ticker):
    for i in range(3):  # retry 3 times
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False, threads=False)
    
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
    
            df = df.dropna()
            return df
        except:
            time.sleep(2)
            return pd.DataFrame()

#Yahoo + Google for the NEWS

def get_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news

        if news:
            return news

        # Google fallback
        query = ticker.replace(".NS", "")
        url = f"https://news.google.com/rss/search?q={query}+stock"

        response = requests.get(url)

        if response.status_code == 200:
            root = ET.fromstring(response.content)

            articles = []
            for item in root.findall(".//item")[:5]:
                articles.append({
                    "title": item.find("title").text,
                    "link": item.find("link").text
                })
            return articles

        return []

    except:
        return []

# SENTIMENT (FIXED)
def get_sentiment(news):
    if not news:
        return 0

    sia = SentimentIntensityAnalyzer()
    scores = []

    for article in news[:10]:
        if 'content' in article:
            title = article['content'].get('title', '')
        else:
            title = article.get('title', '')

        score = sia.polarity_scores(title)["compound"]
        scores.append(score)

    return sum(scores) / len(scores)

#FEATURES

def create_features(df):
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(10).std()

    df.dropna(inplace=True)
    return df

# TARGET
def create_target(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df


#MODEL
def train_model(df):
    X = df[['MA10', 'MA50', 'Returns', 'Volatility']]
    y = df['Target']

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    return model


# PREDICTION
def predict(model, df, sentiment):
    latest_df = df[['MA10', 'MA50', 'Returns', 'Volatility']].iloc[[-1]]

    pred = model.predict(latest_df)[0]

    if sentiment > 0.05:
        pred = 1
    elif sentiment < -0.05:
        pred = 0

    return "BUY 📈" if pred == 1 else "SELL 📉"

#REASONING
def generate_reason(df, sentiment, decision, news):
    latest = df.iloc[-1]

    ma10 = float(latest['MA10'])
    ma50 = float(latest['MA50'])
    volatility = float(latest['Volatility'])

    reasons = []

    if ma10 > ma50:
        reasons.append("📈 Uptrend (MA10 > MA50)")
    else:
        reasons.append("📉 Downtrend (MA10 < MA50)")

    if sentiment > 0.05:
        reasons.append("📰 Positive sentiment")
    elif sentiment < -0.05:
        reasons.append("📰 Negative sentiment")
    else:
        reasons.append("📰 Neutral sentiment")

    if volatility > 0.02:
        reasons.append("⚠️ High volatility")
    else:
        reasons.append("✅ Stable movement")

    if decision.startswith("BUY"):
        reasons.append("👉 Suggest BUY based on indicators")
    else:
        reasons.append("👉 Suggest SELL based on indicators")

    # Add headline
    if news:
        if 'content' in news[0]:
            title = news[0]['content'].get('title', '')
        else:
            title = news[0].get('title', '')

        reasons.append(f"🗞️ Key News: {title}")

    return reasons

#UI(Streamlit)
st.set_page_config(page_title="TradeIQ", layout="wide")

st.title("📊 TradeIQ - Smart Stock Predictor")

selected_stock = st.selectbox("🔍 Select Stock", list(stock_options.keys()))
ticker = stock_options[selected_stock]

st.write(f"📌 Selected: **{selected_stock}**")

if st.button("🔍 Predict"):

    df = get_stock_data(ticker)

    if df.empty:
        st.error("No stock data found")
    else:
        df = create_features(df)
        df = create_target(df)

        model = train_model(df)

        news = get_news(ticker)
        sentiment = get_sentiment(news)

        decision = predict(model, df, sentiment)
        reasons = generate_reason(df, sentiment, decision, news)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Prediction")
            st.success(decision)
            st.write(f"Sentiment Score: {sentiment:.2f}")

        with col2:
            st.subheader("📈 Chart")
            if 'Close' in df.columns:
                st.line_chart(df['Close'])

        # Reasons
        st.subheader("🧠 Why this recommendation?")
        for r in reasons:
            st.write(f"- {r}")

        # News display (FIXED)
        st.subheader("📰 Latest News")

        if news:
            for article in news[:5]:

                if 'content' in article:
                    title = article['content'].get('title', 'No Title')
                    link = article['content'].get('canonicalUrl', {}).get('url', '')
                else:
                    title = article.get('title', 'No Title')
                    link = article.get('link', '')

                st.write(f"**{title}**")
                st.write(link)
                st.write("---")
        else:
            st.warning("No news available")
