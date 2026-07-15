import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

st.title("📈 Stock Price Prediction Using Machine Learning")

# User input for stock symbol
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, INFY.NS):", "AAPL")

# Date range selection
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-01-01"))

if st.button("Predict"):
    # Fetch stock data
    data = yf.download(symbol, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found! Please check the stock symbol or date range.")
    else:
        data['Date'] = data.index
        data['Day'] = range(1, len(data) + 1)

        # Prepare data
        X = data[['Day']]
        y = data['Close']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        pred = model.predict(X_test)

        # Show results
        st.subheader(f"Predicted vs Actual Prices for {symbol}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data['Date'].iloc[-len(y_test):], y_test, label='Actual Price', color='blue')
        ax.plot(data['Date'].iloc[-len(y_test):], pred, label='Predicted Price', color='orange')
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price ($)")
        ax.legend()
        st.pyplot(fig)

        # Prediction for the next day
        next_day = [[len(data) + 1]]
        next_price = model.predict(next_day)[0]
        st.success(f"📅 Predicted next day price for {symbol}: *${next_price:.2f}*")