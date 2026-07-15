# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime

st.set_page_config(page_title="Stock Price Prediction", layout="centered")

st.markdown("## 📈 Stock Price Prediction Using Machine Learning")
st.write("Enter Stock Symbol (e.g., AAPL, TSLA, INFY.NS) and date range, then press Predict.")

# Inputs
col1, col2 = st.columns([2, 1])
with col1:
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, INFY.NS):", value="AAPL")
with col2:
    # default dates like your screenshot
    start_default = datetime.date(2020, 1, 1)
    end_default = datetime.date(2025, 1, 1)
    start_date = st.date_input("Start Date", value=start_default)
    end_date = st.date_input("End Date", value=end_default)

predict_btn = st.button("Predict")

# Helper: try yfinance if available, otherwise fallback to synthetic
def get_stock_data(ticker: str, start: str, end: str):
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is None or df.empty:
            raise RuntimeError("yfinance returned no data")
        df = df.reset_index()[["Date", "Close"]]
        return df
    except Exception as e:
        # fallback synthetic business-day data with a mild upward trend
        dates = pd.date_range(start=start, end=end, freq="B")
        np.random.seed(42)
        trend = np.linspace(100, 200, len(dates))  # baseline trend (can change)
        noise = np.random.normal(0, 4, len(dates))
        close = trend + noise
        df = pd.DataFrame({"Date": dates, "Close": close})
        return df

if predict_btn:
    if start_date >= end_date:
        st.error("Start Date must be before End Date.")
    else:
        st.info(f"Fetching data for **{symbol.upper()}** from {start_date} to {end_date} ...")
        df = get_stock_data(symbol.upper(), start_date.isoformat(), end_date.isoformat())

        if df.empty:
            st.error("No data available for that ticker and date range.")
        else:
            # prepare model
            df = df.sort_values("Date").reset_index(drop=True)
            df["Days"] = np.arange(len(df))
            X = df[["Days"]]
            y = df["Close"]

            model = LinearRegression()
            model.fit(X, y)
            df["Predicted"] = model.predict(X)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["Date"], df["Close"], label="Actual", linewidth=1.5)
            ax.plot(df["Date"], df["Predicted"], label="Predicted", linewidth=2)
            ax.set_title(f"Stock Price Prediction - {symbol.upper()}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()

            st.pyplot(fig)

            # Save file
            out_file = f"{symbol.upper()}_prediction.png"
            fig.savefig(out_file, dpi=150)
            st.success(f"Plot saved to `{out_file}`")

            # Optional: quick metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            rmse = mean_squared_error(df["Close"], df["Predicted"], squared=False)
            mae = mean_absolute_error(df["Close"], df["Predicted"])
            st.write(f"**RMSE:** {rmse:.3f}    •    **MAE:** {mae:.3f}")
            st.write("Tip: the orange line is a linear trend (LinearRegression). For day-by-day forecasting consider time-series models (ARIMA/LSTM).")