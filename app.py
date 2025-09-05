import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
st.markdown("<h1 style='text-align: center;'>Stock Trend Prediction Using LSTM</h1>", unsafe_allow_html=True)
st.markdown("---")

# Load trained model
model = load_model("stock_dl_model.h5")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Settings")
stock = st.sidebar.text_input("Enter Stock Symbol", "POWERGRID.NS")
start = st.sidebar.date_input("Start Date", dt.date(2000, 1, 1))
end = st.sidebar.date_input("End Date", dt.date(2024, 10, 1))

# -------------------------------
# Fetch Stock Data
# -------------------------------
df = yf.download(stock, start=start, end=end)

if not df.empty:
    # Tabs for navigation
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Data Overview", "EMA Charts", "Predictions", "Download"]
    )

    with tab1:
        st.subheader(f"Stock Data for {stock}")
        st.dataframe(df.tail(10))  # Show last 10 rows
        st.subheader("Descriptive Statistics")
        st.write(df.describe())

    with tab2:
        # EMA Calculations
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()

        # Plot 20 & 50 EMA
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.Close, 'y', label='Closing Price')
        ax1.plot(ema20, 'g', label='EMA 20')
        ax1.plot(ema50, 'r', label='EMA 50')
        ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
        ax1.legend()
        st.pyplot(fig1)

        # Plot 100 & 200 EMA
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.Close, 'y', label='Closing Price')
        ax2.plot(ema100, 'g', label='EMA 100')
        ax2.plot(ema200, 'r', label='EMA 200')
        ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
        ax2.legend()
        st.pyplot(fig2)

    with tab3:
        # Train/Test Split
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

        # Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Prepare test data
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Predictions
        y_predicted = model.predict(x_test)

        # Inverse scaling
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Plot Predictions
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, 'g', label="Original Price", linewidth=1)
        ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth=1)
        ax3.set_title("Prediction vs Original Trend")
        ax3.legend()
        st.pyplot(fig3)

    with tab4:
        csv = df.to_csv().encode("utf-8")
        st.download_button(
            "Download Dataset as CSV",
            csv,
            f"{stock}_dataset.csv",
            "text/csv",
            key="download-csv"
        )

else:
    st.warning("⚠️ No data found for this stock symbol. Please try another.")
