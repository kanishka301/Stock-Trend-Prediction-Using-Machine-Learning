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
st.markdown("<h1 style='text-align: center;'>Stock Trend Prediction Using Stacked LSTM</h1>", unsafe_allow_html=True)
st.markdown("---")

# Load trained stacked LSTM model
model = load_model("stock_dl_model.h5")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Settings")
stock = st.sidebar.text_input("Enter Stock Symbol", "POWERGRID.NS")
start = st.sidebar.date_input("Start Date", dt.date(2020, 1, 1))
end = st.sidebar.date_input("End Date", dt.date(2025, 9, 1))

# -------------------------------
# Fetch Stock Data
# -------------------------------
df = yf.download(stock, start=start, end=end)

if not df.empty:
    # Tabs for navigation
    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "EMA Charts", "Predictions", "Download"])

    with tab1:
        st.subheader(f"Stock Data for {stock}")
        st.dataframe(df.tail(10))
        st.subheader("Descriptive Statistics")
        st.write(df.describe())

    with tab2:
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.Close, label='Closing Price', color='yellow')
        ax1.plot(ema100, label='EMA 100', color='green')
        ax1.plot(ema200, label='EMA 200', color='red')
        ax1.set_title("Closing Price vs Time (100 & 200 Days EMA)")
        ax1.legend()
        st.pyplot(fig1)

    with tab3:
        # Prepare data
        data = df[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        training_size = int(len(scaled_data) * 0.70)
        train_data = scaled_data[:training_size]
        test_data = scaled_data[training_size:]

        def create_dataset(dataset, time_step=100):
            x, y = [], []
            for i in range(time_step, len(dataset)):
                x.append(dataset[i-time_step:i, 0])
                y.append(dataset[i, 0])
            return np.array(x), np.array(y)

        x_train, y_train = create_dataset(train_data)
        x_test, y_test = create_dataset(test_data)

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        # Predict
        y_predicted = model.predict(x_test)
        y_predicted = scaler.inverse_transform(y_predicted.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(y_test, label="Original Price", color='green')
        ax2.plot(y_predicted, label="Predicted Price", color='red')
        ax2.set_title("Prediction vs Original Trend")
        ax2.legend()
        st.pyplot(fig2)

        # Forecast next 30 days
        x_input = test_data[-100:].reshape(1, -1)
        temp_input = list(x_input[0])
        lst_output = []

        for i in range(30):
            x_input = np.array(temp_input[-100:]).reshape(1, 100, 1)
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])


            # Combine historical and forecast data
        last_date = df.index[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)

        # Create a new DataFrame for plotting
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).flatten()
        })

        # Plot historical + forecast
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        ax4.plot(df.index, df['Close'], label="Historical Price", color='blue')
        ax4.axvline(x=last_date, color='gray', linestyle='--', label='Forecast Start')
        ax4.plot(forecast_df['Date'], forecast_df['Forecast'], label="30-Day Forecast", color='orange')
        ax4.set_title("Stock Price with 30-Day Forecast Extension")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Price")
        ax4.legend()
        st.pyplot(fig4)


        future_days = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(future_days, label="Next 30 Days Forecast", color='blue')
        ax3.set_title("Future Stock Price Forecast")
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
    st.warning("No data found for this stock symbol. Please try another.")
