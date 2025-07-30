
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pickle


# Load the dataset
data = pd.read_csv('C:/Users/kayas/model deployment/crude-oil-price-ak.csv')
data = data.rename(columns={'date': 'ds', 'price': 'y'})

# Normalize the data
scaler = MinMaxScaler()
data['y'] = scaler.fit_transform(data['y'].values.reshape(-1, 1))

# Prepare the data for LSTM model
window_size = 30  # Number of past days to consider for prediction
X, y = [], []
for i in range(len(data) - window_size):
    X.append(data['y'].values[i:i+window_size])
    y.append(data['y'].values[i+window_size])
X = np.array(X)
y = np.array(y)

# Create the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(64, activation='relu', input_shape=(window_size, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')

# Train the LSTM model
model_lstm.fit(X, y, epochs=10, batch_size=16)

# Save the model
filename = 'Forecasting_LSTM_ak.pkl'
pickle.dump(model_lstm, open(filename, 'wb'))


# Define a function to perform the forecast



def perform_forecast(date):
    # Load the LSTM model
    model_lstm = pickle.load(open('Forecasting_LSTM_ak.pkl', 'rb'))

    # Prepare the input data for prediction
    input_data = data['y'].values[-window_size:].reshape(1, window_size, 1)

    # Perform the forecast
    forecast = model_lstm.predict(input_data)[0][0]

    # Denormalize the forecasted value
    forecast = scaler.inverse_transform([[forecast]])[0][0]

    return forecast


def main():
    st.markdown("<h1 class='stTitle'>Crude Oil Price Forecasting</h1>", unsafe_allow_html=True)

    # Get user input for the forecast date
    forecast_date = st.date_input("Select a date for oil price prediction")

    # Perform the forecast
    forecast_price = perform_forecast(forecast_date)

    # Display the forecasted price
    st.subheader("Forecasted Crude Oil Price:")
    st.write(f"${forecast_price:.2f}")

    # Plot the historical prices
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data['ds'], data['y'], label='Historical Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Crude Oil Price')
    ax.legend()
    st.subheader('Historical Prices')
    st.pyplot(fig)


if __name__ == '__main__':
    main()
