import numpy as np
import pandas as pd
import tensorflow as tf
from alpha_vantage.timeseries import TimeSeries

# Define Alpha Vantage API key
API_KEY = 'DYI8M7PUS3RQP6U7'

# Define function to retrieve historical data from Alpha Vantage
def get_historical_data(symbol):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, _ = ts.get_intraday(symbol=symbol, interval='1min', outputsize='full')
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data.iloc[::-1]
    return data

# Define function to preprocess data for LSTM model
def preprocess_data(data):
    data = data.drop(['Volume'], axis=1)
    data = data.pct_change().dropna()
    return data

# Define function to split data into training and testing sets
def split_data(data):
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# Define LSTM model
def build_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, input_shape=input_shape),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Retrieve historical data from Alpha Vantage
symbol = 'AAPL'
data = get_historical_data(symbol)

# Preprocess data for LSTM model
data = preprocess_data(data)

# Split data into training and testing sets
train_data, test_data = split_data(data)

# Prepare training and testing data for LSTM model
train_X = np.array(train_data.drop(['Close'], axis=1))
train_Y = np.array(train_data['Close'] > 0).astype(int)
test_X = np.array(test_data.drop(['Close'], axis=1))
test_Y = np.array(test_data['Close'] > 0).astype(int)

# Build LSTM model
input_shape = (train_X.shape[1], 1)
model = build_model(input_shape)

# Train LSTM model
model.fit(train_X, train_Y, epochs=400, batch_size=32, validation_data=(test_X, test_Y))

# Evaluate LSTM model
test_loss, test_accuracy = model.evaluate(test_X, test_Y)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

#this doesnt predict the market after 5 seconds
