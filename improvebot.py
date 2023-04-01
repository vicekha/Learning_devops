import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data from Alpha Vantage API
api_key = 'DYI8M7PUS3RQP6U7'
symbol = 'AAPL'
interval = '1min'
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_intraday(symbol=symbol,interval=interval, outputsize='full')

# Preprocess data
def preprocess_data(data):
    # Calculate percent change in price from one minute to the next
    pct_change = data.pct_change()[1:]
    pct_change = pct_change[['1. open', '2. high', '3. low', '4. close']].values
    # Scale data to be between 0 and 1
    scaler = MinMaxScaler()
    pct_change = scaler.fit_transform(pct_change)
    return pct_change

data = preprocess_data(data)

# Split data into training and testing sets
def split_data(data, train_size=0.8):
    train_size = int(len(data) * train_size)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

train_data, test_data = split_data(data)

# Define model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

input_shape = (5, 4) # 5 timesteps, 4 features
model = build_model(input_shape)

# Train model
X_train, y_train = [], []
for i in range(5, len(train_data)):
    X_train.append(train_data[i-5:i])
    y_train.append(1 if train_data[i, 3] > train_data[i-1, 3] else 0) # 1 if close price is up, 0 if down
X_train, y_train = np.array(X_train), np.array(y_train)
model.fit(X_train, y_train, epochs=1000, batch_size=64)

# Evaluate model
X_test, y_test = [], []
for i in range(5, len(test_data)):
    X_test.append(test_data[i-5:i])
    y_test.append(1 if test_data[i, 3] > test_data[i-1, 3] else 0) # 1 if close price is up, 0 if down
X_test, y_test = np.array(X_test), np.array(y_test)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# Make predictions
last_five = test_data[-5:].reshape(1, 5, 4) # last 5 minutes of test data
pred = model.predict(last_five)
if pred > 0.5:
    print('Predicted direction: up')
else:
    print('Predicted direction: down')

model.save('lstm_model.h5')
