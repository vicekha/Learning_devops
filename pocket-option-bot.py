import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from torch.utils.data import TensorDataset, DataLoader

# download historical data for Apple stock
data = yf.download("AAPL", start="2010-01-01", end="2023-03-18")

# create a new column that indicates whether the stock price will increase or decrease in the next 5 seconds
data["target"] = np.where(data["Close"].shift(-10) > data["Close"], 1, 0)

# prepare data for training the LSTM model
inputs = torch.tensor(data[["Open", "High", "Low", "Close", "Volume"]].values, dtype=torch.float32)
targets = torch.tensor(data["target"].values, dtype=torch.long)

# define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])

        return out



# set hyperparameters for the LSTM model
input_size = 5
hidden_size = 32
num_layers = 2
output_size = 2
learning_rate = 0.001
num_epochs = 50

# initialize the LSTM model and optimizer
model = LSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train the LSTM model
for epoch in range(num_epochs):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# use the LSTM model to predict whether the price of Apple stock will increase or decrease after 5 seconds
prediction = model(inputs[-1, :].view(1, 1, -1))
if torch.argmax(prediction) == 1:
    direction = "above"
else:
    direction = "below"

# place a trade based on the predicted direction
# replace the placeholder values below with your actual account information and trade parameters
ssid = """42["auth",{"session":"a:4:{s:10:\"session_id\";s:32:\"d49c3605de77a07c779efe35452c68cc\";s:10:\"ip_address\";s:14:\"73.180.190.212\";s:10:\"user_agent\";s:117:\"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36\";s:13:\"last_activity\";i:1679206314;}fe2b3de822c5ae395bb524564d3b4a63","isDemo":0,"uid":12501694}]"""
asset = "AAPL"
amount = 1
dir = "call" if direction == "above" else "put"
duration = 5
account = PocketOption(ssid)
check_connect, message = account.connect()
if check_connect:
    account.change_balance("PRACTICE")  # "REAL"
    buy_info = account.buy(asset, amount, dir, duration)
    print(f"Predicted direction: {direction}, Trade result: {account.check_win(buy_info['id'])}")
    account.close()
else:
    print("Failed to connect to PocketOption")




# <!-- # Split the data into training and testing sets
# train_size = int(len(X) * 0.8)
# test_size = len(X) - train_size
# train_dataset = TensorDataset(torch.from_numpy(X[:train_size]), torch.from_numpy(y[:train_size]))
# test_dataset = TensorDataset(torch.from_numpy(X[train_size:]), torch.from_numpy(y[train_size:]))

# # Define the model
# model = LSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)

# # Define the loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001) -->

# <!-- # Train the model
# num_epochs = 100
# batch_size = 64
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# for epoch in range(num_epochs):
#     for batch_inputs, batch_targets in train_loader:
#         optimizer.zero_grad()
#         batch_outputs = model(batch_inputs)
#         loss = criterion(batch_outputs, batch_targets)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}') -->