import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from torch.utils.data import TensorDataset, DataLoader



# Download Apple financial data
stock_data = yf.download("AAPL", start="2023-03-11", end="2023-03-18", interval="1m")

# Define input size, hidden size, number of layers, and output size for the LSTM model
input_size = 6
hidden_size = 64
num_layers = 10
output_size = 1
batch_size = 32

# Define the LSTM model
class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out


# Create the input and output sequences
x = []
y = []

for i in range(len(stock_data) - 6):
    x.append(stock_data.iloc[i:i+5].values)
    if stock_data.iloc[i+5]["Close"] > stock_data.iloc[i+4]["Close"]:
        y.append([1])
    else:
        y.append([0])

# Convert the input and output sequences to PyTorch tensors
x = torch.Tensor(x)
y = torch.Tensor(y)

# Split the data into training and testing sets
train_size = int(len(x) * 0.8)
test_size = len(x) - train_size

x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train the LSTM model
learning_rate = 0.001
num_epochs = 100

model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = model(x_train)

    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))



# Test the LSTM model
with torch.no_grad():
    outputs = model(x_test)
    predicted = (outputs > 0).int()
    
    correct = (predicted == y_test).sum().item()
    total = y_test.size(0)
    accuracy = correct / total * 100
    print("Accuracy: {:.2f}%".format(accuracy))
