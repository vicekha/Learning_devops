import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# define start and end date of the data you want to fetch
start_date = "2023-02-12"
end_date = "2023-03-19"
writer = SummaryWriter()
# split the date range into multiple intervals of 7 days each
date_ranges = pd.date_range(start=start_date, end=end_date, freq="7D")

# create an empty DataFrame to store the data
df = pd.DataFrame()

# fetch data for each 7-day interval and append it to the DataFrame
for i in range(len(date_ranges)-1):
    start = date_ranges[i].strftime('%Y-%m-%d')
    end = date_ranges[i+1].strftime('%Y-%m-%d')
    temp = yf.download("AAPL", start=start, end=end, interval="1m")
    df = pd.concat([df, temp])



# Download Apple financial data
stock_data = df
# Normalize the input data
def normalize_data(data):
    data = data.copy()
    for feature_name in data.columns:
        min_value = data[feature_name].min()
        max_value = data[feature_name].max()
        data[feature_name] = (data[feature_name] - min_value) / (max_value - min_value)
    return data

stock_data_normalized = normalize_data(stock_data)

# Define input size, hidden size, number of layers, and output size for the LSTM model
input_size = 6
hidden_size = 32
num_layers = 5
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
    x.append(stock_data_normalized.iloc[i:i+5].values)
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
learning_rate = 0.005
num_epochs = 40000

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
    writer.add_scalar('Loss/train', loss.item(), epoch)
# Test the LSTM model
with torch.no_grad():
    x_test_normalized = normalize_data(stock_data.iloc[train_size+5:len(stock_data)-1])
    x_test = torch.Tensor([x_test_normalized.iloc[i:i+5].values for i in range(len(x_test_normalized)-5)])
    y_test = torch.Tensor([1 if stock_data.iloc[train_size+i+5]["Close"] > stock_data.iloc[train_size+i+4]["Close"] else 0 for i in range(len(x_test_normalized)-5)])
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Make predictions on the test dataset using the trained model
    
    outputs = model(x_test)
    predicted_probs = torch.sigmoid(outputs)
    predicted_labels = (predicted_probs > 0.5).float().squeeze()

# Convert the true labels to a NumPy array
    true_labels = y_test.numpy()

# Calculate the accuracy
    accuracy = (predicted_labels.numpy() == true_labels).mean() * 100
    print("Accuracy: {:.2f}%".format(accuracy))
    writer.add_scalar('Accuracy/test', accuracy, 0)

writer.close()

# Save the trained LSTM model
torch.save(model.state_dict(), 'lstm_model.pth')

# # Load the saved LSTM model
# model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size)
# model.load_state_dict(torch.load('lstm_model.pth'))
