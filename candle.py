

from pocketoptionapi.stable_api import PocketOption

ssid=r"""42["auth",{"session":"a:4:{s:10:\"session_id\";s:32:\"d49c3605de77a07c779efe35452c68cc\";s:10:\"ip_address\";s:14:\"73.180.190.212\";s:10:\"user_agent\";s:117:\"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36\";s:13:\"last_activity\";i:1679206314;}fe2b3de822c5ae395bb524564d3b4a63","isDemo":0,"uid":12501694}]"""

account = PocketOption(ssid)

check_connect, message = account.connect()

if check_connect:
    account.change_balance("PRACTICE")  # "REAL"
    asset = "EURUSD"
    amount = 1
    dir = "call"  # "call"/"put"
    duration = 30  # sec
    print("Balance: ", account.get_balance())
    buy_info = account.buy(asset, amount, dir, duration)
    # need this to close the connect
    print("----Trade----")
    print("Get: ", account.check_win(buy_info["id"]))
    print("----Trade----")
    print("Balance: ", account.get_balance())
    # need close ping server thread
    account.close()



# import yfinance as yf
# import torch
# import numpy as np
# import time

# # Define the LSTM model class
# class LSTM(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#         self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = torch.nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

#         out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
#         out = self.fc(out[:, -1, :])

#         return out

# Define input size, hidden size, number of layers, and output size for the LSTM model
# input_size = 6
# hidden_size = 32
# num_layers = 5
# output_size = 1
# batch_size = 1

# # Define the function to fetch real-time stock data
# def fetch_realtime_data(ticker):
#     # Fetch the latest minute data for the given stock ticker
#     data = yf.download(ticker, start=time.strftime('%Y-%m-%d')+" 09:30:00", end=time.strftime('%Y-%m-%d')+" 16:00:00", interval="1m")
#     # Normalize the input data
#     data_normalized = normalize_data(data)
#     # Convert the input data to a PyTorch tensor
#     x = torch.Tensor([data_normalized.iloc[-6:-1].values])
#     return x

# # Define the function to make predictions on the latest data using the trained model
# def predict(model, x):
#     with torch.no_grad():
#         outputs = model(x)
#         predicted_prob = torch.sigmoid(outputs).item()
#         predicted_label = 1 if predicted_prob > 0.5 else 0
#     return predicted_label

# # Load the trained LSTM model
# model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size)
# model.load_state_dict(torch.load('lstm_model.pth'))

# # Set the stock ticker for which you want to make real-time predictions
# ticker = "AAPL"

# # Fetch and make predictions on the latest stock data every 5 seconds
# while True:
#     # Fetch the latest minute data for the given stock ticker
#     x = fetch_realtime_data(ticker)
#     # Make predictions on the latest data using the trained model
#     predicted_label = predict(model, x)
#     # Place a call option if the predicted label is 1, otherwise place a put option
#     if predicted_label == 1:
#         print("Place a call option")
#     else:
#         print("Place a put option")
#     # Wait for 5 seconds before making predictions on the latest data again
#     time.sleep(5)