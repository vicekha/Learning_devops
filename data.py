import yfinance as yf
import pandas as pd

# define start and end date of the data you want to fetch
start_date = "2023-02-11"
end_date = "2023-03-19"

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

# print the data
print(df)
