from flask import Flask, render_template
from alpha_vantage.timeseries import TimeSeries
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import atexit
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
api_key = 'NR9YKP5UQOMVATFG'
model = load_model('lstm_model.h5')
scaler = MinMaxScaler()
prev_preds = []

def predict(symbol):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=symbol,interval='1min', outputsize='full')
    pct_change = data.pct_change()[1:]
    pct_change = pct_change[['1. open', '2. high', '3. low', '4. close']].values
    pct_change = scaler.fit_transform(pct_change)
    last_five = pct_change[-5:].reshape(1, 5, 4)
    pred = model.predict(last_five)
    if pred > 0.5:
        direction = 'Up'
    else:
        direction = 'Down'
    return direction

def get_prev_predictions_with_correctness():
    predictions = []
    for i in range(len(prev_preds)):
        direction = prev_preds[i]
        if i < len(prev_preds)-1:
            correct = prev_preds[i] == prev_preds[i+1]
        else:
            correct = None
        predictions.append({'direction': direction, 'correct': correct})
    return predictions

def update_prediction():
    global prev_preds
    symbol = 'AAPL'
    prediction = predict(symbol)
    with open('prediction.txt', 'a') as file:
        if len(prev_preds) > 0:
            prev_direction = prev_preds[-1]
            if prev_direction == prediction:
                file.write(f"{symbol}: {prediction}, correct\n")
            else:
                file.write(f"{symbol}: {prediction}, incorrect\n")
        prev_preds.append(prediction)
        if len(prev_preds) > 3:
            prev_preds.pop(0)

scheduler = BackgroundScheduler()
scheduler.add_job(func=update_prediction, trigger="interval", minutes=1)
scheduler.start()

atexit.register(lambda: scheduler.shutdown())

@app.route('/')
def home():
    prediction = prev_preds[-1] if len(prev_preds) > 0 else 'Unknown'
    prev_predictions = get_prev_predictions_with_correctness()
    return render_template('index.html', prediction=prediction, prev_predictions=prev_predictions)

if __name__ == '__main__':
    app.run(debug=True)
