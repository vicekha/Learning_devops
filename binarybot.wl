(* Load necessary packages *)
Needs["PredictiveInterface`"];
Needs["MachineLearning`"];
Needs["FinancialData`"];

(* Define the stock symbol and time frame *)
symbol = "AAPL";
null

timeFrame = {{2021, 1, 1}, DateList[]};

(* Retrieve historical data with 1 minute candlesticks *)
appleData = FinancialData[AAPL, "OHLCV", timeFrame, "Minute"]

FinancialData[AAPL, "OHLCV", timeFrame, "Minute"]

FinancialData[AAPL, "OHLCV", timeFrame, "Minute"]

(* Define a function to label the data as up or down *)
labelData[data_] := 
  Map[If[#[[2]] > #[[5]], <|"Input" -> Most[#], "Output" -> 1|>, 
    <|"Input" -> Most[#], "Output" -> 0|>] &, data];

(* Split the data into training and validation sets *)
split = Floor[Length[appleData]*0.8];
trainingData = labelData[appleData[[1 ;; split]]];
validationData = labelData[appleData[[split + 1 ;;]]];

(* Train a predictive model with Logistic Regression *)
predictiveModel = Predict[trainingData, Method -> "LogisticRegression"];

(* Define a function to predict the stock price *)
predictStockPrice[] :=
 Module[{newData, prediction},
  (* Retrieve the current stock data *)
  newData = FinancialData[symbol, "OHLCV", {{DateList[] - Quantity[5, "Seconds"], DateList[]}}, "Minute"];

  (* Use the model to predict if the price will go up or down *)
  prediction = predictiveModel[Most[newData[[1]]]];

  (* Print the prediction and accuracy of the model *)
  If[prediction == 1,
    Print["The price of Apple stock is predicted to go up in the next 5 seconds."],
    Print["The price of Apple stock is predicted to go down in the next 5 seconds."]
  ];
  accuracy = ClassifierMeasurements[predictiveModel, validationData, "Accuracy"];
  Print["Model Accuracy: ", Round[accuracy*100, 2], "%"];
 ];




(* Call the predictStockPrice function *)
predictStockPrice[]



