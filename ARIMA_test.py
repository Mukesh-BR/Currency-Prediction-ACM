import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing,SimpleExpSmoothing, Holt
from matplotlib import pyplot as plt
import pandas as pd
from pandas import read_csv
from pandas.plotting import lag_plot
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

names=["Date","Value"]
series = read_csv('INR-vs-USD.csv',index_col=0,names=names,header=0)
series.index = pd.to_datetime(series.index)
resampled_series = series['Value'].resample('MS').mean() #Resample with Month Start frequency
resampled_series.plot()
#series.plot()
plt.show()


#Function that calls ARIMA model to fit and forecast the data
def StartARIMAForecasting(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast()[0]
    return prediction



NumberOfElements = len(resampled_series)

#Use 70% of data as training, rest 30% to Test model
TrainingSize = int(NumberOfElements * 0.7)
TrainingData = resampled_series[0:TrainingSize]
TestData = resampled_series[TrainingSize:NumberOfElements]

#new arrays to store actual and predictions
Actual = [x for x in TrainingData]
Predictions = list()


#in a for loop, predict values using ARIMA model
for timepoint in range(len(TestData)):
    ActualValue =  TestData[timepoint]
    #forcast value
    Prediction = StartARIMAForecasting(Actual, 3,1,0)    
    print('Actual=%f, Predicted=%f' % (ActualValue, Prediction))
    #add it in the list
    Predictions.append(Prediction)
    Actual.append(ActualValue)

#MSE
Error = mean_squared_error(TestData, Predictions)
print('Test Mean Squared Error: %.3f' % Error)

#need to plot