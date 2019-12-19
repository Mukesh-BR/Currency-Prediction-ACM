

import pandas as pd
from google.colab import files
dataset=files.upload()

import numpy as np
dataset=pd.read_csv('INR-vs-USD.csv')

# Setting the date as the index
df = dataset.copy()
df['Date'] = pd.to_datetime(df['Date'])
df.index = df['Date']
del df['Date']

#print(df.head(10))

# Interpolation
df_interpol = df.resample('D').mean()
df_interpol['INR vs USD'] = df_interpol['INR vs USD'].interpolate()

# Resetting index as [0,1...]
k=df_interpol.index
df_interpol.index = np.arange(0, len(df_interpol))
df_interpol = df_interpol.assign(Date = k) 
print(df_interpol)

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import numpy

def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return Series(diff)

diff_values = difference(df_interpol['INR vs USD'].values)
print(diff_values)

def timeseries_to_supervised(data, n_steps):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, n_steps+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	return df

n_steps=30
supervised = timeseries_to_supervised(diff_values, n_steps)
supervised_values = supervised.values
print(supervised_values.shape)

train_size = int(len(df_interpol) * 0.80)
train, test = supervised_values[0:train_size], supervised_values[train_size:]
print(train.shape,test.shape)

#Scaling 
def scale(train, test):

	scaler = MinMaxScaler(feature_range=(-1, 1))
 
	scaler = scaler.fit(train)

	train = train.reshape(train.shape[0], train.shape[1])
 
	train_scaled = scaler.transform(train)

	test = test.reshape(test.shape[0], test.shape[1])
 
	test_scaled = scaler.transform(test)
 
	return scaler, train_scaled, test_scaled

scaler, train_scaled, test_scaled = scale(train, test)

# model architecture - used stochastic gradient descent
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(X,y,epochs=nb_epoch,batch_size=batch_size,verbose=1)
	return model

lstm_model = fit_lstm(train_scaled, 1, 5, 100)

#NORMAL FORECASTING USING LSTM
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]




predictions = list()
actual_y=list()
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	# print(X.shape)
	yhat = forecast_lstm(lstm_model, 1, X)
	# yhat=y
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(df_interpol['INR vs USD'].values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = df_interpol['INR vs USD'].values[len(train) + i + 1]
	actual_y.append(expected)
	print(' Predicted=%f , actual %f' % (yhat,expected))

# report performance
rmse = sqrt(mean_squared_error(actual_y, predictions))
print('Test RMSE: %f' % rmse)

print(test_scaled.shape)

#ROLLING FORECASTING PART

rolling_forecast_result=list()
actual_result=list()
X=test_scaled[0, 0:-1]
print(initial_x.shape)
initial_y=forecast_lstm(lstm_model,1,initial_x)





for i in range(1,len(test_scaled)):
	# make one-step forecast
	X= X[1:]
	X=numpy.append(X,initial_y)
	y=test_scaled[i,-1]
	# X.append(initial_y)
	# X=numpy.append(X,initial_y) 
	# print(X.shape)
	yhat = forecast_lstm(lstm_model, 1, X)
	initial_y=yhat
	# yhat=y
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(df_interpol['INR vs USD'].values, yhat, len(test_scaled)+1-i)
	# store forecast
	rolling_forecast_result.append(yhat)
	expected = df_interpol['INR vs USD'].values[len(train) + i + 1]
	actual_result.append(expected)


	print(' rolling forecast =%f , actual %f' % (yhat,expected))

# report performance
rmse = sqrt(mean_squared_error(rolling_forecast_result, actual_result))
print('Test RMSE: %f' % rmse)