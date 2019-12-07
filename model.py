from numpy import array
from keras.models import Sequential
from keras.layers import Dense

def full_model(period):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=period))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

