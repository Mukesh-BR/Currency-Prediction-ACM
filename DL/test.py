import numpy as np
from model import full_model
from data_preprocess import generate_data
import keras
period = 30
x,y=generate_data(period)
total=len(x)
x_train=np.array(x[:int(0.8*total)])
y_train=np.array(y[:int(0.8*total)])

x_test=np.array(x[int(0.8*total):])
y_test=np.array(y[int(0.8*total):])

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)


model=full_model(period)
model.compile(optimizer='adam', loss='mse')

model.load_weights("bestweights.hdf5")
y_pred=[]
test=x_test[0]
print(test.shape)


for i in range(len(x_test)):
    test_new=np.reshape(test,(1,period))
    val=model.predict(test_new, verbose=0)
    y_pred.append(val[0])
    test=np.delete(test,0)
    test=np.insert(test,len(test),val[0][0])
    print(test_new,val[0])

for i in range(len(y_test)):
    print(y_pred[i][0],y_test[i])