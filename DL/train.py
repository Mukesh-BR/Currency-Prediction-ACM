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

callbacks = [
    keras.callbacks.EarlyStopping(monitor='loss', patience=25, verbose=1),
    keras.callbacks.ModelCheckpoint("Resnet_50_{epoch:03d}.hdf5", monitor='loss', verbose=1, mode='auto'),
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=1e-6),
    keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    #NotifyCB
]

model=full_model(period)
model.compile(optimizer='adam', loss='mse')
model.fit(x_train,y_train,batch_size=16,validation_data=(x_test,y_test),epochs=100,verbose=1,callbacks=callbacks)
model.load_weights("Resnet_50_100.hdf5")
y_pred=[]
test=x_test[0]
print(test.shape)
# for i in range(len(x_test)):
#     test_new=np.reshape(test,(1,period))
#     val=model.predict(test_new, verbose=0)
#     y_pred.append(val[0])
#     test=np.delete(test,0)
#     test=np.insert(test,len(test),val[0][0])
#     print(test_new,val[0])

# for i in range(len(y_test)):
#     print(y_pred[i][0],y_test[i])