from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import numpy as np

model = Sequential([
    TimeDistributed(Conv2D(2, (2,2), activation= 'relu'), input_shape=(None,150,80,3)),
    TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
    TimeDistributed(Flatten()),
    LSTM(256),
    Dense(1, activation='sigmoid'),
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()


test = np.load('data/test.npz')
train = np.load('data/train.npz')
x_train, y_train  = train["images"], train["labels"]
x_test, y_test  = test["images"], test["labels"]

model.fit(x_train, y_train,validation_data=(x_test, y_test), batch_size=32, epochs=10)
model.save("cnn-lstm-10.h5")
