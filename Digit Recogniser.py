import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split

#data import
train = pd.read_csv('Resources/train.csv')
test = pd.read_csv('Resources/test.csv')
#Splitting Data
X_train = train.drop('label', axis=1).copy()
X_test = test.copy()
Y_train = train['label'].copy()
#Reshaping
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1, 28, 28, 1)
#Coverting Data To Float
X_test= X_test.astype('float32')
X_train= X_train.astype('float32')
#Normalizing Data
X_train =  X_train/ 255.0
X_test = X_test / 255.0

validation_size= 0.2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size= validation_size)
#Categorising Data
testY = tf.keras.utils.to_categorical(Y_train)
trainY = tf.keras.utils.to_categorical(Y_val)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# Initialising the CNN
model = Sequential()

# Step 1 - Convolution
model.add(Convolution2D(32, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2),padding='same'))
model.add(layers.Dropout(0.2))

model.add(Convolution2D(32, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2),padding='same'))
model.add(Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss= tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=64, epochs=50, verbose=1)
model.summary()
model.save("my_model")

_, acc = model.evaluate(X_val, Y_val, verbose=0)
acc*100