import tensorflow as tf
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import keras
from keras import models
from keras import layers
from keras import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

max_len=11
buckets=20

labels=['background noise', 'backward', 'bed', 'bird', 'cat', 'dog', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn',
        'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
indices=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,34]


PATH = "./npy_data/"
def get_train_test(split_ratio=0.9, random_state=42):
    # Get available labels
    #labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(PATH+labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(PATH+label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

# # Feature dimension
channels = 1
#config.epochs = 50
#config.batch_size = 100
num_classes = 35

X_train = X_train.reshape(X_train.shape[0], buckets, max_len, channels)
X_test = X_test.reshape(X_test.shape[0], buckets, max_len, channels)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

# build model
model = keras.Sequential()
model.add(layers.Conv2D(32,(3,3),input_shape=(buckets, max_len,channels),
    activation='relu'))
model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])

#wandb.init()
model.fit(X_train, y_train_hot, epochs=50, validation_data=(X_test, y_test_hot))

# Save JSON config to disk

json_config = model.to_json()
with open('model_full.json', 'w') as json_file:
    json_file.write(json_config)
# Save weights to disk
model.save_weights('model_full_to_my_weights.h5')
