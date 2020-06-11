import tensorflow as tf
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import winsound
import keras
from keras import models
from keras import layers
from keras import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import os
from tqdm import tqdm
path = "./test_data_full/"
label = "cat"
wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

labels=['background noise', 'backward', 'bed', 'bird', 'cat', 'dog', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn',
        'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
indices=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,34]

# Reload the model from the 2 files we saved
with open('CNN_model.json') as json_file:
    json_config = json_file.read()
new_model = keras.models.model_from_json(json_config)
new_model.load_weights('CNN_model_weights.h5')

labels_word = pd.Series(labels)
indices = pd.Series(indices)
data=pd.DataFrame({ 'Total words': labels_word , 'Word indices': indices })

for audio_path in wavfiles:
    #audio_path='test_data_full/tree/eb0676ec_nohash_0.wav'
    print(audio_path)
    winsound.PlaySound(audio_path, winsound.SND_FILENAME)
    y,sr = librosa.load(audio_path,mono=True,sr=None)
    wave = np.asfortranarray(y[::3])
    mfcc = librosa.feature.mfcc(wave,sr=16000, n_mfcc=20)
    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (11 > mfcc.shape[1]):
      pad_width = 11 - mfcc.shape[1]
      mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
      mfcc = mfcc[:, :11]
    Xnew = mfcc.reshape(1,20,11,1)
    ynew = new_model.predict_classes(Xnew)
    print(ynew)

    print(data['Total words'][ynew])