# Plotting functions
# please install 
# pip install python_speech_features

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank

# Basic Libraries
import pandas as pd
import numpy as np

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Building the Model 
import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc


def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

######################################################
##  READ DATA
######################################################
# Reading in data 
df = pd.read_csv('audio/instruments.csv')
print(df.head)
# use file name as index
df = df.set_index("fname")
print(df.head)

# read the length of the file 
for f in df.index:
    rate,signal = wavfile.read('audio/wavfiles/'+f)
    df.at[f,'length'] = signal.shape[0]/rate

# This give us the length of the signal in seconds. 
df.head

# grab the verious different classes
classes = list(np.unique(df.label))
print(classes)

# group by each class can calculate the mean
# length per group.

class_dist = df.groupby(['label'])
print(class_dist['length'].mean())

# reset index to default as we do not need it for now.
df = df.reset_index()

# Plotting the files
# this will give us one file at index 0,2 out of the 30 or so 
# for each class (labels)


import librosa

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return Y, freq

signals = {}
fft = {}
fbank = {}
mfccs = {}

# Plotting the files
# this will give us one file at index 0,0 out of the 30 or so 
# for each class (labels)


import librosa

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return Y, freq

signals = {}
fft = {}
fbank = {}
mfccs = {}


for c in classes:
    print(c)
    wav_files = df[df.label==c]  # all the 30 or so files
    wav_file = wav_files.iloc[0,0]
    try:
        signal, rate = librosa.load('audio/wavfiles/'+wav_file, sr=44100)
        signals[c] = signal
        fft[c] = calc_fft(signal, rate)
    
    # nfft = (44100samples/second) / (40 intervals/sec) = 1103
    # 40 intervals/sec = 25 milliseconds (resolution)
    # nfft = 1103 samples/interval (size of the window)
    # using 26 mel-spec filters 
    # getting data for one second i.e., signal[:44100] or 44100 samples
    
        bank = logfbank(signal[:rate],rate, nfilt=26,nfft=1103).T
        fbank[c] = bank
    
    ## note that nfilt was 26 so we throw away half giving 
    ## us the numcep of 13. So our features (y axis) will be 13
    ## since 1 second = 
    
        mfc= mfcc(signal[:rate],rate, numcep = 13, nfilt=26, nfft=1103).T
        mfccs[c] = mfc
    except: 
         print("Error: class=", c, " FileName=", wav_file)

# Let us do all the plotting
plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()

plot_fbank(fbank)
plt.show()

plot_mfccs(mfccs)
plt.show()

# determine how many samples we can have of a particular length.
# in this case we use 0.1 second as the duration
# summing all the lengths and dividing by 0.1 gives
# us the total number of 0.1 second segments
n_samples = 2 * int(df['length'].sum()/0.1)

# Do stratified sampling 
classes = list(np.unique(df.label))
class_dist = df.groupby('label')['length'].mean()
prob_dist = class_dist/class_dist.sum()
print(prob_dist)
choices= np.random.choice(class_dist.index,p=prob_dist)
print(choices)

# make a class with all the configuration parameters
class Config:
    def __init__(self, nfilt=26, nfeat=13, nfft=1103, rate=44100):
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = (rate/10)
        # each step is 1/10th of a second 
        # will be 44100 values for 0.1 seconds

config = Config()
print(config.step)




# Building and normalizing 
df = df.set_index("fname")
def build_random_feat():
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        # randomly generate a label based on distribution
        rand_class= np.random.choice(class_dist.index,p=prob_dist)
        # randomly select one record of the class selected
        df_rand = df[df.label==rand_class]
        file = np.random.choice(df_rand.index)
        # read the wav file
        rate,wav = wavfile.read('audio/wavfiles/'+file)
        # find the corresponding label
        label = df.at[file,'label']
        # get a random sample of 0.1 seconds from the sample
        # subtract 0.1 second to make sure we always get
        # at least one second
        random_index = np.random.randint(0,wav.shape[0]-config.step)
        sample = wav[int(random_index):int(random_index+config.step)]
        # Now create the feature set and transpose it because we 
        # want time to be the x-axis and mfcc to be y
        X_sample = mfcc(sample, 
                        rate, 
                        numcep = config.nfeat,
                        nfilt = config.nfilt,
                        nfft = config.nfft)
        # Update min and max
        __min = np.amin(X_sample)
        __max = np.amax(X_sample)
        
        if(__min<_min):
            _min = __min
        if(__max>_max):
            _max = __max
        
        # add X and y to correponding lists
        X.append(X_sample)
        # add the integer label correponding to the class
        y.append(classes.index(label))
        
    # convert into numpy arrays
    
    X, y = np.array(X), np.array(y)
    
    # normalize X
    X = (X - _min)/ (_max-_min)
    
    # Reshape for the NN
    
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = to_categorical(y,num_classes=10)
    
    return X, y

# actually build it

X, y = build_random_feat()

# Let us see the shapes
print(X.shape)
print(y.shape)

# the original 1D tensor for y
y_flat = np.argmax(y, axis=1)

#define the convolution network 
input_shape = (X.shape[1], X.shape[2], 1)

# convolutional model 
model = Sequential()
model.add(Conv2D(16,(3,3),activation='relu',
                 strides=(1,1), padding='same',
                 input_shape=input_shape))
model.add(Conv2D(32,(3,3),activation='relu',
                 strides=(1,1), padding='same'))
model.add(Conv2D(64,(3,3),activation='relu',
                 strides=(1,1), padding='same'))
model.add(Conv2D(128,(3,3),activation='relu',
                 strides=(1,1), padding='same'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64,  activation='relu'))
model.add(Dense(10,  activation='softmax'))

model.summary()

# Compile the model 
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics =['acc'])
# handling imbalanced data 
class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
model.fit(X,y,epochs=100, batch_size=32, class_weight=class_weight)


# =============================================================================
# 
# =============================================================================
