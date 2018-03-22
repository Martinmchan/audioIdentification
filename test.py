import numpy as np
import tensorflow as tf
import os, keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
import vggish_input

path = r"C:\Users\marti\Documents\NeuralNetworks\audioIdentification\train_data_scream"
os.chdir(path)
files = os.listdir()
scream_examples = vggish_input.wavfile_to_examples(files[0])
scream_labels = np.array([[1]]*scream_examples.shape[0])
print(np.shape(scream_examples))
print(np.shape(scream_labels))

scream_examples = scream_examples.reshape(10,96,64,1)
scream_examples = np.repeat(scream_examples,3,axis=3)
print(np.shape(scream_examples))

#model = Sequential()
#model.add(Conv2D(64, 3, 3,batch_size = None, input_shape=(96,64,1)))
#model.add(Activation('relu'))
#model.add(Dense(5))
#model.add(Activation('relu'))
#model.add(Flatten())
#model.add(Dense(1))
#model.add(Activation('sigmoid'))
#model.compile(optimizer='rmsprop',
#          loss='binary_crossentropy',
#          metrics=['accuracy'])


modelVGG = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',input_shape=(96,64,3), pooling='max')


my_model = Sequential()
my_model.add(Dense(256,input_shape=modelVGG.output_shape[1:]))
my_model.add(Activation('relu'))
my_model.add(Dense(256))
my_model.add(Activation('relu'))
my_model.add(Dense(256))
my_model.add(Activation('relu'))
my_model.add(Dense(1))
my_model.add(Activation('sigmoid'))

model = keras.models.Model(inputs= modelVGG.input, outputs= my_model(modelVGG.output))

model.compile(optimizer='rmsprop',
          loss='binary_crossentropy',
          metrics=['accuracy'])

model.fit(scream_examples, scream_labels,batch_size=10)


























