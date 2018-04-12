import numpy as np
import tensorflow as tf
import os, keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from sklearn.preprocessing import StandardScaler
from keras import optimizers
import vggish_input
from random import shuffle
import soundfile as sf


def _folder_to_mel(path):
  scaler = StandardScaler()
  os.chdir(path)
  files = os.listdir(".")
  sound_examples = vggish_input.wavfile_to_examples(files[0])
  for i in range(0, sound_examples.shape[0]):
	sound_examples[i,:,:] = scaler.fit_transform(sound_examples[i,:,:])
  sound_examples = sound_examples.reshape(sound_examples.shape[0],96,64,1)
  sound_examples = np.repeat(sound_examples,3,axis=3)
  for i in range(1,len(files)):
	print(sf.SoundFile(files[i]).name)
	if (sf.SoundFile(files[i]).subtype) == "PCM_16":  	
		temp_example = vggish_input.wavfile_to_examples(files[i])
		for j in range(0, temp_example.shape[0]):
			temp_example[j,:,:] = scaler.fit_transform(temp_example[j,:,:])
  		temp_example = temp_example.reshape(temp_example.shape[0],96,64,1)
  		temp_example = np.repeat(temp_example,3,axis=3)
  		sound_examples = np.concatenate((sound_examples, temp_example))
  return sound_examples


def _get_all_data_and_label():
	path = "/home/martinch/Documents/UrbanSound8K/audio/fold1"
	noise_examples = _folder_to_mel(path)
	noise_labels = np.array([[1, 0]] * noise_examples.shape[0])
	path = "/home/martinch/Documents/audioIdentification/people_data"
	people_examples = _folder_to_mel(path)
	people_labels = np.array([[0, 1]] * people_examples.shape[0])

	all_examples = np.concatenate((noise_examples, people_examples))
	all_labels = np.concatenate((noise_labels, people_labels))
	labeled_examples = list(zip(all_examples, all_labels))
	shuffle(labeled_examples)

	features = [example for (example, _) in labeled_examples]
	labels = [label for (_, label) in labeled_examples]
	return (features, labels)


(data, labels) = _get_all_data_and_label()

data = np.array(data)
labels = np.array(labels)

modelVGG = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',input_shape=(96,64,3), pooling='max')


my_model = Sequential()
my_model.add(Dense(1024,input_shape=modelVGG.output_shape[1:]))
my_model.add(Activation('relu'))
my_model.add(Dense(1024))
my_model.add(Activation('relu'))
my_model.add(Dense(1024))
my_model.add(Activation('relu'))
my_model.add(Dense(2))
my_model.add(Activation('sigmoid'))
model = keras.models.Model(inputs= modelVGG.input, outputs= my_model(modelVGG.output))



#for layer in model.layers[:16]:
#    layer.trainable = False

opt = optimizers.Adam(lr = 0.0001, epsilon = 0.00000001)
model.compile(opt,
          loss='binary_crossentropy',
          metrics=['accuracy'])



model.fit(data, labels, epochs = 20)























