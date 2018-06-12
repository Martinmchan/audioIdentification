import numpy as np
import tensorflow as tf
import os, keras
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from keras import optimizers
import vggish_input
from random import shuffle
import soundfile as sf

def predict_all(model):
	scaler = StandardScaler()
	path = "./audio_files/testdata"
	os.chdir(path)
	files = os.listdir(".")
	for i in range(0,len(files)):
		if (sf.SoundFile(files[i]).subtype) == "PCM_16":		
			print(files[i])
			unknown = vggish_input.wavfile_to_examples(files[i])
			for j in range(0, unknown.shape[0]):
				unknown[j,:,:] = scaler.fit_transform(unknown[j,:,:])
			unknown = unknown.reshape(unknown.shape[0],96,64,1)
  			unknown = np.repeat(unknown,3,axis=3)
			prediction = model.predict(unknown)			
			for i in range(np.array(prediction).shape[0]):			
				if prediction[i,0] > prediction[i,1]:
					print("gun")
				else:
					print("not_gun")
				


model = load_model("./model2048.h5")

predict_all(model)





