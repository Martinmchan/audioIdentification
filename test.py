from __future__ import print_function

import numpy as np

import os
import vggish_input
import vggish_params
import vggish_slim

def _folder_to_mel(path):
  os.chdir(path)
  files = os.listdir(path)
  sound_examples = vggish_input.wavfile_to_examples(files[0])
  for i in range(1,len(files)):
  	sound_examples = np.concatenate((sound_examples, vggish_input.wavfile_to_examples(files[i])))
  return sound_examples


path = r"C:\Users\marti\Documents\NeuralNetworks\audioIdentification\train_data_scream"
screams = _folder_to_mel(path)
#print(screams.shape([0]))

#path = r"C:\Users\marti\Documents\NeuralNetworks\audioIdentification\train_data_scream"
#os.chdir(path)
#vggish_input.wavfile_to_examples('-GI5PbO6j50.wav')

