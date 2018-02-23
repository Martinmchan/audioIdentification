from __future__ import print_function

from random import shuffle

import numpy as np
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_slim

scream = vggish_input.wavfile_to_examples('RjF9D8xDYFg.wav')