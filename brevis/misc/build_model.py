###evaluate the completed models. ####
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools
import sys

# from keras.models import load_model
# from keras.utils import CustomObjectScope
# from keras.initializers import glorot_uniform

import math
import pydot
import os
#os.environ["PATH"] += os.pathsep + "C:\Program Files\Graphviz\bin"
#from tensorflow.keras.utils import plot_model
from brevis.utils import *

from AlexNet_v3 import * 
from branchyNet import BranchyNet




if __name__ == "__main__":
    
    branchy = BranchyNet()
    branchy.ALEXNET = True
    #load the dataset
    x = tf.keras.models.load_model("models/alexNetv4_new.hdf5")
    x.summary()
    #print the model structure summary
    #eval the model
    # buildandcompileModel(x)
    branchy.eval_branches(x, tf.keras.datasets.cifar10.load_data(),"accuracy")
    #print the results

    pass