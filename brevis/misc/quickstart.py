
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import itertools
import glob
import os
import pandas as pd
import math
import pydot
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import matplotlib.cm as cm

from brevis.utils import *
import brevis as branching
def Run_alexNet(numEpocs = 2, modelName="", saveName ="",transfer = True,customOptions=""):
    x = tf.keras.models.load_model("models/{}".format(modelName))

    x.summary()
    if saveName =="":
        saveName = modelName
    tf.keras.utils.plot_model(x, to_file="images/{}.png".format(saveName), show_shapes=True, show_layer_names=True)
    # funcModel = models.Model([input_layer], [prev_layer])
    # funcModel = branchingdnn.branches.add(x,["dense","conv2d","max_pooling2d","batch_normalization","dense","dropout"],newBranch)
    # ["max_pooling2d","max_pooling2d_1","dense"]
    funcModel = branching.add(x,["max_pooling2d","max_pooling2d_1","dense"],branch.newBranch_flatten, exact=True, target_input = False)
    # funcModel = branchingdnn.branches.add(x,["dense","dense_1"],newBranch_oneLayer,exact=True)
    # funcModel= x
    funcModel.summary()
    funcModel.save("models/{}".format(saveName))
    dataset = prepare.dataset(tf.keras.datasets.cifar10.load_data(),32,5000,22500,(227,227))

    funcModel = branchingdnn.models.trainModelTransfer(funcModel,dataset, epocs = numEpocs,  transfer = transfer, saveName = saveName,customOptions=customOptions)
    # funcModel.save("models/{}".format(saveName))
    # x = keras.Model(inputs=x.inputs, outputs=x.outputs, name="{}_normal".format(x.name))
    return x

def Run_alexNet_evidence(numEpocs = 2, modelName="", saveName ="",transfer = True,customOptions=""):
    x = tf.keras.models.load_model("models/{}".format(modelName))

    x.summary()
    if saveName =="":
        saveName = modelName
    # tf.keras.utils.plot_model(x, to_file="images/{}.png".format(saveName), show_shapes=True, show_layer_names=True)
    # funcModel = models.Model([input_layer], [prev_layer])
    # funcModel = branchingdnn.branches.add(x,["dense","conv2d","max_pooling2d","batch_normalization","dense","dropout"],newBranch)
    # ["max_pooling2d","max_pooling2d_1","dense"]

    # funcModel.layers.pop()
    # layer = branch.EvidenceEndpoint

    funcModel = branch.add(x,["max_pooling2d"],branch.newBranch_flatten_evidence2,exact=True)
    # funcModel = branchingdnn.branches.add(x,["dense","dense_1"],newBranch_oneLayer,exact=True)
    # funcModel= x
    funcModel.summary()
    funcModel.save("models/{}".format(saveName))
    dataset = prepare.dataset(tf.keras.datasets.cifar10.load_data(),64,5000,22500,(227,227),include_targets=True)

    funcModel = branchingdnn.models.trainModelTransfer(funcModel,dataset, epocs = numEpocs, save = False, transfer = transfer, saveName = saveName,customOptions=customOptions)
    # funcModel.save("models/{}".format(saveName))
    # x = keras.Model(inputs=x.inputs, outputs=x.outputs, name="{}_normal".format(x.name))
    return x

def Run_resnet50v2( numEpocs = 2, modelName="", saveName ="",transfer = True):
    # x = tf.keras.models.load_model("models/{}".format(modelName))
    x = tf.keras.models.load_model("models/{}".format(modelName))

    x.summary()
    if saveName =="":
        saveName = modelName
    tf.keras.utils.plot_model(x, to_file="{}.png".format(saveName), show_shapes=True, show_layer_names=True)
    # funcModel = models.Model([input_layer], [prev_layer])
    # funcModel = branchingdnn.branches.add(x,["dense","conv2d","max_pooling2d","batch_normalization","dense","dropout"],newBranch)
    funcModel = branch.add(x,["conv1_block3_out","conv3_block2_out","conv5_block3_out"],branch.newBranch_flatten,exact=True)

    # funcModel = branchingdnn.branches.add(x,["dense","dense_1"],newBranch_oneLayer,exact=True)
    funcModel.summary()
    funcModel.save("models/{}".format(saveName))
    dataset = prepare.dataset(tf.keras.datasets.cifar10.load_data(),32,5000,22500,(227,227))

    funcModel = branchingdnn.models.trainModelTransfer(funcModel,dataset, epocs = numEpocs, save = False, transfer = transfer, saveName = saveName)
    # funcModel.save("models/{}".format(saveName))
    # x = keras.Model(inputs=x.inputs, outputs=x.outputs, name="{}_normal".format(x.name))
    return x

def Run_inceptionv3( numEpocs = 2, modelName="", saveName ="",transfer = True):
    # x = tf.keras.models.load_model("models/{}".format(modelName))
    x = tf.keras.models.load_model("models/{}".format(modelName))

    x.summary()
    if saveName =="":
        saveName = modelName
    tf.keras.utils.plot_model(x, to_file="{}.png".format(saveName), show_shapes=True, show_layer_names=True)
    # funcModel = models.Model([input_layer], [prev_layer])
    # funcModel = branchingdnn.branches.add(x,["dense","conv2d","max_pooling2d","batch_normalization","dense","dropout"],newBranch)
    funcModel = branch.add(x,["mixed1","mixed3","mixed6"],branch.newBranch_flatten,exact=True)

    # funcModel = branchingdnn.branches.add(x,["dense","dense_1"],newBranch_oneLayer,exact=True)
    funcModel.summary()
    funcModel.save("models/{}".format(saveName))
    dataset = prepare.dataset(tf.keras.datasets.cifar10.load_data(),32,5000,22500,(227,227))

    funcModel = branchingdnn.models.trainModelTransfer(funcModel,dataset, epocs = numEpocs, save = False, transfer = transfer, saveName = saveName)
    # funcModel.save("models/{}".format(saveName))
    # x = keras.Model(inputs=x.inputs, outputs=x.outputs, name="{}_normal".format(x.name))
    return x
