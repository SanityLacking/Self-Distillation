import os, re, time, json
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from matplotlib import pyplot as plt


model = tf.keras.models.load_model("resnet50_finetuned.hdf5")
model.summary()

for i, layer in enumerate(model.layers):
    print(layer.name)