import os, re, time, json
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds

print("Tensorflow version " + tf.__version__)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


BATCH_SIZE = 32 
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#Matplotlib config
plt.rc('image', cmap='gray')
plt.rc('grid', linewidth=0)
plt.rc('xtick', top=False, bottom=False, labelsize='large')
plt.rc('ytick', left=False, right=False, labelsize='large')
plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
plt.rc('text', color='a8151a')
plt.rc('figure', facecolor='F0F0F0')# Matplotlib fonts
MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")


(training_images, training_labels) , (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()

def preprocess_image_input(input_images):
  input_images = input_images.astype('float32')
  output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
  return output_ims

train_X = preprocess_image_input(training_images)
valid_X = preprocess_image_input(validation_images)

'''
Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
Input size is 224 x 224.
'''
def feature_extractor(inputs):

  feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')(inputs)
  return feature_extractor


'''
Defines final dense layers and subsequent softmax layer for classification.
'''
def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x

'''
Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
Connect the feature extraction and "classifier" layers to build the model.
'''
def final_model(inputs):

    resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)

    return classification_output

'''
Define the model and compile it. 
Use Stochastic Gradient Descent as the optimizer.
Use Sparse Categorical CrossEntropy as the loss function.
'''

def prepareDataset( batchsize=64):
        # tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

        CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # validation_images, validation_labels = train_images[:5000], alt_trainLabels[:5000]
        # train_ds = tf.data.Dataset.from_tensor_slices((train_images, alt_trainLabels))
        # test_ds = tf.data.Dataset.from_tensor_slices((test_images, alt_testLabels))
        train_labels = tf.keras.utils.to_categorical(train_labels,10)   
        test_labels = tf.keras.utils.to_categorical(test_labels,10)
        
        ###normal method
        validation_images, validation_labels = train_images[:5000], train_labels[:5000]
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

        def augment_images(image, label):
            # Normalize images to have a mean of 0 and standard deviation of 1
            # image = tf.image.per_image_standardization(image)
            # Resize images from 32x32 to 277x277
            image = tf.image.resize(image, (224,224))
            return image, label


        train_ds_size = len(list(train_ds))
        test_ds_size = len(list(test_ds))
        validation_ds_size = len(list(validation_ds))

        print("trainSize {}".format(train_ds_size))
        print("testSize {}".format(test_ds_size))

        train_ds = (train_ds
                        .map(augment_images)
                        .shuffle(buffer_size=train_ds_size)
                        .batch(batch_size=batchsize, drop_remainder=True))

        test_ds = (test_ds
                        .map(augment_images)
                        #   .shuffle(buffer_size=train_ds_size)
                        .batch(batch_size=batchsize, drop_remainder=True))

        validation_ds = (validation_ds
                        .map(augment_images)
                        #   .shuffle(buffer_size=validation_ds_size)
                        .batch(batch_size=batchsize, drop_remainder=True))

        print("testSize2 {}".format(len(list(test_ds))))
        return train_ds, test_ds, validation_ds


def define_compile_model():
  inputs = tf.keras.layers.Input(shape=(32,32,3))
  
  classification_output = final_model(inputs) 
  model = tf.keras.Model(inputs=inputs, outputs = classification_output)
 
  model.compile(optimizer='SGD', 
                loss='sparse_categorical_crossentropy',
                metrics = ['accuracy'])
  
  return model

base_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=(32, 32, 3),
     weights='imagenet',include_top=False)


x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=x)


model.compile(optimizer='SGD', 
                loss='categorical_crossentropy',
                metrics = ['accuracy'])

model.summary()
train_ds, test_ds, validation_ds = prepareDataset(32)

EPOCHS = 3
for i in range(EPOCHS):
    history = model.fit(train_ds, epochs=EPOCHS, validation_data = validation_ds, batch_size=32)
    loss, accuracy = model.evaluate(test_ds, batch_size=32)
    model.save("inception_finetuned.hdf5")