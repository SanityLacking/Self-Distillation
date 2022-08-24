import os, shutil, random, glob
# import cv2
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# CUDA_VISIBLE_DEVICES = 2
import tensorflow as tf
# import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
import matplotlib.pyplot as plt
root_logdir = os.path.join(os.curdir, "logs\\fit\\")

def AlexLoadModel():
#load Model
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        new_model = load_model('models/saved-model-alexnet-03-0.80.hdf5')
        new_model.summary()
    return new_model

def loadData():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        # width_shift_range=[-200,200],
        rotation_range=90,
        brightness_range=[0.2,1.0],
        # height_shift_range=0.5,
        shear_range=0.2,
        zoom_range=0.2,
        # zca_whitening=True,
        horizontal_flip=True,
        vertical_flip=True,
        
        )

    test_datagen = ImageDataGenerator(rescale=1./255, brightness_range=[0.2,1.0],horizontal_flip=True,
        vertical_flip=True)

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=32,
        shuffle=True,
        seed = 42,
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(224, 224),
        batch_size=32,
        shuffle=False,
        seed = 42,
        class_mode='categorical')

    
    return train_generator, validation_generator


def loadData_temp():
    train_ds, test_ds, validation_ds = loadDataset()
    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
    print("Training data size:", train_ds_size)
    print("Test data size:", test_ds_size)
    print("Validation data size:", validation_ds_size)

    train_ds = (train_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))
    test_ds = (test_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))
    validation_ds = (validation_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))

    for images, labels in train_ds.take(-1):  # only take first element of dataset
        train_images = images.numpy()
        train_labels = labels.numpy()
    for images, labels in test_ds.take(-1):  # only take first element of dataset
        test_images = images.numpy()
        test_labels = labels.numpy()
    return (train_images, train_labels), (test_images, test_labels)

# from tensorflow.contrib import slim
def loadDataset():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    
    CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    validation_images, validation_labels = train_images[:5000], train_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

    return train_ds, test_ds, validation_ds
def loadDataPipeline():
    train_ds, test_ds, validation_ds = loadDataset()
    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
    print("Training data size:", train_ds_size)
    print("Test data size:", test_ds_size)
    print("Validation data size:", validation_ds_size)

    train_ds = (train_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))
    test_ds = (test_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))
    validation_ds = (validation_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))
    return train_ds, test_ds, validation_ds



def visualize(train_ds):
    CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(20,20))
    for i, (image, label) in enumerate(train_ds.take(5)):
        ax = plt.subplot(5,5,i+1)
        plt.imshow(image)
        plt.title(CLASS_NAMES[label.numpy()[0]])
        plt.axis('off')
    return
def get_run_logdir(name=""):
    run_id = time.strftime("run_{}_%Y_%m_%d-%H_%M_%S".format(name))
    return os.path.join(root_logdir, run_id)

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label

def runAndTrainModel():
    resize = 224

    #define the model
    # train_generator, validation_generator = loadData()
    # print(train_generator.class_indices)
    # print(validation_generator.class_indices)
    # AlexNet
    # model = Sequential()

    train_ds, test_ds, validation_ds = loadDataset()
    # visualize(train_ds)
    loadDataPipeline()


    outputs =[]
    # inputs = keras.Input(shape=(227,227,3))
    # x = layers.Conv2D(filters=96, kernel_size=(11,11),
    #                 strides=(4,4), padding='valid',
    #                 input_shape=(resize,resize,3),
    #                 activation='relu')(inputs)
    # x = layers.BatchNormalization()(x)
    # x = layers.MaxPooling2D(pool_size=(3,3),
    #                     strides=(2,2),
    #                     padding='valid')(x)
    # #
    # x = layers.Conv2D(filters=256, kernel_size=(5,5),
    #                 strides=(1,1), padding='same',
    #                 activation='relu')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.MaxPooling2D(pool_size=(3,3),
    #                     strides=(2,2),
    #                     padding='valid')(x)
    # #
    # x = layers.Conv2D(filters=384, kernel_size=(3,3),
    #                 strides=(1,1), padding='same',
    #                 activation='relu')(x)

    # x = layers.Conv2D(filters=384, kernel_size=(3,3),
    #                 strides=(1,1), padding='same',
    # activation='relu')(x)

    # x = layers.Conv2D(filters=256, kernel_size=(3,3),
    #                 strides=(1,1), padding='same',
    #                 activation='relu')(x)
    # x = layers.MaxPool2D(pool_size=(3,3),
    #                     strides=(2,2), padding='valid')(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(4096, activation="relu")(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(4096, activation="relu")(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(1000, activation="relu")(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(2, name="output")(x)
    # x = layers.Softmax()(x)


    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])

    # model = keras.Model(inputs=inputs, outputs=x, name="mnist_model_normal")
    model.summary()

    # #第一段
    # model.add(Conv2D(filters=96, kernel_size=(11,11),
    #                 strides=(4,4), padding='valid',
    #                 input_shape=(resize,resize,3),
    #                 activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(3,3),
    #                     strides=(2,2),
    #                     padding='valid'))
    # #第二段
    # model.add(Conv2D(filters=256, kernel_size=(5,5),
    #                 strides=(1,1), padding='same',
    #                 activation='relu'))
    
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(3,3),
    #                     strides=(2,2),
    #                     padding='valid'))
    # #第三段
    # model.add(Conv2D(filters=384, kernel_size=(3,3),
    #                 strides=(1,1), padding='same',
    #                 activation='relu'))
  
    # model.add(Conv2D(filters=384, kernel_size=(3,3),
    #                 strides=(1,1), padding='same',
    # activation='relu'))
    # model.add(Conv2D(filters=256, kernel_size=(3,3),
    #                 strides=(1,1), padding='same',
    #                 activation='relu'))
    # model.add(MaxPooling2D(pool_size=(3,3),
    #                     strides=(2,2), padding='valid'))
    # #第四段
    # model.add(Flatten())
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))

    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))

    # model.add(Dense(1000, activation='relu'))
    # model.add(Dropout(0.5))

    # # Output Layer
    # model.add(Dense(2, name='output'))
    # model.add(Activation('softmax'))
    # #set the data for the model to fit on

    # model.input

    # for layer in model.layers:
    #     slim.model_analyzer.analyze_vars([layer.output ], print_info=True)


    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata= tf.RunMetadata()

    #train the model, save the model every 10 epochs.
    model.compile(loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy']
            )
    # model.summary()
    # filepath = "models/saved-model-alexnet-{epoch:02d}-{val_acc:.2f}.hdf5"
    # checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    # model.fit(train_data, train_label,
    #     batch_size = 64,
    #     epochs = 50,
    #     #   validation_split = 0.2,
    #     shuffle = True)
    # trainData = tf.data.Dataset.from_generator(train_generator,([tf.float32,tf.float32,tf.float32], tf.float32))

    checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss',verbose=1,save_best_only=True, mode='auto',period=1)


    run_logdir = get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    
    model.fit(train_ds,
        epochs=50,
        validation_data = validation_ds,
        validation_freq = 1,
        callbacks=[tensorboard_cb],        
        )

    test_scores = model.evaluate(test_ds)


    print("overall loss: {}".format(test_scores[0]))
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=5000,
    #     epochs=10,
    #     validation_data=validation_generator,
    #     validation_steps=1000,
    #     #callbacks=callbacks_list,
    #     shuffle=True,
    #     max_queue_size = 50,
    #     use_multiprocessing = True,
    #     workers = 6
    #     )
    model.save('models/alexNet_New.hdf5')
    return 0 
    
def loadAndEvalModel():
    model = AlexLoadModel()
    train_generator, validation_generator = loadData()
    print(train_generator.class_indices)
    print(validation_generator.class_indices)

    # import modelProfiler
    # layerBytes = modelProfiler.getLayerBytes(model,'alexnet')
    #modelProfiler.getFlopsFromArchitecture(model,'alexnet')
    # layerFlops = modelProfiler.getLayerFlops('models/saved-model-alexnet-03-0.80.hdf5','alexnet')


    predict = model.evaluate_generator(validation_generator,steps = 32)
    print(predict)
    # from sklearn.utils.extmath import softmax
    # results = softmax(predict)
    # index_max = np.argmax(results)
    return 0


if __name__ == '__main__':
    
    
   

    # x = tf.keras.models.load_model("models/saved-model-alexnet-03-0.80.hdf5")
    # x.summary()
    # train_generator, validation_generator = loadData()
    runAndTrainModel()
    
    # print(np.array(next(train_generator)).shape)
    
    #loadAndEvalModel()
    print("Task Complete")

    