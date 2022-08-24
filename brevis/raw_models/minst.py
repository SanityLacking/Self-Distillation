import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools
import time
import json
import brevis
from brevis.utils import *


class mnistBranch(BranchyNet):
    """ subclass of BranchyNet for Mnist specific functions and testing
    """ 
    def mainBranch(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(784,)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        return model

    def mnistNormal(self):
        outputs =[]
        inputs = keras.Input(shape=(784,))
        x = layers.Flatten(input_shape=(28,28))(inputs)
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 2
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 3
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 4
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 5
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 1 The main branch exit is refered to as "exit 1" or "main exit" to avoid confusion when adding addtional exits
        output1 = layers.Dense(10, name="output1")(x)
        softmax = layers.Softmax()(output1)

        outputs.append(softmax)
        print(len(outputs))
        model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model_normal")
        model.summary()
        #visualize_model(model,"mnist_normal")
        print(len(model.outputs))

        return model

    def mnistBranchy(self):

        outputs =[]
        inputs = keras.Input(shape=(784,))
        x = layers.Flatten(input_shape=(28,28))(inputs)
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)        
        #exit 2
        outputs = newBranch(x,outputs)
        # outputs.append(layers.Dense(10, name="output2")(x))

        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 3
        # outputs.append(layers.Dense(10, name="output3")(x))
        outputs = newBranch(x,outputs)
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 4
        # outputs.append(layers.Dense(10, name="output4")(x))
        outputs = newBranch(x,outputs)
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 5
        # outputs.append(layers.Dense(10, name="output5")(x))
        outputs = newBranch(x,outputs)
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 1 The main branch exit is refered to as "exit 1" or "main exit" to avoid confusion when adding addtional exits
        output1 = layers.Dense(10, name="output1")(x)
        softmax = layers.Softmax()(output1)
        # x = layers.Dense(64, activation="relu")(x)
        # output2 = layers.Dense(10, name="output2")(x)
        outputs.append(softmax)
        model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model_branched")
        model.summary()
        visualize_model(model,"mnist_branched")
        return model


    def mnistAddBranches(self,model):
        """add branches to the mnist model, aka modifying an existing model to include branches."""
        print(model.inputs)
        inputs = model.inputs
        outputs = []
        print(model.outputs)
        outputs.append(model.outputs)
        for i in range(len(model.layers)):
            print(model.layers[i].name)
            if "dense" in model.layers[i].name:

                outputs = newBranch(model.layers[i].output,outputs)
            # for j in range(len(model.layers[i].inbound_nodes)):
            #     print(dir(model.layers[i].inbound_nodes[j]))
            #     print("inboundNode: " + model.layers[i].inbound_nodes[j].name)
            #     print("outboundNode: " + model.layers[i].outbound_nodes[j].name)
        model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model_branched")
        return model


    def Run_mnistNormal(self, numEpocs = 2):
        """ load a mnist model, add branches to it and train using transfer learning function
        """
        x = self.mnistNormal()
        x = self.trainModel(x,self.loadTrainingData(), epocs = numEpocs,save = True)
        return x

    def Run_mnistTransfer(self, numEpocs = 2):
        """ load a mnist model, add branches to it and train using transfer learning function
        """
        x = tf.keras.models.load_model("models/mnist_trained_.hdf5")
        x = self.addBranches(x,["dropout_1","dropout_2","dropout_3","dropout_4",],newBranch)
        x = self.trainModelTransfer(x,self.loadTrainingData(),epocs = numEpocs, save = True)
        return x


if __name__ == '__main__':
    num_classes = 10
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")


    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()
    batch_size = 128
    epochs = 15

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
