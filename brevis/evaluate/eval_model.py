###evaluate the completed models. ####
from re import X
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools
import sys
import math
import pydot
import os
#os.environ["PATH"] += os.pathsep + "C:\Program Files\Graphviz\bin"
#from tensorflow.keras.utils import plot_model
# from Alexnet_kaggle_v2 import * 
import brevis as branching

from brevis.utils import *


def evalBranchMatrix_old(model, input, labels=""):
    num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
    print(type(input))
    if labels == "":
        if type(input)=="tensorflow.python.data.ops.dataset_ops.BatchDataset":
            print("yes")
            pass
        else: 
            print("no")
    iterator = iter(input)
    item = iterator.get_next()
    pred=[]
    labels=[]
    for i in range(100):
        pred.append(model.predict(item[0]))
        labels.append(item[1])
    
    results = throughputMatrix(pred, labels, num_outputs)
    print(results)
    print(pd.DataFrame(results).T)

    return

def loss_function(annealing_rate=1, momentum=1, decay=1, global_loss=False):
    def crossEntropy_loss(labels, outputs): 
#         softmax = tf.nn.softmax(outputs)
        loss = tf.keras.losses.categorical_crossentropy(labels, outputs)
        return loss
    return  crossEntropy_loss
    
class EntropyEndpoint(tf.keras.layers.Layer):
        def __init__(self, num_outputs, name=None, **kwargs):
            super(EntropyEndpoint, self).__init__(name=name)
            self.num_outputs = num_outputs
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        # def build(self, input_shape):
        #     self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'num_outputs': self.num_outputs,
                'name': self.name
            })
            return config

        def call(self, inputs, labels,learning_rate=1):
            # outputs = tf.matmul(inputs,self.kernel)
            # outputs = inputs
            tf.print("inputs",inputs)
            outputs = tf.nn.softmax(inputs)
            tf.print("softmax",outputs)
            entropy = calcEntropy_Tensors(inputs)
            # tf.print("entropy",entropy)
            # print(entropy)
            pred = tf.argmax(outputs,1)
            # tf.print("pred", pred)
            truth = tf.argmax(labels,1)
            # tf.print("truth", truth)
            match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
            # tf.print("match", match)
            # tf.print("match",match)

            # tf.print("succ",tf.reduce_sum(entropy*match),tf.reduce_sum(match+1e-20))
            # tf.print("fail",tf.reduce_sum(entropy*(1-match)),(tf.reduce_sum(tf.abs(1-match))+1e-20) )
            mean_succ = tf.reduce_sum(entropy*match) / tf.reduce_sum(match+1e-20)
            mean_fail = tf.reduce_sum(entropy*(1-match)) / (tf.reduce_sum(tf.abs(1-match))+1e-20) 
            
            self.add_metric(entropy, name=self.name+"_entropy")
            # self.add_metric(total_entropy, name=self.name+"_entropy",aggregation='mean')
            self.add_metric(mean_succ, name=self.name+"_mean_ev_succ",aggregation='mean')
            self.add_metric(mean_fail, name=self.name+"_mean_ev_fail",aggregation='mean')
            
            return inputs

if __name__ == "__main__":
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    
    # print("model to eval: {}".format(args[0]))

    #which model to eval
    #either list a valid model name, aka mnist/alexnet, etc
    #or specify a model filepath

    #which dataset to eval on?
    #check the model name for one of the valid model types and use the default dataset for that.
    
    print("evalModel")

    #load the model
    #load the dataset
    # x = tf.keras.models.load_model("models/alexnet_branched_new_trained.hdf5")
    # normal one
    # x = tf.keras.models.load_model("models/alexnet_branch_pooling.hdf5")

    # test model
    # x = tf.keras.models.load_model("models/resnet50_branched.hdf5")
    # x.summary()
    # print(x.outputs)

    # x = tf.keras.models.load_model("models/alexNetv5.hdf5")
    x = tf.keras.models.load_model('alexnet_entropy_results.hdf5',custom_objects={"EntropyEndpoint":EntropyEndpoint,"crossEntropy_loss":loss_function()})

    y = branching.core.GetResultsCSV(x, tf.keras.datasets.cifar10.load_data(),"entropy_results")
    # y = branchy.GetResultsCSV(x,keras.datasets.mnist.load_data(), "_mnist")
    
    # y = branching.core.evalModel(x, tf.keras.datasets.cifar10.load_data(),"natural")

    # import modelProfiler
    # layerBytes = modelProfiler.getLayerBytes(x,'alexnet_branch_pooling')
    #modelProfiler.getFlopsFromArchitecture(model,'alexnet')
    # layerFlops = modelProfiler.getLayerFlops_old('models/alexnet_branch_pooling.hdf5','alexnet_branch_pooling')

    # y = branchy.BranchEntropyConfusionMatrix(x, tf.keras.datasets.cifar10.load_data())

    # y = branchy.BranchEntropyMatrix(x, tf.keras.datasets.cifar10.load_data())


    #print the model structure summary
    # x.summary()
    #eval the model
    # branchy.eval_branches(x, tf.keras.datasets.cifar10.load_data())
    # output_names = [i.name for i in x.outputs]
    # print(output_names)
    # y = branchy.evalBranchMatrix(x, tf.keras.datasets.cifar10.load_data())
    #print the results

    pass