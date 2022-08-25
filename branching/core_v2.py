import collections
import copy
import itertools
import warnings
from sqlalchemy import true

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_layer as input_layer_module
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.saving.saved_model import network_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
from tensorflow.python.keras.engine import functional

import itertools
import json
import math
import os
import time

import numpy as np
import pydot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.python.ops.gen_math_ops import Xlogy
import importlib
import warnings
# import the necessary packages
import branching
from .branches import branch
from .dataset import prepare
from .evaluate import evaluate


#neptune remote ML monitoring 

#check if neptune is installed, if not, continue without it after a warning.
neptune_spec = importlib.util.find_spec("neptune")
NepLogging = False
# NepLogging = neptune_spec is not None
# if NepLogging:
#     warnings.warn("Logging module Neptune was found, Cloud Logging can be enabled, call enable_neptune to do so. Check initNeptune.py to configure settings")
#     from initNeptune import Neptune
# else:
#     warnings.warn("Logging module Neptune was not found, Cloud Logging will not be enabled, check https://docs.neptune.ai/getting-started/installation for more information on how to set this up")

# def enable_neptune(val=True):
#     NeptLogging = val
#     print("neptune is now set to {}. is the module loaded to prevent errors later?".format(NepLogging))

#local imports
from .utils import *

# from keras.models import load_model
# from keras.utils import CustomObjectScope
# from keras.initializers import glorot_uniform



#os.environ["PATH"] += os.pathsep + "C:\Program Files\Graphviz\bin"
#from tensorflow.keras.utils import plot_model


# from Alexnet_kaggle_v2 import * 





# ALEXNET = False
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
root_logdir = os.path.join(os.curdir, "logs\\fit\\")





class BranchModel(tf.keras.Model):
    '''
    Branched model sub-class. 
    Acts as a drop in replacement keras model class, with the additional functionality of adding branches to the model.
            
    '''
    
    def __init__(self, inputs=None, outputs=None, name="", model=None, freeze=True,custom_objects={}):
        ## add default custom objects to the custom objects dictionary, this saves having to define them everytime.
        # self.default_custom_objects ={
        #     "tags": []
        # }
        # custom_objects = {**self.default_custom_objects,**custom_objects} 
        if inputs  is None and model is None and name is not "":
            model = tf.keras.models.load_model(name,custom_objects=custom_objects)
            self.saveLocation = name
            super(BranchModel, self).__init__(inputs = model.inputs, outputs=model.outputs,name=model.name)            
        elif model is None:
            super(BranchModel, self).__init__(inputs = inputs, outputs=outputs,name=name)
        elif model is not None:
            super(BranchModel, self).__init__(inputs = model.inputs, outputs=model.outputs,name=name)
        


        self.freeze = freeze
        self.custom_objects = custom_objects
        ##remap the depths of the layers to match the desired layout for branching
        # self._map_graph_network(self.inputs,self.outputs, True)
        self.branch_active = False
        
    def _run_internal_graph(self, inputs, training=None, mask=None):
        """custom version of _run_internal_graph
            used to allow for interuption of the graph by an internal layer if conditions are met.
        Computes output tensors for new inputs.

        # Note:
            - Can be run on non-Keras tensors.

        Args:
            inputs: Tensor or nested structure of Tensors.
            training: Boolean learning phase.
            mask: (Optional) Tensor or nested structure of Tensors.

        Returns:
            output_tensors
        """
        inputs = self._flatten_to_reference_inputs(inputs)
        if mask is None:
            masks = [None] * len(inputs)
        else:
            masks = self._flatten_to_reference_inputs(mask)
        for input_t, mask in zip(inputs, masks):
            input_t._keras_mask = mask

        # Dictionary mapping reference tensors to computed tensors.
        tensor_dict = {}
        tensor_usage_count = self._tensor_usage_count
        for x, y in zip(self.inputs, inputs):
            y = self._conform_to_reference_input(y, ref_input=x)
            x_id = str(id(x))
            tensor_dict[x_id] = [y] * tensor_usage_count[x_id]

        nodes_by_depth = self._nodes_by_depth
        depth_keys = list(nodes_by_depth.keys())
        depth_keys.sort(reverse=True)
    
        for depth in depth_keys:
            nodes = nodes_by_depth[depth]
            for node in nodes:
                # print(node.layer.name)
                if node.is_input:
                    continue  # Input tensors already exist.

                if any(t_id not in tensor_dict for t_id in node.flat_input_ids):
                    continue  # Node is not computable, try skipping.

                args, kwargs = node.map_arguments(tensor_dict)
                outputs = node.layer(*args, **kwargs)
                # Update tensor_dict.
                for x_id, y in zip(node.flat_output_ids, nest.flatten(outputs)):
                    tensor_dict[x_id] = [y] * tensor_usage_count[x_id]
                ## check if branch exiting is turned on and if current layer is a potential exit.
                # print(node.layer.name, hasattr(node.layer, 'branch_exit'))
                if not training:
                    if self.branch_active == True and hasattr(node.layer, 'branch_exit'):  
                        ## check if the confidence of output of the layer is equal to or above the threshold hyperparameter
                        # print("threshold: ", node.layer.threshold, "evidence: ", tf.reduce_sum(node.layer.evidence(outputs)))
                        if node.layer.branch_exit and (tf.reduce_sum(node.layer.evidence(outputs)) >= node.layer.confidence_threshold): ##check if current layer's exit is active
                            # print("branch exit activated")
                            output_tensors = []
                            for x_id, y in zip(node.flat_output_ids, nest.flatten(outputs)):
                                for x in self.outputs:
                                    output_id = str(id(x))  
                                    if output_id == x_id:
                                        output_tensors.append(tensor_dict[x_id])
                                    else:
                                        # print(tensor_dict[x_id][0].shape)
                                        output_tensors.append(tf.zeros(tensor_dict[x_id][0].shape))
                                    # x_id_output = str(id(x))
                                    # assert x_id in tensor_dict, 'Could not compute output ' + str(x)
                                    # output_tensors.append(tensor_dict[x_id])

                            return nest.pack_sequence_as(self._nested_outputs, output_tensors)
        output_tensors = []
        for x in self.outputs:
            x_id = str(id(x))
            assert x_id in tensor_dict, 'Could not compute output ' + str(x)
            output_tensors.append(tensor_dict[x_id].pop())

        return nest.pack_sequence_as(self._nested_outputs, output_tensors)

    def add_branches(self,branchName, branchPoints=[], exact = True, target_input = False, compact = False, loop=True,num_outputs=10):
        if len(branchPoints) == 0:
            return
        # ["max_pooling2d","max_pooling2d_1","dense"]
        # branch.newBranch_flatten
        # if loop:
            # newModel = branch.add_loop(self,branchName, branchPoints,exact=exact, target_input = target_input, compact = compact,num_outputs=num_outputs)
        else:
            newModel = branch.add(self,branchName,branchPoints, exact=exact, target_input = target_input, compact = compact,num_outputs=num_outputs)
        print("branch added", newModel)
        self.__dict__.update(newModel.__dict__)

        return self
    

    def compile(self, loss, optimizer, metrics=['accuracy'], run_eagerly=True,**kwargs):
        ''' compile the model with custom options, either ones provided here or ones already set'''
        super().compile(loss=loss, optimizer=optimizer, metrics=['accuracy'], **kwargs)

    def setTrainable(self,trainable):
        """ sets the trainable status of all main path layers in the model"""
        if trainable == True: 
            print("Freezing Main Layers and setting branch layers training to true")
            for i in range(len(self.layers)):
                if "branch" in self.layers[i].name:
                    self.layers[i].trainable = True
                else: 
                    self.layers[i].trainable = False               
        else:
            print("Setting Main Layers  and branch layers training to true")
            for i in range(len(self.layers)):
                self.layers[i].trainable = True
                # print("setting ",self.layers[i].name," training to true")

    # def fit(self, train_ds, validation_data=None, epochs=1, callbacks=[], saveName = "", transfer = False, customOptions=""):
    #     """Train the model that is passed using transfer learning. This function expects a model with trained main branches and untrained (or randomized) side branches.
    # """
    #     self.setTrainable(transfer)
    #     run_logdir = get_run_logdir(self.name)
    #     tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    #     # if saveName =="":
    #     #     newModelName = "{}_branched".format(self.name )
    #     # else:
    #     #     newModelName = saveName
    #     # checkpoint = keras.callbacks.ModelCheckpoint("models/{}".format(newModelName), monitor='val_loss', verbose=1, mode='max')

    #     history =super().fit(train_ds,
    #             epochs=epochs,
    #             validation_data=validation_data,
    #             validation_freq=1,
    #             callbacks=[tensorboard_cb]+callbacks)
        
    #     return self

    # def fit(self, train_ds, validation_data=None, validation_freq = 1, epochs=1, callbacks=[], saveName = "", freeze = False, custom_objects={}):
    #     """
    #     Train the model that is passed using transfer learning. This function expects a model with trained main branches and untrained (or randomized) side branches.
    #     """
    #     # custom_objects = {**self.default_custom_objects,**custom_objects} 
    #     self.setTrainable(freeze) #Freeze main branch layers
    #     run_logdir = get_run_logdir(self.name)
    #     tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    #     if saveName =="":
    #         newModelName = "{}_branched".format(self.name )
    #     else:
    #         newModelName = saveName
    #     # checkpoint = keras.callbacks.ModelCheckpoint("models/{}".format(newModelName), monitor='val_loss', verbose=1, mode='max')
    #     # print("nep {}".format(NepLogging))
    #     # if NepLogging == True:
    #     #     neptune_cbk = Neptune.getcallback()
    #     #     callbacks = callbacks + neptune_cbk

    #     history =super().fit(train_ds,
    #             epochs=epochs,
    #             validation_data=validation_data,
    #             )
    #             # callbacks=[tensorboard_cb]+callbacks)
        
    #     return self

    # def evaluate(self, *args, **kwargs):
        # return super().evaluate(args, kwargs)


class Distill_BranchModel(BranchModel):        
    '''
        class version of the self-distilling branched model. inherits from the standard branched model class
    '''
    def __init__(self, inputs=None, outputs=None, name="", model=None, freeze=True,custom_objects={}) -> None:
        super(Distill_BranchModel, self).__init__(inputs = inputs, outputs=outputs,name=name, model=model, freeze=freeze,custom_objects=custom_objects)            
        return None

    def add_distill(self, teacher, branch_layers, branch_points,  teaching_features=None, exact = True):
        if len(branch_layers) == 0:
            return
        newModel = branch.add_distil(self, teacher,branch_layers,branch_points, teaching_features,  exact=exact)
        print("branches added, new outputs", newModel.outputs)
        self.__dict__.update(newModel.__dict__)
        return self

    def add_branches(self,branchName, branchPoints=[], exact = True, target_input = False, compact = False, loop=True,num_outputs=10):
        if len(branchPoints) == 0:
            return
        # ["max_pooling2d","max_pooling2d_1","dense"]
        # branch.newBranch_flatten
        # if loop:
            # newModel = branch.add_loop(self,branchName, branchPoints,exact=exact, target_input = target_input, compact = compact,num_outputs=num_outputs)
        else:
            newModel = branch.add(self,branchName,branchPoints, exact=exact, target_input = target_input, compact = compact,num_outputs=num_outputs)
        print("branch added", newModel)
        self.__dict__.update(newModel.__dict__)

        return self
      