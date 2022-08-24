
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

# import the necessary packages
import brevis
#neptune remote ML monitoring 
from initNeptune import Neptune

from .branches import branch
from .dataset import prepare
from .evaluate import evaluate
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
    def __init__(self, inputs=None, outputs=None, name="", model=None, transfer=True,custom_objects={}):
        ## add default custom objects to the custom objects dictionary, this saves having to define them everytime.
        custom_objects = {**branching.default_custom_objects,**custom_objects} 
        if inputs  is None and model is None and name is not "":
            model = tf.keras.models.load_model(name,custom_objects=custom_objects)
            self.saveLocation = name
            super(BranchModel, self).__init__(inputs = model.inputs, outputs=model.outputs,name=model.name)            
        elif model is None:
            super(BranchModel, self).__init__(inputs = inputs, outputs=outputs,name=name)
        elif model is not None:
            super(BranchModel, self).__init__(inputs = model.inputs, outputs=model.outputs,name=name)
        


        self.transfer = transfer
        self.custom_objects = custom_objects
        ##remap the depths of the layers to match the desired layout for branching
        # self._map_graph_network(self.inputs,self.outputs, True)
        self.branch_active = False
    def set_branchExits(self,value,branchName=None):
        '''
        Sets the branch endpoints for the model.
        inputs:
            value: True or False value to set the branch endpoints
            branchName: list of names of the branch(es) to set the endpoints for
        '''

        if value == True:
            self.branch_active = True

        for layer in self.layers:
            if issubclass(type(layer),branch.BranchEndpoint) or issubclass(type(layer),tf.keras.Model): ## check if the layer is a branch endpoint or subclass
                if branchName is not None:
                    if layer.name in branchName:
                        layer.branch_exit = value
                        print(layer.name, ": ", value)
                else:
                    layer.branch_exit = value
                    print(layer.name, ": ", value)
        
    def evaluate_branch_thresholds(self, test_ds, min_accuracy=None, min_throughput=None ,threshold_fn = evaluate.threshold_fn, stopping_point = None):
        '''
        returns a list of thresholds for each branch endpoint
        inputs:
            test_ds: test dataset to evaluate the thresholds on
            min_accuracy: minimum accuracy to set the threshold for
            min_throughput: minimum throughput to set the threshold for
            warning: function is not garuenteed to return a threshold that satisfies the min_accuracy and min_throughput if it does not exist.
        
        returns:
            list of thresholds for each branch endpoint


        TODO implement the min accuracy and throughput functionality.
        '''
        thresholds = {}
        for layers in self.outputs:
            thresholds.setdefault(layers.name,0)

        branch_predictions = evaluate.collectEvidence_branches(self, test_ds, evidence=True,stopping_point=stopping_point)

        for i, Predictions in enumerate(branch_predictions):
            thresholds[self.outputs[i].name] = threshold_fn(Predictions)

        return thresholds

    def set_branch_thresholds(self,thresholds):
        '''
        This function sets the thresholds for each branch endpoint layer within the branch model.
        inputs: 
            thresholds: dictionary of thresholds in the form of {layer_name:threshold}
        '''
        ###### TODO layer name is different from output name... need to fix this
        for layer in self.layers:
            if issubclass(type(layer),branch.BranchEndpoint) or issubclass(type(layer),tf.keras.Model): ## check if the layer is a branch endpoint or subclass
                for key in thresholds.keys():
                    if layer.name in key or layer.name+'/' in key: #hack to work with the dirrent naming of the layers and outputs. 
                        layer.threshold = thresholds[key]
                        print (layer.name, "set to: ", layer.threshold)
           
    
        return None
                
    # def _map_graph_network(self, inputs, outputs,reverse =True):
    #     """Custom version of _map_graph_network implemented in TF functional
    #     Validates a network's topology and gather its layers and nodes.
        
    #     Args:
    #         inputs: List of input tensors.
    #         outputs: List of outputs tensors.

    #     Returns:
    #         A tuple `(nodes, nodes_by_depth, layers, layers_by_depth)`.
    #         - nodes: list of Node instances.
    #         - nodes_by_depth: dict mapping ints (depth) to lists of node instances.
    #         - layers: list of Layer instances.
    #         - layers_by_depth: dict mapping ints (depth) to lists of layer instances.

    #     Raises:
    #         ValueError: In case the network is not valid (e.g. disconnected graph).
    #     """
    #     print("_map_graph_network")
    #     # "depth" is number of layers between output Node and the Node.
    #     # Nodes are ordered from inputs -> outputs.
    #     nodes_in_decreasing_depth, layer_indices = functional._build_map(outputs)
    #     network_nodes = {
    #         functional._make_node_key(node.layer.name, node.layer._inbound_nodes.index(node))
    #         for node in nodes_in_decreasing_depth
    #     }

    #     nodes_depths = {}  # dict {node: depth value}
    #     layers_depths = {}  # dict {layer: depth value}
    #     if reverse:
    #         for node in reversed(nodes_in_decreasing_depth):
    #             # print(node.layer.name)
    #             # If the depth is not set, the node has no outbound nodes (depth 0).

    #             depth = nodes_depths.setdefault(node, 0)
                
    #             # Update the depth of the corresponding layer
    #             previous_depth = layers_depths.get(node.layer, 0)
    #             # If we've seen this layer before at a higher depth,
    #             # we should use that depth instead of the node depth.
    #             # This is necessary for shared layers that have inputs at different
    #             # depth levels in the graph.-
    #             depth = max(depth, previous_depth)
    #             layers_depths[node.layer] = depth
    #             nodes_depths[node] = depth

    #             # Update the depth of inbound nodes.
    #             # The "depth" of a node is the max of the depths
    #             # of all nodes it is connected to + 1.
    #             for node_dep in node.parent_nodes:
    #                 previous_depth = nodes_depths.get(node_dep, 0)
    #                 nodes_depths[node_dep] = max(depth + 1, previous_depth)
    #                 # print(node.layer.name, " depth: ", depth )
    #         # print("")
    #         # input_depth = 0
    #         # for input_t in inputs:
    #         #     input_layer = input_t._keras_history[0]
    #         #     input_depth = max(input_depth, nodes_depths[input_layer._inbound_nodes[0]])
    #         # for input_t in inputs:
    #         #     input_layer = input_t._keras_history[0]
    #         #     # if input_layer not in layers_depths:
    #         #     #     layers_depths[input_layer] = 0
    #         #     #     layer_indices[input_layer] = -1
    #         #     #     nodes_depths[input_layer._inbound_nodes[0]] = 0
    #         #     #     network_nodes.add(functional._make_node_key(input_layer.name, 0))
    #         #     # else:
    #         #     layers_depths[input_layer] = input_depth
    #         #     # nodes_depths[input_layer._inbound_nodes[0]] = input_depth


    #         for node in (nodes_in_decreasing_depth):
    #             # print(node.layer.name)
    #             # If the depth is not set, the node has no outbound nodes (depth 0).

    #             depth = nodes_depths.setdefault(node, 0)

    #             # Update the depth of the corresponding layer
    #             previous_depth = layers_depths.get(node.layer, 0)
    #             # If we've seen this layer before at a higher depth,
    #             # we should use that depth instead of the node depth.
    #             # This is necessary for shared layers that have inputs at different
    #             # depth levels in the graph.
    #             # if depth is not 0:
    #                 # print(depth)
    #             depth = max(depth, previous_depth)
                
    #             layers_depths[node.layer] = depth
    #             nodes_depths[node] = depth

    #             # Update the depth of inbound nodes.
    #             # The "depth" of a node is the max of the depths
    #             # of all nodes it is connected to + 1.
    #             for node_dep in node.parent_nodes:
    #                 previous_depth = nodes_depths.get(node_dep, 0)
    #                 new_depth  = max(previous_depth - 1, 0)
    #                 depth = max(depth, new_depth)
    #                 print(node.layer.name, ": ", depth, " ", node_dep.layer.name, ": ", previous_depth)
                    
    #                 # depth = previous_depth + 1
    #                 nodes_depths[node] = depth
    #     # Handle inputs that are not connected to outputs.
    #     # We do not error out here because the inputs may be used to compute losses
    #     # and metrics.
    #     for input_t in inputs:
    #         input_layer = input_t._keras_history[0]
    #         if input_layer not in layers_depths:
    #             layers_depths[input_layer] = 0
    #             layer_indices[input_layer] = -1
    #             nodes_depths[input_layer._inbound_nodes[0]] = 0
    #             network_nodes.add(functional._make_node_key(input_layer.name, 0))

    #     # Build a dict {depth: list of nodes with this depth}
    #     nodes_by_depth = collections.defaultdict(list)
    #     for node, depth in nodes_depths.items():
    #         nodes_by_depth[depth].append(node)

    #     # Build a dict {depth: list of layers with this depth}
    #     layers_by_depth = collections.defaultdict(list)
    #     for layer, depth in layers_depths.items():
    #         layers_by_depth[depth].append(layer)

    #     # Get sorted list of layer depths.
    #     depth_keys = list(layers_by_depth.keys())
    #     depth_keys.sort(reverse=True)
    #     # print(depth_keys)
    #     # Set self.layers ordered by depth.
    #     layers = []
    #     for depth in depth_keys:
    #         layers_for_depth = layers_by_depth[depth]
    #         # Network.layers needs to have a deterministic order:
    #         # here we order them by traversal order.
    #         layers_for_depth.sort(key=lambda x: layer_indices[x])
    #         layers.extend(layers_for_depth)

    #     # Get sorted list of node depths.
    #     depth_keys = list(nodes_by_depth.keys())
    #     depth_keys.sort(reverse=True)

    #     # Check that all tensors required are computable.
    #     # computable_tensors: all tensors in the graph
    #     # that can be computed from the inputs provided.
    #     computable_tensors = set()
    #     for x in inputs:
    #         computable_tensors.add(id(x))

    #     layers_with_complete_input = []  # To provide a better error msg.
    #     for depth in depth_keys:
    #         for node in nodes_by_depth[depth]:
    #             layer = node.layer
    #             if layer and not node.is_input:
    #                 for x in nest.flatten(node.keras_inputs):
    #                     if id(x) not in computable_tensors:
    #                         raise ValueError('Graph disconnected: '
    #                                         'cannot obtain value for tensor ' + str(x) +
    #                                         ' at layer "' + layer.name + '". '
    #                                         'The following previous layers '
    #                                         'were accessed without issue: ' +
    #                                         str(layers_with_complete_input))
    #                 for x in nest.flatten(node.outputs):
    #                     computable_tensors.add(id(x))
    #                 layers_with_complete_input.append(layer.name)

    #     # Ensure name unicity, which will be crucial for serialization
    #     # (since serialized nodes refer to layers by their name).
    #     all_names = [layer.name for layer in layers]
    #     for name in all_names:
    #         if all_names.count(name) != 1:
    #             raise ValueError('The name "' + name + '" is used ' +
    #                             str(all_names.count(name)) + ' times in the model. '
    #                             'All layer names should be unique.')
            
        
        
    #     self._nodes_by_depth = nodes_by_depth
        
        
        # return network_nodes, nodes_by_depth, layers, layers_by_depth
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
        # print("_run_internal_graph --custom")
        # print("branches enabled", self.branch_active)
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
        if loop:
            newModel = branch.add_loop(self,branchName, branchPoints,exact=exact, target_input = target_input, compact = compact,num_outputs=num_outputs)
        else:
            newModel = branch.add(self,branchName,branchPoints, exact=exact, target_input = target_input, compact = compact,num_outputs=num_outputs)
        print("branch added", newModel)
        self.__dict__.update(newModel.__dict__)

        return self
    def add_targets(self, num_outputs=10):
        outputs = []
        for i in self.outputs:
            outputs.append(i)
        
        inputs = []
        ready = False
        
        targets= None
        
        for i in self.inputs:
            if i.name == "targets":
                ready = True
            inputs.append(i)        
        print("targets already present? ",ready)

        if not ready:
            print("added targets")
            targets = keras.Input(shape=(num_outputs,), name="targets")
            inputs.append(targets) #shape is (1,) for sparse_categorical_crossentropy
        else:
            targets = self.get_layer('targets').output
        new_model = brevis.BranchModel(inputs=inputs, outputs=outputs,name = self.name, transfer=self.transfer, custom_objects=self.custom_objects)
        self.__dict__.update(new_model.__dict__)

        return self

    def compile(self, loss, optimizer, metrics=['accuracy'], run_eagerly=True, preset="",**kwargs):
        ''' compile the model with custom options, either ones provided here or ones already set'''

        # if preset == "":
            # preset = self.customOptions
        print(preset)
        if preset == "customLoss": 
            print("preset: customLoss")
            loss_fn = evidence_crossentropy()
            super().compile(loss=loss_fn, optimizer=tf.optimizers.SGD(learning_rate=0.001,momentum=0.9), metrics=['accuracy'],run_eagerly=True,**kwargs)
        elif preset == "customLoss_onehot": 
            print("preset: CrossE_onehot")
            super().compile( loss={"dense_2":keras.losses.CategoricalCrossentropy(from_logits=True)}, optimizer=tf.optimizers.SGD(learning_rate=0.01,momentum=0.9), metrics=['accuracy'],run_eagerly=True,**kwargs)

        elif preset == "CrossE": 
            print("preset: CrossE")
            super().compile( loss =tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.optimizers.SGD(learning_rate=0.01,momentum=0.9), metrics=['accuracy'],run_eagerly=True,**kwargs)

        elif preset == "CrossE_Eadd":
            print("preset: CrossE_Eadd")
            entropyAdd = entropyAddition_loss()
            super().compile( optimizer=tf.optimizers.SGD(learning_rate=0.01,momentum=0.9,clipvalue=0.5), loss=[keras.losses.SparseCategoricalCrossentropy(),entropyAdd,entropyAdd,entropyAdd], metrics=['accuracy',confidenceScore, unconfidence],run_eagerly=True,**kwargs)
            # model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.001), loss=[crossE_test, entropyAdd, entropyAdd, entropyAdd], metrics=['accuracy',confidenceScore, unconfidence],run_eagerly=True)
        else:
            print("preset: Other")
        # model.compile(loss=entropyAddition, optimizer=tf.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'],run_eagerly=True)
            super().compile(loss=loss, optimizer=optimizer, metrics=['accuracy'], **kwargs)

    def setTrainable(self,trainable):
        """ sets the trainable status of all main path layers in the model"""
        if trainable == True: 
            print("Freezing Main Layers and setting branch layers training to true")
            for i in range(len(self.layers)):
                # print(model.layers[i].name)
                if "branch" in self.layers[i].name:
                    # print("setting ",self.layers[i].name," training to true")
                    self.layers[i].trainable = True
                else: 
                    # print("setting ",self.layers[i].name," training to false")
                    self.layers[i].trainable = False               
        else:
            print("Setting Main Layers  and branch layers training to true")
            for i in range(len(self.layers)):
                # print(model.layers[i].name)
                self.layers[i].trainable = True
                # print("setting ",self.layers[i].name," training to true")


    def fit(self, train_ds, validation_data=None, epochs=1, callbacks=[], saveName = "", transfer = False, customOptions=""):
        """Train the model that is passed using transfer learning. This function expects a model with trained main branches and untrained (or randomized) side branches.
    """
        logs = []
        num_outputs = len(self.outputs) # the number of output layers for the purpose of providing labels
        # train_ds, test_ds, validation_ds = dataset
        # train_ds, test_ds, validation_ds = prepare.prepareMnistDataset(dataset, batch_size=32)

        #Freeze main branch layers
        #how to iterate through layers and find main branch ones?
        #simple fix for now: all branch nodes get branch in name.
        self.setTrainable(transfer)
        
        
        run_logdir = get_run_logdir(self.name)
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)


        if saveName =="":
            newModelName = "{}_branched".format(self.name )
        else:
            newModelName = saveName
        checkpoint = keras.callbacks.ModelCheckpoint("models/{}".format(newModelName), monitor='val_loss', verbose=1, mode='max')

        # neptune_cbk = Neptune.getcallback(name = newModelName, tags =tags)
        # print("epoc: {}".format(j))
        # results = [j]           
        history =super().fit(train_ds,
                epochs=epochs,
                validation_data=validation_data,
                validation_freq=1,
                callbacks=[tensorboard_cb]+callbacks)
        # print(history)
        
        return self

    # def evaluate(self, *args, **kwargs):
    #     dataset = kwargs.pop('dataset', False)
    #     if dataset:
    #         train_ds, test_ds, validation_ds = dataset
    #         return super().evaluate(test_ds)
    #     else:
    #         return super().evaluate(args, kwargs)



    def trainTransfer(self,numEpocs, loss, optimizer, transfer = True,customOptions=""):
        self.model = brevis.models.trainModelTransfer(self, self.dataset, optimizer=optimizer, epocs = numEpocs, loss=loss, saveName =self.saveName, transfer = transfer,customOptions=customOptions)
        return self

'''
    class version of the self-distilling branched model. inherits from the standard branched model class
'''
class distilled_branch_model(BranchModel):
    def __init__(self, modelName="",saveName="",transfer=True,customOptions="") -> None:
        self.modelName=modelName
        self.saveName=saveName
        self.transfer=transfer
        self.customOptions=customOptions
        self.model = tf.keras.models.load_model("{}".format(modelName))
        # self.model = self.originalModel
        self.branchName = ""
        self.dataset =""
        return None

    def add_branches(self,branchName, branchPoints=[], exact = True, target_input = False):
        if len(branchPoints) == 0:
            return
        # ["max_pooling2d","max_pooling2d_1","dense"]
        # branch.newBranch_flatten
        self.model = branch.add(self.model,branchPoints,branchName, exact=exact, target_input = target_input)
        print(self)
        return self

    def add_distill(self,branchName, branchPoints, teacher_softmax, teaching_features, exact = True, target_input = False):
        if len(branchPoints) == 0:
            return
        # ["max_pooling2d","max_pooling2d_1","dense"]
        # branch.newBranch_flatten
        self.model = branch.add_distil(self.model, teacher_softmax, teaching_features, branchPoints,branchName, exact=exact, target_input = target_input)
        print(self)
        return self


       
class branching():

    
    ''' class version of the model short cut functions
        designed to provide better control of the entire process at the start rather then having to change the internal files over and over
    '''

    # def __init__(self, modelName="",saveName="",transfer=True,custom_objects={}) -> None:
    #         self.modelName=modelName
    #         self.saveName=saveName
    #         self.transfer=transfer
    #         if modelName is not "":
    #             self.model = tf.keras.models.load_model("{}".format(modelName),custom_objects=custom_objects)
    #         else:
    #             self.model = self.branched_model()
    #         self.branchName = ""
    #         self.dataset =""
    #         return None

    ####TODO make this a version that creates a deep copy of a model but of the different subclass.
    def make_branched(model,input_tensors=None,name=""):
        '''
        This function takes a model and returns a branched model
        '''
        if not isinstance(model, models.Model):
            raise ValueError('Expected `model` argument '
                            'to be a `Model` instance, got ', model)
        if isinstance(model, models.Sequential):
            raise ValueError('Expected `model` argument '
                            'to be a functional `Model` instance, '
                            'got a `Sequential` instance instead:', model)
        if not model._is_graph_network:
            raise ValueError('Expected `model` argument '
                            'to be a functional `Model` instance, '
                            'but got a subclass model instead.')

        new_input_layers = {}  # Cache for created layers.
        if input_tensors is not None:
            # Make sure that all input tensors come from a Keras layer.
            input_tensors = nest.flatten(input_tensors)
            for i, input_tensor in enumerate(input_tensors):
                original_input_layer = models._input_layers[i]

            # Cache input layer. Create a new layer if the tensor is originally not
            # from a Keras layer.
            if not backend.is_keras_tensor(input_tensor):
                name = original_input_layer.name
                input_tensor = Input(tensor=input_tensor,
                                    name='input_wrapper_for_' + name)
                newly_created_input_layer = input_tensor._keras_history.layer
                new_input_layers[original_input_layer] = newly_created_input_layer
            else:
                new_input_layers[original_input_layer] = original_input_layer
        model_configs, created_layers = branching._clone_layers_and_model_config(
            model, new_input_layers, branching._clone_layer)
        # Reconstruct model from the config, using the cloned layers.
        input_tensors, output_tensors, created_layers = (
            functional.reconstruct_from_config(model_configs,
                                                created_layers=created_layers))
        return input_tensors,output_tensors, name
    
    def _clone_layer(layer):
        return layer.__class__.from_config(layer.get_config())

    def _clone_layers_and_model_config(model, input_layers, layer_fn):
        """Clones all layers, and returns the model config without serializing layers.

        This function ensures that only the node graph is retrieved when getting the
        model config. The `layer_fn` used to clone layers might not rely on
        `layer.get_config()`, so some custom layers do not define `get_config`.
        Trying to retrieve the config results in errors.

        Args:
            model: A Functional model.
            input_layers: Dictionary mapping input layers in `model` to new input layers
            layer_fn: Function used to clone all non-input layers.

        Returns:
            Model config object, and a dictionary of newly created layers.
        """
        created_layers = {}
        def _copy_layer(layer):
            # Whenever the network config attempts to get the layer serialization,
            # return a dummy dictionary.
            if layer in input_layers:
                created_layers[layer.name] = input_layers[layer]
            elif layer in model._input_layers:
                created_layers[layer.name] = InputLayer(**layer.get_config())
            else:
                created_layers[layer.name] = layer_fn(layer)
            return {}

        config = functional.get_network_config(
            model, serialize_layer_fn=_copy_layer)
        return config, created_layers

    default_custom_objects = {
        #Models
        "BranchModel": BranchModel,
        
        #layers
        "CrossEntropyEndpoint":branch.CrossEntropyEndpoint,
        "EvidenceEndpoint":branch.EvidenceEndpoint,
        
        #losses
        "cross_entropy_evidence":evidence_crossentropy,
    }    