


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
from .evaluate import branchy_eval as eval
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



class BranchingDnn:

    ALEXNET = False
        ###### RUN MODEL SHORTCUTS ######
    
    ''' class version of the model short cut functions
        designed to provide better control of the entire process at the start rather then having to change the internal files over and over
    '''
    class branched_model:
        def __init__(self, modelName="",saveName="",transfer=True,custom_objects={}) -> None:
            self.modelName=modelName
            self.saveName=saveName
            self.transfer=transfer
            
            self.model = tf.keras.models.load_model("{}".format(modelName),custom_objects=custom_objects)
            # self.model = self.originalModel
            self.branchName = ""
            self.dataset =""
            return None
        
     

        def build(self):
            return
        
        def saveName(self, saveName):
            self.saveName=saveName
            return self

        def save(self,modelName):
            saveModel(self.model,"modelName")
            return self
        ### set the model to an already existing model. This is used primarly to continue training of previously started branch models. 
        def set_model(self, model):
            if isinstance(model,str):
                self.model = tf.keras.models.load_model("{}".format(model))
            else:
                self.model = model
            return self

        def set_branches(self, branchName=""):
            self.branchName = branchName
            return self

        def add_branches(self,branchName, branchPoints=[], exact = True, target_input = False):
            if len(branchPoints) == 0:
                return
            # ["max_pooling2d","max_pooling2d_1","dense"]
            # branch.newBranch_flatten
            self.model = branch.add(self.model,branchPoints,branchName, exact=exact, target_input = target_input)
            print(self)
            return self

        def set_dataset(self, dataset):
            self.dataset = dataset
            return self
        
        def fit(self, train_ds, validation_ds, numEpocs, transfer = True):
            
        def train (self,numEpocs, loss, optimizer, saveName = "", transfer = True,customOptions=""):
            self.model = brevis.models.trainModel(self.model, loss, optimizer, self.dataset, epocs = numEpocs, saveName =saveName, transfer = transfer,customOptions=customOptions)
            return self
        def trainTransfer(self,numEpocs, loss, optimizer, transfer = True,customOptions=""):
            self.model = brevis.models.trainModelTransfer(self.model,self.dataset, optimizer=optimizer, epocs = numEpocs, loss=loss, saveName =self.saveName, transfer = transfer,customOptions=customOptions)
            return self

    '''
        class version of the self-distilling branched model. inherits from the standard branched model class
    '''
    class distilled_branch_model(branched_model):
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


        
    def Run_alexNet(numEpocs = 2, modelName="", saveName ="",transfer = True,customOptions=""):
        x = tf.keras.models.load_model("models/{}".format(modelName))

        x.summary()
        if saveName =="":
            saveName = modelName
        tf.keras.utils.plot_model(x, to_file="images/{}.png".format(saveName), show_shapes=True, show_layer_names=True)
        # funcModel = models.Model([input_layer], [prev_layer])
        # funcModel = branchingdnn.branches.add(x,["dense","conv2d","max_pooling2d","batch_normalization","dense","dropout"],newBranch)
        # ["max_pooling2d","max_pooling2d_1","dense"]
        funcModel = branch.add(x,["max_pooling2d","max_pooling2d_1","dense"],branch.newBranch_flatten, exact=True, target_input = False)
        # funcModel = branchingdnn.branches.add(x,["dense","dense_1"],newBranch_oneLayer,exact=True)
        # funcModel= x
        funcModel.summary()
        funcModel.save("models/{}".format(saveName))
        dataset = prepare.dataset(tf.keras.datasets.cifar10.load_data(),32,5000,22500,(227,227))

        funcModel = brevis.models.trainModelTransfer(funcModel,dataset, epocs = numEpocs,  transfer = transfer, saveName = saveName,customOptions=customOptions)
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

        funcModel = brevis.models.trainModelTransfer(funcModel,dataset, epocs = numEpocs, save = False, transfer = transfer, saveName = saveName,customOptions=customOptions)
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

        funcModel = brevis.models.trainModelTransfer(funcModel,dataset, epocs = numEpocs, save = False, transfer = transfer, saveName = saveName)
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

        funcModel = brevis.models.trainModelTransfer(funcModel,dataset, epocs = numEpocs, save = False, transfer = transfer, saveName = saveName)
        # funcModel.save("models/{}".format(saveName))
        # x = keras.Model(inputs=x.inputs, outputs=x.outputs, name="{}_normal".format(x.name))
        return x


    def Run_mnistNet( numEpocs = 5, modelName="", saveName ="",transfer = True):
        x = tf.keras.models.load_model("models/{}".format(modelName))
        x.summary()
        if saveName =="":
            saveName = modelName
        # funcModel = models.Model([input_layer], [prev_layer])
        # funcModel = branchingdnn.branches.add(x,["dense","conv2d","max_pooling2d","batch_normalization","dense","dropout"],newBranch)
        funcModel = branch.add(x,["dense","dense_2","dense_3"],branch.newBranch,exact=True)
        funcModel.summary()
        if saveName == "":
            funcModel.save("models/{}_branched.hdf5".format(modelName))
        else: 
            funcModel.save("models/{}".format(saveName))
        dataset = prepare.dataset(tf.keras.datasets.cifar10.load_data(),32,5000,22500,(227,227))

        funcModel = brevis.models.trainModelTransfer(funcModel,dataset,epocs = numEpocs, transfer = transfer, saveName = saveName)
        if saveName == "":
            funcModel.save("models/{}_branched.hdf5".format(modelName))
        else: 
            funcModel.save("models/{}".format(saveName))

        # x = keras.Model(inputs=x.inputs, outputs=x.outputs, name="{}_normal".format(x.name))
        return x
    def Run_train_model( model_name, dataset=None, numEpocs =2):
        """ generic training function. takes a model or model name and trains the model on the dataset for the specificed epocs.
        """
        x=None
        modelDetails = None
        print(model_name)
        if type(model_name) == type(""):
            #if the model_name is a valid filepath:
            if os.path.isfile(model_name):
                try:
                    x = tf.keras.models.load_model(model_name)
                except Exception as e:
                    print(e)
                    print("model {} could not be loaded".format(model_name))
            else:
                modelDetails = BranchingDnn.checkModelName(model_name) 
                print(modelDetails)
                #load newest version of model of known type.
                try:
                    x = tf.keras.models.load_model(newestModelPath(modelDetails["name"]))
                except Exception as e:
                    print(e)
                    print("could not load the newest model of known type: {}".format(modelDetails["name"]))
                    raise


        # elif type(model_name) == type(Model):
        x = model_name                        

        x.summary()

        #load dataset
        if dataset ==None:
            #check name of model, if recognized against known models, use the default dataset.
            modelDetails = BranchingDnn.checkModelName(model_name)
            if modelDetails:
                try:
                    dataset = modelDetails["dataset"](BranchingDnn)
                except expression as identifier:
                    print("dataset was not able to be identified")
            else: 
                print("model doesn't match any known models, dataset could not be loaded.")
                raise
            #else 


        #run training
        dataset = prepare.dataset(tf.keras.datasets.cifar10.load_data(),32,5000,22500,(227,227))

        x = brevis.models.trainModelTransfer(x,dataset,epocs = numEpocs, save = False)


        return x
    
    def eval_branches( model, dataset, count = 1, options="accuracy"):
        """ evaulate func for checking how well a branched model is performing.
            function may be moved to eval_model.py in the future.
        """ 
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels

        (train_images, train_labels), (test_images, test_labels) = dataset
        
        print("ALEXNET {}".format(BranchingDnn.ALEXNET))
        if BranchingDnn.ALEXNET:
            train_ds, test_ds, validation_ds = prepare.prepareAlexNetDataset(dataset,32)                       
        else: 
            test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
            test_ds_size = len(list(test_ds))
            test_ds =  (test_ds
                .batch(batch_size=64, drop_remainder=True))
        
        if BranchingDnn.ALEXNET: 
            model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        else:
            model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(),metrics=["accuracy"])

        run_logdir = get_run_logdir(model.name)
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir +"/eval")

        if options == "accuracy":
            test_scores = model.evaluate(test_ds, verbose=2)
            printTestScores(test_scores,num_outputs)
        elif options == "entropy":
            if BranchingDnn.ALEXNET:
                train_ds, test_ds, validation_ds = prepare.prepareAlexNetDataset(dataset,1)                       
            else: 
                test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
                test_ds_size = len(list(test_ds))
                test_ds =  (test_ds
                    # .map(augment_images)
                    # .shuffle(buffer_size=int(test_ds_size))
                    .batch(batch_size=1, drop_remainder=True))
        
            iterator = iter(test_ds)
            item = iterator.get_next()
            results = model.predict(item[0])

            for output in results:
                for result in output: 
                    print(result)
                    Pclass = np.argmax(result)
                    print("predicted class:{}, actual class: {}".format(Pclass, item[1]))

                    for softmax in result:
                        entropy = calcEntropy(softmax)
                        print("entropy:{}".format(entropy))
            print("answer: {}".format(item[1]))
            # results = calcEntropy(y_hat)
            # print(results)
            pass
        elif options == "throughput":
            if BranchingDnn.ALEXNET:
                train_ds, test_ds, validation_ds = prepare.prepareAlexNetDataset(dataset,1)                       
            else: 
                test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
                test_ds_size = len(list(test_ds))
                test_ds =  (test_ds
                    # .map(augment_images)
                    # .shuffle(buffer_size=int(test_ds_size))
                    .batch(batch_size=1, drop_remainder=True))
        
            iterator = iter(test_ds)
            item = iterator.get_next()
            pred=[]
            labels=[]
            for i in range(len(test_ds)):
                pred.append(model.predict(item[0]))
                labels.append(item[1])
            
            results = throughputMatrix(pred, labels, numOutput)
            print(results)
            print(pd.DataFrame(results).T)
            pass


        return 

    def predict( model, dataset, thresholds=[]):
        """ run the model using the provided confidence thresholds            
        """ 
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels

        train_ds, test_ds, validation_ds = prepare.prepareAlexNetDataset(dataset,32)                       
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])

        run_logdir = get_run_logdir(model.name)
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir +"/eval")
        predictions = []
        labels = []
        labelClasses = [0,1,2,3,4,5,6,7,8,9]



        """ For the first version testing purpose of this function, the whole model is run for each 
            input item. in future versions the model will be run sequentially from branch to branch, exiting run when 
            an accepted confidence score per item is achieved.

        """

        iterator = iter(test_ds)
        for j in range(len(test_ds)):
        # for j in range(10):
            # print("prediction: {} of {}".format(j,len(test_ds)),end='\r')
            item = iterator.get_next()
            prediction = model.predict(item[0])
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist()) #put a copy of the label (second element in the item tuple) in the labels list.
        labels = [expandlabels(x,num_outputs)for x in labels]
        predEntropy =[]
        predClasses =[]
        print("predictions complete, analyizing")


        for i,output in enumerate(predictions):
            for k, pred in enumerate(output):
                pred_classes=[]
                pred_entropy = []
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass) 
                    pred_entropy.append(calcEntropy(branch[0]))                       
                predClasses.append(pred_classes)
                predEntropy.append(pred_entropy)
        
        results = np.equal(predictions, labels)
        labels = np.array(labels)
        transpose_results = np.transpose(results) #truths
        transpose_labels = np.transpose(labels)

        mAcc = exitAccuracy(transpose_results[0],transpose_labels[0], labelClasses)
        # results = eval.KneeGraphClasses(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
        return

    def old_entropyMatrix( model, dataset):
        """
            calculate the entropy of the branches for the test set.
        """
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        (train_images, train_labels), (test_images, test_labels) = dataset
        
        
        print("ALEXNET {}".format(BranchingDnn.ALEXNET))
        if BranchingDnn.ALEXNET:
            train_ds, test_ds, validation_ds = prepare.prepareAlexNetDataset(dataset,1)
        else: 
            test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
            test_ds_size = len(list(test_ds))
            test_ds =  (test_ds
                .batch(batch_size=1, drop_remainder=True))
        
        predictions = []
        labels = []
        if BranchingDnn.ALEXNET: 
            model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
            iterator = iter(test_ds)
            indices = []
            for j in range(5):
                item = iterator.get_next()
                # for j in [33,2,3,4]:
                # img = test_images[j].reshape(1,784)
                prediction = model.predict(item[0])
                # print(prediction)
                predictions.append(prediction)
                labels.append(item[1].numpy().tolist())
                # print(item[1])
            predClasses =[]
            predEntropy =[]
            labelClasses = []
            for i,output in enumerate(predictions):
                print(output)
                for k, pred in enumerate(output):
                    pred_classes=[]
                    pred_entropy = []
                    label_classes = []
                    # print(i,end='\r')
                    for l, branch in enumerate(pred):
                        print(l)
                        Pclass = np.argmax(branch[0])
                        pred_classes.append(Pclass)     
                        pred_entropy.append(calcEntropy(branch[0]))                   
                        print("{}, class {}".format(branch[0], Pclass))
                    predClasses.append(pred_classes)
                    predEntropy.append(pred_entropy)
                    # Pprob = exit[0][Pclass]
                    # print("image:{}, exit:{}, pred Class:{}, probability: {} actual Class:{}".format(j,i, Pclass,Pprob, item[1]))
                    # if Pclass != item[1]:
                        # indices.append(j)
                    # entropy = calcEntropy(elem)
                    # print("entropy:{}".format(entropy))                
            # labels = [item for sublist in labels for item in sublist]
           
            labels = list(map(expandlabels,labels))

            print(predClasses)
            print(labels)
            print(predEntropy)
            # matrix = []cmd
            # for i, prediction in enumerate(predClasses):
            #     for j, branch in enumerate(prediction):
            #         if branch == labels[i]:

            #             matrix.append()


        return 
    
    def evalModel(model,dataset,suffix="", validation=True):
        train_ds, test_ds, validation_ds = prepare.dataset(dataset,32,5000,22500,(227,227))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        history = model.evaluate(validation_ds, verbose=2)
        history = model.evaluate(test_ds, verbose=2)

        return history


    def GetResultsCSV(model,dataset,suffix="", validation=True):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        # if BranchingDnn.ALEXNET:
            # train_ds, test_ds, validation_ds = prepare.prepareAlexNetDataset_old(1)
        # else:
            # train_ds, test_ds, validation_ds = prepare.prepareMnistDataset(dataset,1)
        # train_ds, test_ds, validation_ds = prepare.dataset(dataset,1,5000,22500,(227,227))
        train_ds, test_ds, validation_ds = dataset

        predictions = []
        labels = []
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])

        # test_scores = model.evaluate(test_ds, verbose=2)
        # print(test_scores)

        iterator = iter(test_ds)
        print(len(test_ds))

        
        # for j in range(len(test_ds)-1):
        for j in range(10):

            print("prediction: {} of {}".format(j,len(test_ds)),end='\r')

            item = iterator.get_next()
            prediction = model.predict(item[0])
            # print("predictions {}".format(prediction))
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist())
        # print("labels")
        # print(labels)
        # labels = np.argmax(labels)
        # print(labels)
        if BranchingDnn.ALEXNET:
            labels = np.argmax([expandlabels(x,num_outputs)for x in labels])
        else:
            for i, val in enumerate(labels):
                # print(i)
                labels[i]= [np.argmax(val)]* num_outputs

        predEntropy =[]
        predClasses =[]
        predRaw=[]
        print("predictions complete, analyizing") 
        for i,output in enumerate(predictions):
            for k, pred in enumerate(output):
                pred_classes=[]
                pred_entropy = []
                pred_Raw=[]
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    pred_Raw.append(branch[0])
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass) 
                    # if labels[i][0] == 0:
                        # print("class {}".format(Pclass))
                        # print("label {}".format(labels[i]))
                    # print(branch)
                    pred_entropy.append(branch[0])  
                    # print("entropy {}".format(pred_entropy))                     
                predRaw.append(pred_Raw)
                predClasses.append(pred_classes)
                predEntropy.append(pred_entropy)
                
        # print(predClasses)
        # print(predEntropy)
        # print(labels)
        # labels = list(map(expandlabels,labels,num_outputs))
        labelClasses = [0,1,2,3,4,5,6,7,8,9]
        predClasses = pd.DataFrame(predClasses)
        labels = pd.DataFrame(labels)
        predEntropy = pd.DataFrame(predEntropy)
        
        
        print("save to csv")
        PredRaw = pd.DataFrame(predRaw)
        PredRaw.to_csv("results/predRaw_temp.csv", sep=',', mode='w',index=False)

        predClasses.to_csv("results/predClasses{}.csv".format(suffix), sep=',', mode='w',index=False)
        labels.to_csv("results/labels{}.csv".format(suffix), sep=',', mode='w',index=False)
        predEntropy.to_csv("results/predEntropy{}.csv".format(suffix), sep=',', mode='w',index=False)


        # results = KneeGraph(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
        # results.to_csv("logs_entropy/{}_{}_entropyStats.csv".format(model.name,time.strftime("%Y%m%d_%H%M%S")), sep=',', mode='a')
        return

    
    def GetResultsCSV_evidence(model,dataset,suffix="", validation=True):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        train_ds, test_ds, validation_ds = (dataset)
        predictions = []
        labels = []
        #already compiled
        # model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        iterator = iter(test_ds)
        print(len(test_ds))

        # for j in range(len(test_ds)):
        for j in range(10):
            print("prediction: {} of {}".format(j,len(test_ds)),end='\r')
            item = iterator.get_next()
            prediction = model.predict(item[0])
            # print("predictions {}".format(prediction))
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist())
        print("labels")
        print(labels)
        if BranchingDnn.ALEXNET:
            labels = [expandlabels(x,num_outputs)for x in labels]
        else:
            for i, val in enumerate(labels):
                # print(i)
                labels[i]= [np.argmax(val)]* num_outputs

        predEvidence_fail =[]
        predEvidence_true = []

        predClasses =[]
        predRaw=[]
        # print("predictions", predictions)
        print("predictions complete, analyizing") 
        for i,output in enumerate(predictions):
            for k, pred in enumerate(output):
                pred_classes=[]
                pred_evidence = []
                pred_Raw=[]
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    
                    pred_Raw.append(branch[0])
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass) 
                    evidence = exp_evidence(branch[0])
                    total_evidence = tf.reduce_sum([evidence],1, keepdims=True) 
                    match = tf.reshape(tf.cast(tf.equal(Pclass, labels[i][l]), tf.float32),(-1,1))
                    if l ==0:
                        print("match", match.numpy())
                        print(total_evidence.numpy())
                    pred_evidence.append(total_evidence)
                predRaw.append(pred_Raw)
                predClasses.append(pred_classes)
                predEvidence_fail.append(pred_evidence)
                predEvidence_true.append(pred_evidence)
                
        labelClasses = [0,1,2,3,4,5,6,7,8,9]
        predClasses = pd.DataFrame(predClasses)
        labels = pd.DataFrame(labels)
        predEvidence_fail = pd.DataFrame(predEvidence_fail)
        predEvidence_true = pd.DataFrame(predEvidence_true)
        
        PredRaw = pd.DataFrame(predRaw)
        PredRaw.to_csv("results/predRaw_temp.csv", sep=',', mode='w',index=False)
        predClasses.to_csv("results/predClasses{}.csv".format(suffix), sep=',', mode='w',index=False)
        labels.to_csv("results/labels{}.csv".format(suffix), sep=',', mode='w',index=False)
        predEvidence_fail.to_csv("results/predEvidence_fail{}.csv".format(suffix), sep=',', mode='w',index=False)
        predEvidence_true.to_csv("results/predEvidence_true{}.csv".format(suffix), sep=',', mode='w',index=False)
        return

    def BranchKneeGraph(model,dataset):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        
        output_names = [i.name for i in model.outputs]
        train_ds, test_ds, validation_ds = prepare.prepareAlexNetDataset(dataset,1)
        
        predictions = []
        labels = []
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        iterator = iter(test_ds)
        indices = []
        for j in range(len(test_ds)):
        # for j in range(10):

            print("prediction: {} of {}".format(j,len(test_ds)),end='\r')

            item = iterator.get_next()
            prediction = model.predict(item[0])
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist())
        labels = [expandlabels(x,num_outputs)for x in labels]
        predEntropy =[]
        predClasses =[]
        print("predictions complete, analyizing")
        for i,output in enumerate(predictions):
            # print(output)
            for k, pred in enumerate(output):
                pred_classes=[]
                pred_entropy = []
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass) 
                    pred_entropy.append(calcEntropy(branch[0]))                       
                predClasses.append(pred_classes)
                predEntropy.append(pred_entropy)
                
        # print(predClasses)
        # print(predEntropy)
        # print(labels)
        # labels = list(map(expandlabels,labels,num_outputs))
        labelClasses = [0,1,2,3,4,5,6,7,8,9]
        results = eval.KneeGraph(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
        # f = open("logs_entropy/{}_{}_entropyStats.txt".format(model.name,time.strftime("%Y%m%d_%H%M%S")), "w")
        # f.write(json.dumps(results))
        results.to_csv("logs_entropy/{}_{}_entropyStats.csv".format(model.name,time.strftime("%Y%m%d_%H%M%S")), sep=',', mode='a')
        # f.close()
        # print(results)
        # print(pd.DataFrame(results).T)
        return

    def BranchKneeGraphClasses(model,dataset):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        
        output_names = [i.name for i in model.outputs]
        train_ds, test_ds, validation_ds = prepare.prepareAlexNetDataset(dataset,1)
        
        predictions = []
        labels = []
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        iterator = iter(test_ds)
        indices = []
        for j in range(len(test_ds)):
        # for j in range(10):
            print("prediction: {} of {}".format(j,len(test_ds)),end='\r')
            item = iterator.get_next()
            prediction = model.predict(item[0])
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist())
        labels = [expandlabels(x,num_outputs)for x in labels]
        predEntropy =[]
        predClasses =[]
        print("predictions complete, analyizing")
        for i,output in enumerate(predictions):
            # print(output)
            for k, pred in enumerate(output):
                pred_classes=[]
                pred_entropy = []
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass) 
                    pred_entropy.append(calcEntropy(branch[0]))                       
                predClasses.append(pred_classes)
                predEntropy.append(pred_entropy)
                
        # print(predClasses)
        # print(predEntropy)
        # print(labels)
        # labels = list(map(expandlabels,labels,num_outputs))
        labelClasses = [0,1,2,3,4,5,6,7,8,9]
        results = eval.KneeGraphClasses(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
        # f = open("logs_entropy/{}_{}_entropyStats.txt".format(model.name,time.strftime("%Y%m%d_%H%M%S")), "w")
        # f.write(json.dumps(results))
        results.to_csv("logs_entropy/{}_{}_PredictedClassesStats.csv".format(model.name,time.strftime("%Y%m%d_%H%M%S")), sep=',', mode='a')
        # f.close()
        # print(results)
        # print(pd.DataFrame(results).T)
        return

    def BranchKneeGraphPredictedClasses(model,dataset):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        
        output_names = [i.name for i in model.outputs]
        (train_images, train_labels), (test_images, test_labels) = dataset
        train_ds, test_ds, validation_ds = prepare.prepareAlexNetDataset(dataset,1)
        
        predictions = []
        labels = []
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        iterator = iter(test_ds)
        indices = []
        # for j in range(len(test_ds)):
        for j in range(100):
            print("prediction: {} of {}".format(j,len(test_ds)),end='\r')
            item = iterator.get_next()
            prediction = model.predict(item[0])
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist())
        labels = [expandlabels(x,num_outputs)for x in labels]
        predEntropy =[]
        predClasses =[]
        print("predictions complete, analyizing")
        for i,output in enumerate(predictions):
            # print(output)
            for k, pred in enumerate(output):
                pred_classes=[]
                pred_entropy = []
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass) 
                    pred_entropy.append(calcEntropy(branch[0]))                       
                predClasses.append(pred_classes)
                predEntropy.append(pred_entropy)
                
        # print(predClasses)
        # print(predEntropy)
        # print(labels)
        # labels = list(map(expandlabels,labels,num_outputs))
        labelClasses = [0,1,2,3,4,5,6,7,8,9]
        results = eval.KneeGraphPredictedClasses(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
        # f = open("logs_entropy/{}_{}_entropyStats.txt".format(model.name,time.strftime("%Y%m%d_%H%M%S")), "w")
        # f.write(json.dumps(results))
        # results.to_csv("logs_entropy/{}_{}_entropyClassesStats.csv".format(model.name,time.strftime("%Y%m%d_%H%M%S")), sep=',', mode='a')
        # f.close()
        # print(results)
        # print(pd.DataFrame(results).T)
        return


    def BranchEntropyConfusionMatrix( model, dataset):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        
        (train_images, train_labels), (test_images, test_labels) = dataset
        train_ds, test_ds, validation_ds = prepare.prepareAlexNetDataset(dataset,1)
        
        predictions = []
        labels = []
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        iterator = iter(test_ds)
        indices = []
        for j in range(len(test_ds)):
        # for j in range(5):
            print("prediction: {} of {}".format(j,len(test_ds)),end='\r')
            item = iterator.get_next()
            prediction = model.predict(item[0])
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist())
        labels = [expandlabels(x,num_outputs)for x in labels]
        predEntropy =[]
        predClasses =[]
        print("predictions complete, analyizing")
        for i,output in enumerate(predictions):
            # print(output)
            for k, pred in enumerate(output):
                pred_classes=[]
                pred_entropy = []
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass) 
                    pred_entropy.append(calcEntropy(branch[0]))                       
                predClasses.append(pred_classes)
                predEntropy.append(pred_entropy)
                
        # print(predClasses)
        # print(predEntropy)
        # print(labels)
        # labels = list(map(expandlabels,labels,num_outputs))
        labelClasses = [0,1,2,3,4,5,6,7,8,9]
        results = eval.entropyConfusionMatrix(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
        print(results)
        # print(pd.DataFrame(results).T)
        return

    def BranchEntropyMatrix( model, dataset):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        
        output_names = [i.name for i in model.outputs]
        (train_images, train_labels), (test_images, test_labels) = dataset
        train_ds, test_ds, validation_ds = prepare.prepareAlexNetDataset(dataset,1)
        
        predictions = []
        labels = []
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        iterator = iter(test_ds)
        indices = []
        # for j in range(len(test_ds)):
        for j in range(5):

            print("prediction: {} of {}".format(j,len(test_ds)),end='\r')

            item = iterator.get_next()
            prediction = model.predict(item[0])
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist())
        labels = [expandlabels(x,num_outputs)for x in labels]
        predEntropy =[]
        predClasses =[]
        print("predictions complete, analyizing")
        for i,output in enumerate(predictions):
            # print(output)
            for k, pred in enumerate(output):
                pred_classes=[]
                pred_entropy = []
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass) 
                    pred_entropy.append(calcEntropy(branch[0]))                       
                predClasses.append(pred_classes)
                predEntropy.append(pred_entropy)
                
        print(predClasses)
        # print(predEntropy)
        # print(labels)
        # labels = list(map(expandlabels,labels,num_outputs))
        labelClasses = [0,1,2,3,4,5,6,7,8,9]
        results = eval.entropyMatrix(predEntropy, labels, num_outputs,labelClasses,output_names)
        print(results)
        # print(pd.DataFrame(results).T)
        return

    def evalBranchMatrix( model, dataset):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        
        output_names = [i.name for i in model.outputs]
        train_ds, test_ds, validation_ds = prepare.prepareAlexNetDataset(dataset,1)
        
        predictions = []
        labels = []
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        iterator = iter(test_ds)
        indices = []
        # for j in range(len(test_ds)):
        for j in range(len(test_ds)):
            print("prediction: {} of {}".format(j,len(test_ds)),end='\r')
            item = iterator.get_next()
            prediction = model.predict(item[0])
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist())
        labels = [expandlabels(x,num_outputs)for x in labels]
        predClasses =[]
        print("predictions complete, analyizing")
        for i,output in enumerate(predictions):
            # print(output)
            for k, pred in enumerate(output):
                pred_classes=[]               
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    # print(l)
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass)     
                predClasses.append(pred_classes)
                


        # labels = list(map(expandlabels,labels,num_outputs))
        labelClasses = [0,1,2,3,4,5,6,7,8,9]
        results = eval.throughputMatrix(predClasses, labels, num_outputs,labelClasses,output_names)
        print(results)
        print(pd.DataFrame(results).T)
        return