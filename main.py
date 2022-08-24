# import the necessary packages
from brevis.models import train
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools
#add some code
# from keras.models import load_model
# from keras.utils import CustomObjectScope
# from keras.initializers import glorot_uniform

import math
import pydot
import os
#os.environ["PATH"] += os.pathsep + "C:\Program Files\Graphviz\bin"
#from tensorflow.keras.utils import plot_model
# from utils import *

# from Alexnet_kaggle_v2 import * 
import brevis as branching
from brevis.utils import * 

# ALEXNET = False
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
root_logdir = os.path.join(os.curdir, "logs\\fit\\")


# tf.debugging.experimental.enable_dump_debug_info("logs/", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)






if __name__ == "__main__":
    #### build alexnet model
    # x = branching.core.Run_alexNet_evidence( 20, modelName="alexNetv6_logits.hdf5", saveName = "alexNetv6_evidence_3_cross_3b",transfer = True ,customOptions="customLoss")
    # x = branching.core.Run_alexNet( 30, modelName="alexNetv6_logits.hdf5", saveName = "alexNetv6_entropy_dense",transfer = True ,customOptions="CrossE")
    dataset = branching.dataset.prepare.dataset(tf.keras.datasets.cifar10.load_data(),32,5000,22500,(227,227))
    
    ####### the original entropy model ####
    # brevis = (branching.core.branched_model(modelName="models/alexNetv6_logits.hdf5",saveName="alexNetv6_entropy_class",transfer=True,customOptions="")
    #         .add_branches(branching.branches.branch.newBranch_flatten,["max_pooling2d","max_pooling2d_1","dense"])
    #         .set_dataset(dataset)
    #         .train(30,True, loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9), customOptions="CrossE")
    #         )
            
    #### the new improved evidence model ####
    brevis = (branching.core.branched_model(modelName="models/alexNetv6_logits.hdf5",saveName="alexNetv6_entropy_class",transfer=True,customOptions="")
            .add_branches(branching.branches.branch.newBranch_flatten_evidence,["max_pooling2d","max_pooling2d_1","dense"])
            .set_dataset(dataset)
            .train(30,loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9), transfer=True, customOptions="CrossE")
            )


    # x = branching.core.Run_alexNet( 30, modelName="alexNetv6_logits.hdf5", saveName = "alexNetv6_entropy",transfer = True ,customOptions="CrossE")
    
    
    # x = branching.models.SelfDistilation.alexnet( 10, modelName="alexNetv6.hdf5",
    #                                          saveName = "alexNetv6_evidence",
    #                                          transfer = True,customOptions="customLoss")
    



    # student = tf.keras.models.load_model("models/alexNetv6.hdf5")
    # teacher = tf.keras.models.load_model("models/alexNetv6.hdf5")
    # y = branching.models.selfDistilation.SelfDistilation.normal_distillation(student, teacher,tf.keras.datasets.cifar10.load_data(),["branch_softmax"])
    # y = branching.models.selfDistilation.SelfDistilation.normal(student,tf.keras.datasets.cifar10.load_data(), ["branch_softmax"])

    # y = branching.core.evalModel(x, tf.keras.datasets.cifar10.load_data(),"natural")
  
    # x = tf.keras.models.load_model("models/alexNetv6_feat_distill_4.hdf5", custom_objects={'confidenceScore': confidenceScore,
    #                                                                      'unconfidence': unconfidence,
    #                                                                      'confidenceDifference': confidenceDifference,
    #                                                                      'BranchEndpoint': branching.branches.branch.BranchEndpoint,
    #                                                                      'FeatureDistillation': branching.branches.branch.FeatureDistillation,
    #                                                                      'FeatureDistillation_clear': branching.branches.branch.FeatureDistillation_clear})
    # y = branching.core.GetResultsCSV(x, tf.keras.datasets.cifar10.load_data(),"_distil_4")
  
    # x = tf.keras.models.load_model("models/alexNetv5_crossE_Eadd.hdf5")
    # y = branching.GetResultsCSV(x, tf.keras.datasets.cifar10.load_data(),"_crossE_Eadd")
    


    # x = branching.Run_alexNet( 10, modelName="alexNetv5.hdf5", saveName = "alexNetv5_customLoss_3",transfer = False, custom=True)

    


    # x = tf.keras.models.load_model("models/alexNetv5_customLoss_2.hdf5")
    # y = branching.GetResultsCSV(x, tf.keras.datasets.cifar10.load_data(),"custloss_2")

    # x = branching.Run_inceptionv3( 3, modelName="inception_finetuned.hdf5", saveName = "inception_branched",transfer = False)
    # x = branching.Run_resnet50v2( 3, modelName="resnet50_finetuned.hdf5", saveName = "resnet50_branched",transfer = False)

    # x = branching.Run_mnistNet( 5, modelName="mnistNormal.hdf5", saveName = "mnistNormal_branched",transfer = True)
    
    """
    Various model versions:
        alexNetv5 : up to date version of testing, trained using the augmented, not self standardized images. base for most other versions that I tried out
        
        models with alt in the name are models I made trying to track down what is going on with the missing 0 class from branches
        alexNetv5_alt6: model with branches on dense layers, this model actually lost a second class completely as well, class 1. 
    """

    # x = tf.keras.models.load_model("models/mnist_transfer_trained_21-01-04_125846.hdf5")
    # x.summary()
    # branching.eval_branches(x,branching.loadTrainingData(),1,"accuracy")
    # branching.eval_branches(x,branching.loadTrainingData(),1,"entropy")
    # branching.find_mistakes(x,branching.loadTrainingData(),1)
    
    # x = tf.keras.models.load_model("models/alexnet_branched_new_trained.hdf5")
    # x.summary()
    # branching.entropyMatrix(x,tf.keras.datasets.cifar10.load_data())


    # branching.eval_branches(x,tf.keras.datasets.cifar10.load_data(),1,"entropy")
    # branching.find_mistakes(x,tf.keras.datasets.cifar10.load_data(),1)



    # x.summary()
    # branching.eval_branches(x,tf.keras.datasets.cifar10.load_data(),1,)



    ####Make a new model
    # x = branching.Run_alexNet(50, saveName = "alexnext_branched_fullModel_trained",transfer = False)
    # x.summary()

    # branching.datasetStats(tf.keras.datasets.cifar10.load_data())

    # x = tf.keras.models.load_model("models/alexnext_branched_fullModel_trained_branched_branched.hdf5")
    # x.summary()
    # branching.eval_branches(x,tf.keras.datasets.cifar10.load_data())
    # """ 
    # x = tf.keras.models.load_model("models/alexnet_branch_pooling.hdf5")
    # x.summary()
    # branching.eval_branches(x,tf.keras.datasets.cifar10.load_data())
    # x = tf.keras.models.load_model("models/alexnet_branched_new_trained.hdf5")
    # x.summary()
    # branching.eval_branches(x,tf.keras.datasets.cifar10.load_data())
    # """
    # x = branching.Run_train_model(x,tf.keras.datasets.cifar10.load_data(),10)
    # x.save("models/alexnet_branched_new_trained.hdf5")

    # x = branching.Run_alexNet(1)

    # x = branching.mnistbranching()
    

    # x = branching.loadModel("models/mnist_trained_20-12-15_112434.hdf5")
    # x = tf.keras.models.load_model("models/mnist2_transfer_trained_.tf")

    # x.save("models/mnistNormal2_trained.hdf5")
    # saveModel(x,"mnist2_transfer_trained_final",includeDate=False)
    pass

