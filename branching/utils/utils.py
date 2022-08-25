import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import itertools
import glob
import os
import pandas as pd
# from keras.models import load_model
# from keras.utils import CustomObjectScope
# from keras.initializers import glorot_uniform
import time
import pandas as pd
import math
import pydot
import os
import math
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
root_logdir = os.path.join(os.curdir, "logs\\fit\\")

#os.environ["PATH"] += os.pathsep + "C:\Program Files\Graphviz\bin"
#from tensorflow.keras.utils import plot_model

MODEL_DIR = "models/"

def expandlabels(label,num_outputs):
    flattened = [val for sublist in label for val in sublist]
    label = flattened * num_outputs
    return label
    
def fullprint(*args, **kwargs):
        from pprint import pprint
        import numpy
        opt = numpy.get_printoptions()
        numpy.set_printoptions(threshold=numpy.inf)
        pprint(*args, **kwargs)
        numpy.set_printoptions(**opt)



def calcEntropy(y_hat):
        #entropy is the sum of y * log(y) for all possible labels.
        if isinstance(y_hat, list):
            y_hat = np.array(y_hat)
        sum_entropy = 0
        if y_hat.ndim >1:
            return list(map(calcEntropy,y_hat))
        for i in range(len(y_hat)):
            if y_hat[i] != 0: # log of zero is undefined, see MacKay's book "Information Theory, Inference, and Learning Algorithms"  for more info on this workaround reasoning.
                entropy =y_hat[i] * math.log(y_hat[i],2)
                sum_entropy +=  entropy

        return -sum_entropy

def Entropy_raw(y_hat):
        #entropy is the sum of y * log(y) for all possible labels.
        # if isinstance(y_hat, list):
        #     y_hat = np.array(y_hat)
        # sum_entropy = 0
        # if y_hat.ndim >1:
        #     return list(map(calcEntropy,y_hat))
        # for i in range(len(y_hat)):
        #     if y_hat[i] != 0: # log of zero is undefined, see MacKay's book "Information Theory, Inference, and Learning Algorithms"  for more info on this workaround reasoning.
        #         entropy =y_hat[i] * math.log(y_hat[i],2)
        #         sum_entropy +=  entropy

        return -sum_entropy

def calcEntropy_Tensors(y_hat):
        #entropy is the sum of y * log(y) for all possible labels.
        #log(0) is evaulated as NAN and then clipped to approaching zero
        #rank is used to reduce multi-dim arrays but leave alone 1d arrays.
        rank = tf.rank(y_hat)
        def calc_E(y_hat):
            results = tf.clip_by_value((tf.math.log(y_hat)/tf.math.log(tf.constant(2, dtype=y_hat.dtype))), -1e12, 1e12)
#             results = tf.clip_by_value(results, -1e12, 1e12)
#             print("res ", results)
            return tf.reduce_sum(y_hat * results)

        sumEntropies = (tf.map_fn(calc_E,tf.cast(y_hat,'float')))
        
        if rank == 1:
            sumEntropies = tf.reduce_sum(sumEntropies)
        return -sumEntropies

def calcEntropy_Tensors2(y_hat):
    #entropy is the sum of y * log(y) for all possible labels.
    #doesn't deal with cases of log(0)
    val = y_hat * tf.math.log(y_hat)/tf.math.log(tf.constant(2, dtype=y_hat.dtype))
    sumEntropies =  tf.reduce_sum(tf.boolean_mask(val,tf.math.is_finite(val)))
    return -sumEntropies

from scipy.special import (comb, chndtr, entr, rel_entr, xlogy, ive)
def entropy(pk, qk=None, base=None):
    #taken from branchynet github
    """Calculate the entropy of a distribution for given probability values.

    If only probabilities `pk` are given, the entropy is calculated as
    ``S = -sum(pk * log(pk), axis=0)``.

    If `qk` is not None, then compute the Kullback-Leibler divergence
    ``S = sum(pk * log(pk / qk), axis=0)``.

    This routine will normalize `pk` and `qk` if they don't sum to 1.

    Parameters
    ----------
    pk : sequence
        Defines the (discrete) distribution. ``pk[i]`` is the (possibly
        unnormalized) probability of event ``i``.
    qk : sequence, optional
        Sequence against which the relative entropy is computed. Should be in
        the same format as `pk`.
    base : float, optional
        The logarithmic base to use, defaults to ``e`` (natural logarithm).

    Returns
    -------
    S : float
        The calculated entropy.

    """
    pk = np.asarray(pk)
    print(pk)
    print(1.0*pk)
    print(np.sum(pk,axis=0))
    pk = 1.0*pk / np.sum(pk, axis=0) 
    print(pk)
    if qk is None:
        vec = entr(pk)
    else:
        qk = np.asarray(qk)
        if len(qk) != len(pk):
            raise ValueError("qk and pk must have same length.")
        qk = 1.0*qk / np.sum(qk, axis=0)
        vec = rel_entr(pk, qk)
    print(vec)
    S = np.sum(vec, axis=0)
    if base is not None:
        S /= math.log(base)
    return S



def saveModel(model,name,overwrite = True, includeDate= True, folder ="models", fileFormat = "hdf5"):
    from datetime import datetime
    import os
    now = datetime.now() # current date and time
    stringName =""
    date =""
    if not os.path.exists(folder):
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass
    try:
        if includeDate:
            date =now.strftime("%y-%m-%d_%H%M%S")

        stringName = "{}{}_{}.{}".format(folder+"\\",name,date,fileFormat)
        model.save(stringName, save_format="fileFormat")
        print("saved Model:{}".format(stringName))
    except OSError:
        pass

    return stringName

def printTestScores(test_scores,num_outputs):
    print("overall loss: {}".format(test_scores[0]))
    if num_outputs > 1:
        for i in range(num_outputs):
            print("Output {}: Test loss: {}, Test accuracy {}".format(i, test_scores[i+1], test_scores[i+1+num_outputs]))
    else:
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])


#https://github.com/keras-team/keras/issues/341
def reset_model_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
            reset_weights(layer) #apply function recursively
            continue

        #where are the initializers?
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key: #is this item an initializer?
                  continue #if no, skip it

            # find the corresponding variable, like the kernel or the bias
            if key == 'recurrent_initializer': #special case check
                var = getattr(init_container, 'recurrent_kernel')
            else:
                var = getattr(init_container, key.replace("_initializer", ""))

            var.assign(initializer(var.shape, var.dtype))
            #use the initializer

def reset_layer_weights(layer):
    """ reset the weights for a specific layer.
    """

    #where are the initializers?
    if hasattr(layer, 'cell'):
        init_container = layer.cell
    else:
        init_container = layer

    for key, initializer in init_container.__dict__.items():
        if "initializer" not in key: #is this item an initializer?
                continue #if no, skip it

        # find the corresponding variable, like the kernel or the bias
        if key == 'recurrent_initializer': #special case check
            var = getattr(init_container, 'recurrent_kernel')
        else:
            var = getattr(init_container, key.replace("_initializer", ""))

        var.assign(initializer(var.shape, var.dtype))
        #use the initializer

def reset_branch_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
            reset_branch_weights(layer) #apply function recursively
            continue
        if "branch" in layer.name:
            print("reseting weights for {}".format(layer.name))
             #where are the initializers?
            if hasattr(layer, 'cell'):
                init_container = layer.cell
            else:
                init_container = layer

            for key, initializer in init_container.__dict__.items():
                if "initializer" not in key: #is this item an initializer?
                    continue #if no, skip it

                # find the corresponding variable, like the kernel or the bias
                if key == 'recurrent_initializer': #special case check
                    var = getattr(init_container, 'recurrent_kernel')
                else:
                    var = getattr(init_container, key.replace("_initializer", ""))

                var.assign(initializer(var.shape, var.dtype))
                #use the initializer
        else: 
            pass


def newestModelPath(modelNames):
    """ returns the path of the newest model with modelname in the filename.
        does not check the actual model name allocated in the file.
        janky, but works
    """
    # if modelNames is not list:
        # modelNames = [modelNames]
    full_list = os.scandir(MODEL_DIR)
    # print(modelNames)
    # print(full_list)
    items = []
    for i in full_list:
        if modelNames in i.name:
            # print(i.name)
            # print(os.path.getmtime(i.path))
            items.append(i)
    
    items.sort(key=lambda x: os.path.getmtime(x.path), reverse=True)
    result = items[0].path

    return result


def augment_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    # image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label
    
def resize(image):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image

def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)

def exitAccuracy(results, labels, classes=[]):
    """ find the accuracy scores of the main network exit for each class
            if classes is empty, return the average accuracy for all labels
    """
    classAcc = {}
    for i, labelClass in enumerate(classes):
        classAcc[labelClass] = results[np.where(labels==labelClass)].sum()/len(labels[np.where(labels == labelClass)])
    return classAcc

class ConfusionMatrixMetric(tf.keras.metrics.Metric):

            
    def update_state(self, y_true, y_pred,sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true,y_pred))
        return self.total_cm
        
    def result(self):
        return self.process_confusion_matrix()
    
    def confusion_matrix(self,y_true, y_pred):
        """
        Make a confusion matrix
        """
        y_pred=tf.argmax(y_pred,1)
        cm=tf.math.confusion_matrix(y_true,y_pred,dtype=tf.float32,num_classes=self.num_classes)
        return cm
    
    def process_confusion_matrix(self):
        "returns precision, recall and f1 along with overall accuracy"
        cm=self.total_cm
        diag_part=tf.linalg.diag_part(cm)
        precision=diag_part/(tf.reduce_sum(cm,0)+tf.constant(1e-15))
        recall=diag_part/(tf.reduce_sum(cm,1)+tf.constant(1e-15))
        f1=2*precision*recall/(precision+recall+tf.constant(1e-15))
        return precision,recall,f1


class  confidenceConditional(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        # Initialise as normal and add flag variable for when to run computation
        super(confidenceConditional, self).__init__(**kwargs)
        self.metric_variable = self.add_weight(name='metric_variable', initializer='zeros')
        self.update_metric = tf.Variable(False)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Use conditional to determine if computation is done
        if self.update_metric:
            # run computation
            computation_result = confidenceScore_numpy(y_true,y_pred)
            self.metric_variable.assign_add(computation_result)

    def result(self):
        return self.metric_variable

    def reset_states(self):
        self.metric_variable.assign(0.)

class  unconfidenceConditional(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        # Initialise as normal and add flag variable for when to run computation
        super(unconfidenceConditional, self).__init__(**kwargs)
        self.metric_variable = self.add_weight(name='metric_variable', initializer='zeros')
        self.update_metric = tf.Variable(False)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Use conditional to determine if computation is done
        if self.update_metric:
            # run computation
            computation_result = unconfidence(y_true,y_pred)
            self.metric_variable.assign_add(computation_result)

    def result(self):
        return self.metric_variable

    def reset_states(self):
        self.metric_variable.assign(0.)

class  confidenceDifference(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        # Initialise as normal and add flag variable for when to run computation
        super(confidenceDifference, self).__init__(**kwargs)
        self.metric_variable = self.add_weight(name='metric_variable', initializer='zeros')
        self.update_metric = tf.Variable(False)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Use conditional to determine if computation is done
        if self.update_metric:
            # run computation
            computation_result = unconfidence(y_true,y_pred)
            self.metric_variable.assign_add(computation_result)

    def result(self):
        return self.metric_variable

    def reset_states(self):
        self.metric_variable.assign(0.)

class ToggleMetrics(tf.keras.callbacks.Callback):
    '''On test begin (i.e. when evaluate() is called or 
     validation data is run during fit()) toggle metric flag '''
    def on_test_begin(self, logs):
        for metric in self.model.metrics:
            if 'confidenceConditional' or 'unconfidenceConditional' in metric.name:
                metric.on.assign(True)
    def on_test_end(self,  logs):
        for metric in self.model.metrics:
            if 'confidenceConditional' or 'unconfidenceConditional' in metric.name:
                metric.on.assign(False)


def confidenceScore_numpy(y_true, y_pred):
    # Numpy confidence metric version
    y_true =np.array(y_true)
    y_pred = np.array(y_pred)
    def argmax(x):
        return [np.argmax(x)]
    pred_labels = list(map(argmax,np.array(y_pred)))
    x = np.where(np.equal(y_true,pred_labels) ==True)
    y = y_pred[x[0]]
    results = calcEntropy(y)
    # print(results)
    if not results:
        return 1e-8
    else:
        return np.median(results)

def confidenceScore(y_true, y_pred):
    # print(y_pred)
    # print(tf.keras.backend.get_value(y_pred))
    
    pred_labels = tf.math.argmax(y_pred,1)
    indexes = tf.where(tf.math.equal(pred_labels, tf.cast(tf.reshape(y_true,pred_labels.shape),'int64')))
    indexes = tf.reshape(indexes,[-1])
    entropies = tf.gather(y_pred,indexes)
    if tf.equal(tf.size(entropies), 0):
        correctEntropies = tf.cast(1e-8,'float')
    else:
        correctEntropies = calcEntropy_Tensors2(entropies)

    return correctEntropies


def unconfidence(y_true, y_pred):
        #avg confidence of incorrect items.
        # print(y_pred)
        # print(tf.keras.backend.get_value(y_pred))
        
        pred_labels = tf.math.argmax(y_pred,1)
        indexes = tf.where(tf.math.not_equal(pred_labels, tf.cast(tf.reshape(y_true,pred_labels.shape),'int64')))
        indexes = tf.reshape(indexes,[-1])
        entropies = tf.gather(y_pred,indexes)
        if tf.equal(tf.size(entropies), 0):
            incorrectEntropies = tf.cast(1e-8,'float')
        else:
            incorrectEntropies = calcEntropy_Tensors2(entropies)
        
        return incorrectEntropies


def confidenceDifference(y_true, y_pred):
        #difference of the average confidence of correct/incorrect predictions.
        # print(y_pred)
        # print(tf.keras.backend.get_value(y_pred))
        
        pred_labels = tf.math.argmax(y_pred,1)

        correct_indexes = tf.where(tf.math.equal(pred_labels, tf.cast(tf.reshape(y_true,pred_labels.shape),'int64')))
        correct_indexes = tf.reshape(correct_indexes,[-1])
        correct_entropies = tf.gather(y_pred,correct_indexes)


        incorrect_indexes = tf.where(tf.math.not_equal(pred_labels, tf.cast(tf.reshape(y_true,pred_labels.shape),'int64')))
        incorrect_indexes = tf.reshape(incorrect_indexes,[-1])
        incorrect_entropies = tf.gather(y_pred,incorrect_indexes)
        
        if tf.equal(tf.size(correct_entropies), 0):
            correctEntropies = tf.cast(1e-8,'float')
        else:
            correctEntropies = calcEntropy_Tensors2(correct_entropies)
        
        
        if tf.equal(tf.size(incorrect_entropies), 0):
            incorrectEntropies = tf.cast(1e-8,'float')
        else:
            incorrectEntropies = calcEntropy_Tensors2(incorrect_entropies)
        
        return tf.abs(incorrectEntropies - correctEntropies)


class EntropyConfidenceMetric(tf.keras.metrics.Metric):
    #metric of average confidence for correct answers          
    def update_state(self, y_true, y_pred,sample_weight=None):
        self.confidenceScore(y_true,y_pred)

        return self.AvgConfidence
        
    def confidenceScore(self, y_true, y_pred):
        self.AvgConfidence = -1
        pred_label = list(map(np.argmax,np.array(y_pred)))
        countCorrect=0
        for i in range(len(y_pred)):
            if pred_label[i] == y_true[i]:
                countCorrect += 1
                self.AvgConfidence += calcEntropy_Tensors(y_pred[i])
        
        if countCorrect == 0: #hack so i don't divide by zero
            countCorrect = 1

        self.AvgConfidence = self.AvgConfidence/countCorrect

def crossE_test(y_true, y_pred):
    crossE = tf.keras.losses.SparseCategoricalCrossentropy()
    scce = tf.math.add(crossE(y_true, y_pred), y_true.shape[-1])
    # print("crossE",scce)
    return scce

######################################
# functions from DST paper
# This function to generate evidence is used for the first example
def relu_evidence(logits):
    return tf.nn.relu(logits)

# This one usually works better and used for the second and third examples
# For general settings and different datasets, you may try this one first
def exp_evidence(logits): 
    return tf.exp(tf.clip_by_value(logits,-10,10))

# This one is another alternative and 
# usually behaves better than the relu_evidence 
def softplus_evidence(logits):
    return tf.nn.softplus(logits)

#custom KL divergence fucnction
def KL(alpha,K):
    # print("K:",K)
    beta=tf.constant(np.ones((1,K)),dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha,axis=1,keepdims=True)
    S_beta = tf.reduce_sum(beta,axis=1,keepdims=True)
    lnB = tf.compat.v1.lgamma(S_alpha) - tf.reduce_sum(tf.compat.v1.lgamma(alpha),axis=1,keepdims=True)
    lnB_uni = tf.reduce_sum(tf.compat.v1.lgamma(beta),axis=1,keepdims=True) - tf.compat.v1.lgamma(S_beta)
    
    dg0 = tf.compat.v1.digamma(S_alpha)
    dg1 = tf.compat.v1.digamma(alpha)
    # tf.print("alpha",alpha.shape)
    # tf.print("beta",beta.shape)
    kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keepdims=True) + lnB + lnB_uni
    # print("kl", kl)
    return kl

# def loss_function():
#     #create a wrapper function that returns a function
#     kl = tf.keras.losses.KLDivergence()
#     temperature = 1
#     Classes = 10
#     crossE = tf.keras.losses.CategoricalCrossentropy()
#     mse =mse_loss
#     global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
#     annealing_step = 320
#     def EMSE_Loss(y_true, y_pred):
# #         softmax = tf.nn.softmax(y_pred)
#         evidence = exp_evidence(y_pred)
#         alpha  = evidence + 1
        
#         u = Classes / tf.reduce_sum(alpha, axis=1) #uncertainty
        
#         prob = alpha/tf.reduce_sum(alpha, 1,keepdims=True) 
        
#         loss = tf.reduce_mean(mse(y_true, alpha,global_step,annealing_step))
# #         l2_loss = (tf.nn.l2_loss(W3)+tf.nn.l2_loss(W4)) * lmb
# #         l2_loss = tf.reduce_sum(tf.square(inputs - teaching_features))
        
#         final_loss = loss #+ l2_loss
        
# #         kl_loss = kl( tf.nn.softmax(softmax / self.temperature, axis = 1 ),
        
#         return evidence, final_loss
#     return  EMSE_Loss
class AnnealingCallback(keras.callbacks.Callback):
    def __init__(self, annealing_point, verbose=1, **kwargs):
        #annealing_point is the point when the annealing temperature is at max. this is given as a value in terms of batches.
        #at the start of X batches, the temperature will be at max
        #temperature is checked at the start of each batch
        
        self.annealing_point = annealing_point
        self.step_counter = 0
        self.verbose = verbose
        return None
    def on_train_begin(self, logs=None):
        #initialize the annealing at training start 
        
        ### if the annealing_point is 0, then start the full temperature immediately (1).
        if annealing_point == 0:
            self.annealing_rate = 1            
        else:
            self.annealing_rate = 0
        
        self.model.loss = loss_function(self.annealing_point) 
        if self.verbose==2:
            print("Starting training; Loss: {}".format(self.model.loss))
        
    def on_train_batch_begin(self, batch, logs=None):
        self.step_counter = self.step_counter + 1
        self.annealing_rate = tf.minimum(1.0, tf.cast(self.step_counter/self.annealing_point,tf.float32))
        self.model.loss = loss_function(self.annealing_point)
        if self.verbose==2:
            print("...Training: step: {} start of batch {}; annealing_rate = {}".format(self.step_counter, batch, self.annealing_rate))



def evidence_crossentropy(annealing_rate=1, momentum=1, decay=1, global_loss=False,num_outputs=10):
    #create a wrapper function that returns a function
    temperature = 1
    Classes = 10
    keras_kl = tf.keras.losses.KLDivergence()
    annealing_rate = annealing_rate
    momentum_rate = momentum
    decay_rate = decay
    def cross_entropy_evidence(labels, outputs): 
        softmax = tf.nn.softmax(outputs)
        # activated_outputs =tf.keras.activations.sigmoid(softmax)
        evidence = softplus_evidence(outputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True) 
        E = alpha - 1
        m = alpha / S
        A = tf.reduce_sum((labels-m)**2, axis=1, keepdims=True) 
        B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True) 
        # tf.print("A+B",(A+B).shape, tf.reduce_sum(A+B, axis=1))
        annealing_coef = tf.minimum(1.0,tf.cast(annealing_rate,tf.float32))
#         annealing_coef = 1
        alp = E*(1-labels) + 1 
        # print("alp", alp)
        C =  annealing_coef * KL(alp,num_outputs)
        # print("c",C)
        # C = keras_kl(labels,evidence)
        loss = tf.keras.losses.categorical_crossentropy(labels, softmax)
        # tf.print("loss",loss.shape,loss)
        pred = tf.argmax(outputs,1)
        truth = tf.argmax(labels,1)
        match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
        return loss + C
        # return (A + B) + C
    return  cross_entropy_evidence




''' Old evidence based loss, no longer used'''
def evidence_loss(annealing_rate=1, momentum=1, decay=1, evidence_function=softplus_evidence, sparse=True):
    #create a wrapper function that returns a function
    temperature = 1
    Classes = 10
    keras_kl = tf.keras.losses.KLDivergence()
    annealing_rate = annealing_rate
    momentum_rate = momentum
    decay_rate = decay
    # evidence_function = evidence_function

    def sparse_mse_loss_global(labels, outputs):
        labels = tf.one_hot(tf.cast(labels, tf.int32), 10)
#         print("onehot",labels)
        labels = tf.cast(labels, dtype=tf.float32)
        try:
            labels= tf.squeeze(labels,[1])
        except:
                print("loss labels can't be squeezed")
        # labels = tf.keras.utils.to_categorical(labels,10) #TODO change 10 to the number of inputs.
        evidence = evidence_function(outputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True) 
        E = alpha - 1
        m = alpha / S

        A = tf.reduce_sum((labels-m)**2, axis=1, keepdims=True) 
        B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True) 

        annealing_coef = tf.minimum(1.0,tf.cast(annealing_rate,tf.float32))
#         annealing_coef = 1
        alp = E*(1-labels) + 1 
        # print("alp", alp)
        C =  annealing_coef * KL(alp)
#         print(alpha)
#         C = keras_kl(labels, alpha)
        return (A + B) + C

    def mse_loss_global(labels, outputs): 
        
        evidence = evidence_function(outputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True) 
        E = alpha - 1
        m = alpha / S

        A = tf.reduce_sum((labels-m)**2, axis=1, keepdims=True) 
        B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True) 

        annealing_coef = tf.minimum(1.0,tf.cast(annealing_rate,tf.float32))
#         annealing_coef = 1
        alp = E*(1-labels) + 1 
        # print("alp", alp)
        C =  annealing_coef * KL(alp)
#         print(alpha)
#         C = keras_kl(labels, alpha)
        return (A + B) + C
        
    if sparse == True:
        return  sparse_mse_loss_global
    else:
        return  mse_loss_global
####################################################

#old version
# def evidence_loss(classes = 10, temperature=1, global_step=None):
#     #create a wrapper function that returns a function
#     kl = tf.keras.losses.KLDivergence()
#     temperature = temperature
#     Classes = classes
#     # K =10
#     crossE = tf.keras.losses.CategoricalCrossentropy()
#     def relu_evidence(logits):
#         return tf.nn.relu(logits)

#     def exp_evidence(logits): 
#         return tf.exp(tf.clip_by_value(logits,-10,10))

#     def softplus_evidence(logits):
#         return tf.nn.softplus(logits)

#     def KL(alpha):
#         beta=tf.constant(np.ones((1,Classes)),dtype=tf.float32)
#         S_alpha = tf.reduce_sum(alpha,axis=1,keepdims=True)
#         S_beta = tf.reduce_sum(beta,axis=1,keepdims=True)
#         lnB = tf.compat.v1.lgamma(S_alpha) - tf.reduce_sum(tf.compat.v1.lgamma(alpha),axis=1,keepdims=True)
#         lnB_uni = tf.reduce_sum(tf.compat.v1.lgamma(beta),axis=1,keepdims=True) - tf.compat.v1.lgamma(S_beta)
        
#         dg0 = tf.compat.v1.digamma(S_alpha)
#         dg1 = tf.compat.v1.digamma(alpha)
        
#         kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keepdims=True) + lnB + lnB_uni
#         return kl
#     def mse_loss(p, alpha, global_step=global_step, annealing_step=320): 
#         S = tf.reduce_sum(alpha, axis=1, keepdims=True) 
#         E = alpha - 1
#         m = alpha / S
        
#         A = tf.reduce_sum((p-m)**2, axis=1, keepdims=True) 
#         B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True) 
#         # print("global",global_step)
#         # annealing_coef = tf.minimum(1.0,tf.cast(global_step/annealing_step,tf.float32))
        
#         alp = E*(1-p) + 1 

#         # C =  annealing_coef * KL(alp)
        
#         C = KL(alp)
#         print("KL", C)
#         return (A+B)+ C
    
#     global_step= global_step
#     annealing_step = 320
#     def EMSE_Loss(y_true, y_pred):
# #         softmax = tf.nn.softmax(y_pred)
#         evidence = exp_evidence(y_pred)
#         alpha  = evidence + 1
        
#         u = Classes / tf.reduce_sum(alpha, axis=1) #uncertainty
        
#         prob = alpha/tf.reduce_sum(alpha, 1,keepdims=True) 
        
#         loss = tf.reduce_mean(mse_loss(y_true, alpha,global_step,annealing_step))
# #         l2_loss = (tf.nn.l2_loss(W3)+tf.nn.l2_loss(W4)) * lmb
# #         l2_loss = tf.reduce_sum(tf.square(inputs - teaching_features))
        
#         final_loss = loss #+ l2_loss
        
# #         kl_loss = kl( tf.nn.softmax(softmax / self.temperature, axis = 1 ),
#         # print("evidence",evidence)
#         # print("final_loss",final_loss)
#         return evidence, final_loss
#     return  EMSE_Loss




def entropy_loss():
    #create a wrapper function that returns a function

    crossE = tf.keras.losses.SparseCategoricalCrossentropy()
    def entropyLoss(y_true, y_pred):

        pred_labels = tf.math.argmax(y_pred,1)
        indexes = tf.where(tf.math.equal(pred_labels, tf.cast(tf.reshape(y_true,pred_labels.shape),'int64')))
        indexes = tf.reshape(indexes,[-1])
        entropies = tf.gather(y_pred,indexes)
        correctEntropies = calcEntropy_Tensors2(entropies)
        scce = crossE(y_true, y_pred)
        # print("scce",scce)
        # print("loss",correctEntropies)
        loss = scce + (correctEntropies * scce) + 1e-8
        # loss = correctEntropies
        # loss = scce 
        # print(loss)
        return loss
    return entropyLoss
        

def entropyMultiplication(y_true, y_pred):
    crossE = tf.keras.losses.SparseCategoricalCrossentropy()
    #Entropy is added to the CrossE divided by the len of inputs
    pred_labels = tf.math.argmax(y_pred,1)
    indexes = tf.where(tf.math.equal(pred_labels, tf.cast(tf.reshape(y_true,pred_labels.shape),'int64')))
    indexes = tf.reshape(indexes,[-1])
    entropies = tf.gather(y_pred,indexes)
    if tf.equal(tf.size(entropies), 0):
        correctEntropies = tf.cast(0,'float')
    else:
        correctEntropies = tf.reduce_mean(tf.map_fn(calcEntropy_Tensors,tf.cast(entropies,'float')))
    scce = crossE(y_true, y_pred)
    #note: this may cause issues if no correct entropies are found.
    if correctEntropies == 0:
        correctEntropies = 1
    loss = scce + (correctEntropies * scce)
    return loss

def custom_loss_addition(y_true, y_pred):
    #Entropy is added to the CrossE divided by the len of inputs
    pred_label = list(map(np.argmax,np.array(y_pred)))
    crossE = tf.keras.losses.SparseCategoricalCrossentropy()
    sumEntropy = 0
    for i in range(len(y_pred)):
        # print("Entropy : ",calcEntropy(y_pred[i]))
        if pred_label[i] == y_true[i]:
            sumEntropy += calcEntropy(y_pred[i])
    sumEntropy = sumEntropy / len(y_pred)         
    loss = crossE(y_true, y_pred)
    
    loss +=sumEntropy
    return loss

def custom_loss_multi(y_true, y_pred):
    #CrossE is multiplied by the Entropy
    pred_label = list(map(np.argmax,np.array(y_pred)))
    crossE = tf.keras.losses.SparseCategoricalCrossentropy()
    sumLoss = 0
    
    for i in range(len(y_pred)):
        loss = crossE(y_true[i], y_pred[i])
#         print('crossE: ',loss)
        if pred_label[i] == y_true[i]:
#             print('calcEntropy ',calcEntropy(y_pred[i]))
            loss = loss * calcEntropy(y_pred[i])
        sumLoss += loss
    sumLoss = sumLoss / len(y_pred)         
    
#     loss = crossE(y_true, y_pred)
#     print("CrossE : ",loss.numpy())
#     print("Loss : ",sumLoss)
    return sumLoss

def get_run_logdir(name=""):
    run_id = time.strftime("run_{}_%Y_%m_%d-%H_%M_%S".format(name))
    return os.path.join(root_logdir, run_id)