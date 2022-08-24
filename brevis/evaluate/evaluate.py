import multiprocessing
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import itertools
import glob
import os
import pandas as pd
import math
import pydot
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import matplotlib.cm as cm

from numpy import sqrt
from numpy import argmax
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, plot_precision_recall_curve

from brevis.utils import *
import brevis as branching
import brevis.core_v2 as brevis
from scipy.special import gammaln, digamma
from scipy.special import logsumexp


def calc_AUC(output_df,metrics=['energy'],plot=False):
    '''
    AUC calculation function for list of output dataframes
    returns a list of threshold for the gmean of each set of outputs.    
    '''
    lessThanMetrics = ["energy","uncert","entropy"]
    _thresholds = []
    y_test = np.int32(output_df['correct'])
    plots = []
    for metric in metrics:    
        lr_auc = roc_auc_score(y_test, output_df[metric])
        if metric in lessThanMetrics:
            pos_label = 0
        else:
            pos_label = 1
        fpr, tpr, thresholds = roc_curve(y_test, output_df[metric],pos_label=pos_label)
        gmeans = sqrt(tpr * (1-fpr))
        # print(gmeans)
        # locate the index of the largest g-mean
        ix = argmax(gmeans)
        threshold = thresholds[ix]
        print(metric," lr_auc",lr_auc, 'Best Threshold={}, G-Mean={}, TPR={}, FPR={}'.format(threshold, gmeans[ix],tpr[ix],fpr[ix]))
        _thresholds.append(threshold)
        # plot the roc curve for the model
        plots.append({"fpr":fpr,"tpr":tpr,"label":metric, "ix":ix})
    if plot:
        pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
        for plot in plots:
            ix = plot['ix']
            pyplot.plot(plot["fpr"], plot["tpr"],  label=plot['label'])

            pyplot.scatter(plot["fpr"][ix], plot["tpr"][ix], marker='o', color='black')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.title(metric)
        pyplot.legend()
        # show the plot
        pyplot.show()
    return _thresholds
def dirichlet_prior_network_uncertainty(logits, epsilon=1e-10, alpha_correction=True):
    """
    Calculate the Dirichlet prior network uncertainty.
    """
    logits = np.asarray(logits, dtype=np.float64)
    alphas = np.exp(logits)
    alphas = np.clip(alphas, 0, np.finfo(np.dtype("float32")).max)
    if alpha_correction:
        alphas = alphas + 1
    alpha0 = np.sum(alphas, axis=1, keepdims=True)
    probs = alphas / alpha0
    conf = np.max(probs, axis=1)
    entropy_of_exp = -np.sum(probs * np.log(probs + epsilon), axis=1)
    expected_entropy = -np.sum(
        (alphas / alpha0) * (digamma(alphas + 1) - digamma(alpha0 + 1.0)), axis=1
    )
    mutual_info = entropy_of_exp - expected_entropy
    epkl = np.squeeze((alphas.shape[1] - 1.0) / alpha0)
    dentropy = (
        np.sum(
            gammaln(alphas) - (alphas - 1.0) * (digamma(alphas) - digamma(alpha0)),
            axis=1,
            keepdims=True,
        )
        - gammaln(alpha0)
    )
    uncertainty = {
        "confidence_alea_uncert.": np.float32(np.squeeze(conf)),
        "entropy_of_expected": -np.squeeze(entropy_of_exp),
        "expected_entropy": -np.squeeze(expected_entropy),
        "mutual_information": -np.squeeze(mutual_info),
        "EPKL": -epkl,
        "differential_entropy": -np.squeeze(dentropy),
    }
    return uncertainty


def getPredictions_Energy(model, input_set, stopping_point=None,num_classes=10):
    '''
        Function for collecting the model's predictions on a test set. 

        Returns a list of DataFrames for each exit of the model.    
    '''
    num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
    print("outputs",num_outputs)
    #     train_ds, test_ds, validation_ds = (dataset)
    Results=[]
    Pred=[]
    Labels =[]
    Uncert = []
    Outputs = pd.DataFrame()
    Energy = []
    Energy_softmax = []
    Energy_evidence = []
    Energy_alpha = []
    Energy_Mass = []
    Entropy = []
    pAcc=[]
    calibration=[]

    conf=[]
    entropy_of_exp=[]
    expected_entropy=[]
    mutual_info=[]
    epkl=[]
    dentropy=[]
    
    for i in range(num_outputs):
        Results.append([])
        Pred.append([])
        Labels.append([])
        Uncert.append([])
        Energy.append([])
        Energy_softmax.append([])
        Energy_evidence.append([])
        Energy_alpha.append([])
        Energy_Mass.append([])
        Entropy.append([])
        pAcc.append([])
        calibration.append([])

        conf.append([])
        entropy_of_exp.append([])
        expected_entropy.append([])
        mutual_info.append([])
        epkl.append([])
        dentropy.append([])
        
    for i, (x,y) in enumerate(input_set):
        if stopping_point and i > stopping_point:
            break
        try:
            print("prediction: {} of {}".format(i,len(input_set)),end='\r')
        except:
            print("prediction: {}".format(i),end='\r')
            pass
        predictions = model.predict(x)
        if num_outputs > 1:
            _predictions = predictions[0]
        else:
            _predictions = [predictions]
        # print(_predictions)
        for k, outputs in enumerate(_predictions):
            
            # print("outputs ", k, outputs)
            for j, prediction in enumerate(outputs):
                dirch = dirichlet_prior_network_uncertainty([prediction])
                # print(dirch)
                conf[k].append(dirch["confidence_alea_uncert."])
                entropy_of_exp[k].append(dirch["entropy_of_expected"])
                expected_entropy[k].append(dirch["expected_entropy"])
                mutual_info[k].append(dirch["mutual_information"])
                epkl[k].append(dirch["EPKL"])
                dentropy[k].append(dirch["differential_entropy"])

                evidence =exp_evidence(prediction)
                alpha = evidence +1
                S = sum(alpha)
                E = alpha - 1
                Mass = alpha / S
                u = num_classes / S
                Uncert[k].append(u.numpy().mean())
                Results[k].append(np.argmax(prediction))
                Labels[k].append(np.argmax(y[j]))
                Energy[k].append( -(logsumexp(np.array(prediction))))
                Energy_softmax[k].append( -(1 * logsumexp(np.array(tf.nn.softmax(prediction))/1)))
                Energy_evidence[k].append( -(1 * logsumexp(np.array(evidence)/1)))
                Energy_alpha[k].append( -(1 * logsumexp(np.array(alpha)/1)))
                Energy_Mass[k].append( -(1 * logsumexp(np.array(Mass)/1)))
                Entropy[k].append(brevis.utils.calcEntropy_Tensors2(tf.nn.softmax(prediction)).numpy())
                calibration[k].append(np.amax(tf.nn.softmax(prediction).numpy()))
                
    Outputs=[]
    for j in range(num_outputs):
#         "probs":Pred[j],
        # df = pd.DataFrame({"x":Results[j],"y":Labels[j],'sum':Sum[j],'uncert':Uncert[j],"belief masses":Evidence[j]})
        df = pd.DataFrame({"x":Results[j],"y":Labels[j],"uncert":Uncert[j],
                            "energy":Energy[j],
                            "Energy_softmax":Energy_softmax[j],
                            "Energy_evidence":Energy_evidence[j],
                            "Energy_alpha":Energy_alpha[j],
                            "Energy_Mass":Energy_Mass[j],
                            'entropy':Entropy[j],
                            'calibration':calibration[j],
                            "confidence_alea_uncert":conf[j],
                            "entropy_of_expected":entropy_of_exp[j],
                            "expected_entropy":expected_entropy[j],
                            "mutual_information":mutual_info[j],
                            "EPKL":epkl[j],
                            "differential_entropy":dentropy[j],
                          })
        conditions = [df['x'] == df['y'],df['x'] != df['y']]
        choices = [1, 0]
        #create new column in DataFrame that displays results of comparisons
        df['correct'] = np.int32(np.select(conditions, choices, default=None))
        Outputs.append(df)
    return Outputs


def calculateBranching(outputs,metrics=["energy"], threshold=None, main_exit_included=False,plot=True, exit_labels=['exit_1']):
    '''
    Calculate the Correct/Incorrect performance of the threshold for the provided results. 
    for single exit models, set main_exit_included to False to not accept all results at last exit.        
    '''
    lessThanMetrics = ["energy","uncert","entropy"]
    
    if type(outputs ) is not list: #if not a list, put dataframe in a list, this allows us to use the same function for single and multiple exit model results.
        num_outputs = 1
        outputs = [outputs]
    else:
        num_outputs=len(outputs)

    if type(metrics ) is not list:
        metrics = [metrics]
    for j, metric in enumerate(metrics):
        print("metric: ", metric, "threshold: ",threshold)
        rollOver_indices = pd.Index([])
        _predictions = outputs.copy()
            # print(_branch_predictions)
        if main_exit_included:
            _predictions.append(_predictions.pop(0))
        Accepted_df = pd.DataFrame()

        Exit_Name=[]
        Accepted_list =[]
        Acceptance_correct =[]
        Input_predictions =[]
        # Branch_cost =[17443270,29419724,132134023] #flat exit costs
        # Branch_cost =[482376,1517643,80095445,114361924,112698838] #Conv2d exit costs

        # Base_cost = 112698838
        Branch_flops = []
        Thresholds=[]
        Test_accuracy =[]
        Rollover_accuracy=[]
        Results=[]

        _threshold = 0
        for i, output in enumerate(_predictions): 
            Test_accuracy.append(len(output.loc[(output["correct"] == True)])/len(output))
            if threshold:
                if type(threshold) is list:
                    if j >= len(threshold): #no threshold in the array so treat as None.
                        continue
                    _threshold = threshold[j]
                else:
                    _threshold = threshold
                if _threshold == "mean":
                    # _threshold = np.array(ID[metric]).mean()
                    Correct = output.loc[(output["correct"] == True)]
                    _threshold = np.array(Correct[metric]).mean()
                if _threshold == "gmean":
                    AUC_thresholds = calc_AUC(output, metrics=metrics,plot = False)
                    _threshold = AUC_thresholds[j]
                if _threshold == "PR_AUC":
                    precision_, recall_, proba = precision_recall_curve(output['correct'], output[metric])
                    _threshold = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
                else:
                    _threshold = np.float32(_threshold)
            
            if len(rollOver_indices)>0:
                if plot:
                    print("rollover enabled, {} predictions provided".format(len(rollOver_indices)))
                output = output.iloc[rollOver_indices]
            Thresholds.append(_threshold)
            if main_exit_included and i == len(_predictions)-1 :
                Exit_Name.append("Main_exit")
                if plot:
                    print("main_exit")
                Accepted = output
            else:
                if metric in lessThanMetrics: ## metrics that require less than metric
                    Accepted = output.loc[(output[metric] <= _threshold)]
                    Rejected = output.loc[(output[metric] > _threshold)]
                else:
                    Accepted = output.loc[(output[metric] >= _threshold)]
                    Rejected = output.loc[(output[metric] < _threshold)]
                rollOver_indices = Rejected.index
                if i >= len(exit_labels):
                        exit_labels.append("exit_{}".format(i+1))
                print(exit_labels)
                Exit_Name.append(exit_labels[i])
            Results.append(Accepted)
            Accepted_list.append(len(Accepted))
            Acceptance_correct.append(len(Accepted.loc[(Accepted['correct'] == True)]))
            Input_predictions.append(len(output))
            # Branch_flops.append(len(Accepted)* Branch_cost[i]) 
            Correct = output.loc[(output["correct"] == True)]
            Incorrect = output.loc[(output["correct"] == False)]
            Rollover_accuracy.append(len(Correct)/len(output))

            if plot:
                print(len(Accepted),"inputs accepted", len(Accepted.loc[(Accepted['correct'] == True)]),"Correct")
                _ = plt.hist(Correct[metric].tolist(), bins=100)  # arguments are passed to np.histogram
                _ = plt.hist(Incorrect[metric].tolist(), bins=100,color ="red", alpha = 0.5)  # arguments are passed to np.histogram
                plt.axvline(x=_threshold, color='k', linestyle='--',label="threshold")
                plt.title(metric + " outliers")
                plt.legend(["threshold","Correct","Incorrect"])
                plt.xlabel("entropy")
                plt.ylabel("frequency")
                plt.show()
            # cumulativeClassification(output['correct'].tolist(),output['uncert'].tolist(),20,thresholdType="<=")
                print("-----------------")
        _Results = pd.concat(Results)
        # print(_Results)
        # print(_Results.groupby("testy").count())
        df = pd.DataFrame({
            "Exit_Name":Exit_Name,
            "Predictions":Input_predictions,
            "Test_Accuracy":Test_accuracy,
            "RollOver_Accuracy":Rollover_accuracy,
            "Threshold":Thresholds,
            "Accepted":Accepted_list,
            "Accepted_Correct":Acceptance_correct,
            "Accepted_Ratio":np.array(Accepted_list)/np.array(Input_predictions),
            "Acceptance_Accuracy":np.array(Acceptance_correct)/np.array(Accepted_list),
            
            # "Flops":Branch_flops,
            # "Cost Ratio":,                                  
                          })
        with pd.option_context('expand_frame_repr', False):
            print (df)

def buildCompareDistribPlot(ID,OOD,metrics=["energy"], threshold=None, legend=["In Distribution","Out of Distribution"],main_exit_included=True,plot=True,exit_labels=['exit_1']):
        lessThanMetrics = ["energy","uncert","entropy"]
        if type(metrics ) is not list:
            metrics = [metrics]
        for j, metric in enumerate(metrics):
            print("metric: ", metric, "threshold: ",threshold)
            rollOver_ID_indices = pd.Index([])
            rollOver_OOD_indices = pd.Index([])
            Exit_Name=[]
            _ID = ID.copy()
            _OOD = OOD.copy()
                # print(_branch_predictions)
            if main_exit_included:
                _ID.append(_ID.pop(0))
                _OOD.append(_OOD.pop(0))
            Accepted_df = pd.DataFrame()
            Input_ID=[]
            Input_OOD=[]
            Accepted_list =[]
            Accepted_ID_list = []
            Accepted_OOD_list = []
            Acceptance_correct =[]
            Input_predictions =[]
            Accepted_Ratio_list=[]
            Accepted_Accuracy_list=[]
            # Branch_cost =[17443270,29419724,132134023] #flat exit costs
            # Branch_cost =[482376,1517643,80095445,114361924,112698838] #Conv2d exit costs

            # Base_cost = 112698838
            Branch_flops = []
            Thresholds=[]
            Test_accuracy =[]
            Rollover_accuracy=[]
            Results=[]
            for i, (output_ID, output_OOD) in enumerate(zip(_ID, _OOD)): 
                Test_accuracy.append(len(output_ID.loc[(output_ID["correct"] == True)])/len(output_ID))
                
                legend = ["threshold","correct","incorrect", "OOD"]
                Correct = output_ID.loc[(output_ID['correct'] == True)]
                Incorrect = output_ID.loc[(output_ID['correct'] == False)]
                if plot:
                    _ = plt.hist(Correct[metric].tolist(), bins=100)  # arguments are passed to np.histogram
                    _ = plt.hist(Incorrect[metric].tolist(), bins=100,color ="red", alpha = 0.5)  # arguments are passed to np.histogram
                    _ = plt.hist(output_OOD[metric].tolist(), bins=100,color="grey",alpha=0.5)  # arguments are passed to np.histogram

                
                if threshold:
                    if type(threshold) is list:
                        if j >= len(threshold): #no threshold in the array so treat as None.
                            continue
                        _threshold = threshold[j]
                    else:
                        _threshold = threshold
                    if _threshold == "mean":
                        # _threshold = np.array(ID[metric]).mean()
                        Correct = output_ID.loc[(output_ID["correct"] == True)]
                        _threshold = np.array(Correct[metric]).mean()
                    if _threshold == "gmean":
                        AUC_thresholds = calc_AUC(output_ID, metrics=metrics,plot = False)
                        _threshold = AUC_thresholds[j]
                    if _threshold == "PR_AUC":
                        precision_, recall_, proba = precision_recall_curve(output_ID['correct'], output_ID[metric])
                        _threshold = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
                    else:
                        _threshold = np.float32(_threshold)

                if len(rollOver_ID_indices)>0:
                    # print("rollover enabled, {} ID predictions provided".format(len(rollOver_ID_indices)))
                    output_ID = output_ID.iloc[rollOver_ID_indices]
                if len(rollOver_OOD_indices)>0:
                    # if plot:
                    # print("rollover enabled, {} OOD predictions provided".format(len(rollOver_OOD_indices)))
                    output_OOD = output_OOD.iloc[rollOver_OOD_indices]
                    
                if plot:
                    plt.axvline(x=_threshold, color='k', linestyle='--',label="threshold")
                    plt.title(metric + " outliers")
                    plt.legend(legend)
                    plt.xlabel("entropy")
                    plt.ylabel("frequency")
                    plt.show()
                if main_exit_included and i == len(_ID)-1 :
                    Exit_Name.append("Main_exit")
                    _threshold
                    if plot:
                        print("main_exit")
                    OOD_accepted = output_OOD
                    OOD_rejected = None
                    ID_accepted = output_ID
                    ID_rejected = None
                    accepted_correct = ID_accepted.loc[(ID_accepted["correct"] == True )] #TP
                    rejected_correct = None
                    accepted_incorrect = ID_accepted.loc[(ID_accepted[metric] ==False)] #FP
                    rejected_incorrect = None
                    accepted_ID_acc = len(accepted_correct) / (len( ID_accepted))
                    overall_accepted_acc = len(accepted_correct) / (len( ID_accepted) + len(OOD_accepted))
                    _threshold = "NA"
                    ### make a threshold that accepts everything, if less than, set to inf, if greater than, set to neg inf?
                    # if metric in lessThanMetrics:
                        # _threshold = math.inf
                    # else:
                        # _threshold = -math.inf
                # print(_threshold)
                else:
                    if metric in lessThanMetrics: ## metrics that require less than metric
                        OOD_accepted = output_OOD.loc[(output_OOD[metric].tolist() <= _threshold)] #FP
                        OOD_rejected = output_OOD.loc[(output_OOD[metric].tolist() > _threshold)] #TN
                        ID_accepted = output_ID.loc[(output_ID[metric] <= _threshold)] #TP
                        ID_rejected = output_ID.loc[(output_ID[metric] > _threshold)] #FN


                        accepted_correct = ID_accepted.loc[(ID_accepted["correct"] == True )] #TP
                        rejected_correct = ID_rejected.loc[(ID_rejected["correct"] == True)]  #FN

                        accepted_incorrect = ID_accepted.loc[(ID_accepted[metric] ==False)] #FP
                        rejected_incorrect = ID_rejected.loc[(ID_rejected[metric] ==False)] #TN

                        accepted_ID_acc = len(accepted_correct) / (len( ID_accepted))
                        overall_accepted_acc = len(accepted_correct) / (len( ID_accepted) + len(OOD_accepted))
                        # print("OOD accepted:", len(OOD_accepted),": with threshold:",_threshold )
                        # print("ID accepted:", len(ID_accepted), ":with acc:",(accepted_ID_acc))
                        # print("overall Accepted acc:",(overall_accepted_acc))

                        # print("OOD accepted with avg ID ",metric," threshold of ",_threshold, ": ", len(OOD.loc[(OOD[metric].tolist() <= _threshold)]), "out of ", len(OOD))
                        # print("ID accepted with avg ID ",metric," threshold of ",_threshold, ": ", len(ID.loc[(ID[metric] <= _threshold)]), "out of ", len(ID), "with acc of ", len(ID.loc[(ID[metric] <= _threshold) & ID['correct'] == True])/len(ID.loc[(ID[metric] <= _threshold)]))
                        # print("Overall accuracy of accepted inputs:", len(ID.loc[(ID[metric] <= _threshold) & ID['correct'] == True])/(len(ID.loc[(ID[metric] <= _threshold)])+len(OOD.loc[(OOD[metric] <= _threshold)])))
                    else: ### metrics that require greater than metric
                        OOD_accepted = output_OOD.loc[(output_OOD[metric].tolist() >= _threshold)] #FP
                        OOD_rejected = output_OOD.loc[(output_OOD[metric].tolist() < _threshold)] #TN
                        ID_accepted = output_ID.loc[(output_ID[metric] >= _threshold)] #TP
                        ID_rejected = output_ID.loc[(output_ID[metric] < _threshold)] #FN

                        accepted_correct = ID_accepted.loc[(ID_accepted["correct"] == True )] #TP
                        rejected_correct = ID_rejected.loc[(ID_rejected["correct"] == True)]  #FN

                        accepted_incorrect = ID_accepted.loc[(ID_accepted[metric] ==False)] #FP
                        rejected_incorrect = ID_rejected.loc[(ID_rejected[metric] ==False)] #TN



                        accepted_ID_acc = len(accepted_correct) / (len( ID_accepted))
                        overall_accepted_acc = len(accepted_correct) / (len( ID_accepted) + len(OOD_accepted))
                        # print("OOD accepted:", len(OOD_accepted),": with threshold:",_threshold )
                        # print("ID accepted:", len(ID_accepted), ":with acc:",(accepted_ID_acc))
                        # print("overall Accepted acc:",(overall_accepted_acc))

                        # print("OOD accepted with avg ID ",metric," threshold of ",_threshold, ": ", len(OOD.loc[(OOD[metric].tolist() >= _threshold)]), "out of ", len(OOD))
                        # print("ID accepted with avg ID ",metric," threshold of ",_threshold, ": ", len(ID.loc[(ID[metric] >= _threshold)]), "out of ", len(ID), "with acc of ", len(ID.loc[(ID[metric] >= _threshold) & ID['correct'] == True])/len(ID.loc[(ID[metric] >= _threshold)]))
                        # print("Overall accuracy of accepted inputs:", len(ID.loc[(ID[metric] <= _threshold) & ID['correct'] == True])/(len(ID.loc[(ID[metric] >= _threshold)])+len(OOD.loc[(OOD[metric] >= _threshold)])))
                    rollOver_ID_indices = ID_rejected.index
                    rollOver_OOD_indices = OOD_rejected.index
                    if i >= len(exit_labels):
                        exit_labels.append("exit_{}".format(i+1))
                    print(exit_labels)
                    Exit_Name.append(exit_labels[i])
                Thresholds.append(_threshold)
                
                Results.append(accepted_correct +accepted_incorrect)
                Input_ID.append(len(output_ID))
                Input_OOD.append(len(output_OOD))
                Accepted_ID_list.append(len(ID_accepted))
                Accepted_OOD_list.append(len(OOD_accepted))
                Accepted_Ratio_list.append(len(ID_accepted)/(len(ID_accepted)+ len(OOD_accepted)))
                Acceptance_correct.append(len(accepted_correct))
                Accepted_Accuracy_list.append(overall_accepted_acc)
            df = pd.DataFrame({
            "Exit_Name":Exit_Name,
            "ID_Inputs":Input_ID,
            "OOD_Inputs":Input_OOD,
            "Test_Accuracy":Test_accuracy,
            # "RollOver_Accuracy":Rollover_accuracy,
            "Threshold":Thresholds,
            "Accepted ID":Accepted_ID_list,
            "Accepted OOD":Accepted_OOD_list,
                
            "Accepted_Correct":Acceptance_correct,
            "Accepted_ID_Ratio":Accepted_Ratio_list,
            "Acceptance_Accuracy":Accepted_Accuracy_list,

            # "Flops":Branch_flops,
            # "Cost Ratio":,                                  
                          })
            with pd.option_context('expand_frame_repr', False):
                print (df)
                # print("TPR_ID-OOD",len(ID_accepted)/(len(ID_accepted) + len(ID_rejected)))
                # print("TPR_acc",len(accepted_correct)/(len(accepted_correct) + len(rejected_correct)))
                # if len(OOD) > 0:
                #     print("FPR_ID-OOD",len(OOD_accepted)/(len(OOD_accepted) + len(OOD_rejected)))
                # else: 
                #     print("FPR for OOD is div by zero, was OOD included?")
                # print("FPR_acc",len(accepted_incorrect)/(len(accepted_incorrect) + len(rejected_incorrect)))


####TODO 02/22 fix all these functions up to be dataset and model agnostic. also fix the importing of the rest of the module


def threshold_fn(Predictions):
    '''
        function to calcuate the threshold given a set of predictions.
        this function is to be overwritten by for custom thresholding functions.
    '''
    # print(Predictions)
    mean = Predictions.loc[(Predictions['Acc'] == False)].groupby("Acc")["evidence"].mean().iloc[0]
    std = Predictions.loc[(Predictions['Acc'] == False)].groupby("Acc")["evidence"].std().iloc[0]
    threshold = mean + std

    # _Incorrects_missed = Predictions.loc[(Predictions['Acc'] == False)  & (Predictions["overlap"] == 1)] #all the predictions that the main exit got true and the branch got wrong
    #     if len(_Incorrects_missed) > 0 :
    #         mean = _Incorrects_missed.groupby("Acc")["evidence"].mean().iloc[0]
    #         std = _Incorrects_missed.groupby("Acc")["evidence"].std().iloc[0]
    #     else:
    #         mean = Predictions.loc[(Predictions['Acc'] == False)].groupby("Acc")["evidence"].mean().iloc[0]
    #         std = Predictions.loc[(Predictions['Acc'] == False)].groupby("Acc")["evidence"].std().iloc[0]

    return threshold


def collectEvidence_branches(model,test_ds, evidence=True,stopping_point=None):
    num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
    print("outputs",num_outputs)
#     train_ds, test_ds, validation_ds = (dataset)
    predictions = []
    labels = []
    pClass = []
    predictions=[]
    pEvidence = []
    pUncertainty=[]
    pOverlap=[]

    Outputs = pd.DataFrame()
    pAcc=[]
    for i in range(num_outputs):
        pClass.append([])
        predictions.append([])
        pEvidence.append([])
        pUncertainty.append([])
        pAcc.append([])
        pOverlap.append([])
        # pOutputs.append([])

    for i, (x,y) in enumerate(test_ds):
        if stopping_point and i > stopping_point:
            break
        print("prediction: {} of {}".format(i,len(test_ds)),end='\r')
        if evidence: 
            result = model.test_on_batch(x,y)
            if i < 2:
                print(result)
#             print(result)
            for j in range(num_outputs):
#                 print("output",j)
                pClass[j].append(tf.argmax(y[0]).numpy())
#                 print("class",pClass[j][i])
                pAcc[j].append(result[j+(num_outputs)+1])  
#                 print("acc",pAcc[j][i])
                if j ==0:
                    pEvidence[j].append(0)
                else:
#                     print("evid Number",((num_outputs * 2)+1), " ", ((j-1)*3))
                    pEvidence[j].append(result[((num_outputs * 2) + 1)+((j-1)*3)])
#                 print("evid",pEvidence[j][i])

                pOverlap[j].append(pAcc[0][i] - pAcc[j][i])
#                 print("overlap",pOverlap[j][i])
        else:
            result = model.predict(x)[0]
            if i < 2:
                print(result)
    
            for j in range(num_outputs):
                pClass[j].append(tf.argmax(y[0]).numpy())
                # print(pClass[j])
                # print(result)
                prediction = np.argmax(result[j])
                if prediction == pClass[j][i]:
                    pAcc[j].append(1)  
                else:
                    pAcc[j].append(0)  
                # print(branching.utils.calcEntropy_Tensors(result[j]).numpy())
                pEvidence[j].append(branching.utils.calcEntropy_Tensors(result[j]).numpy()[0])

                pOverlap[j].append(pAcc[0][i] - pAcc[j][i])
        '''
        overlap
        if zero, both match, if else they don't match
        TT 1-1 =0
        TF 1-0 =1

        FT 0-1 = -1
        FF 0-0 =0
        
        '''
    Outputs=[]
    for j in range(num_outputs):
        Predictions = pd.DataFrame({"label":pClass[j],"evidence":pEvidence[j],"Acc":pAcc[j], "overlap":pOverlap[j]})
        Outputs.append(Predictions)
    
    print("Done")
    
    return Outputs



def displayEvidence_cascade(branch_predictions, thresholds=None, output_names=["branch_1","branch_2","branch_3","Main_Exit"], Evidence = True):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    series=[]
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()
    Outputs=pd.DataFrame()
    #lets reorder the predictions so that the final layer is at the end
    # _branch_predictions.copy()
    _branch_predictions = branch_predictions.copy()
    # print(_branch_predictions)
    _branch_predictions.append(_branch_predictions.pop(0))
    # print(_branch_predictions)
    rollOver_indices = pd.Index([])
    for i, Predictions in enumerate(_branch_predictions):
        #check if rollover is active, if so, select only the predictions whose indexes match the rollover list
        # print(rollOver_indices)
        test_acc = pd.Series([0,0],[True,False]) ##set defaults 
        test_acc.update(Predictions["Acc"].astype('bool').value_counts())

        test_accuracy = (test_acc.loc[True] /  (test_acc.loc[True] + test_acc.loc[False]))
        if len(rollOver_indices)>0:
            print("rollover enabled, {} predictions provided".format(len(rollOver_indices)))
            Predictions = Predictions.iloc[rollOver_indices]
        # print(Predictions.shape)
        Predictions["Acc"]=Predictions["Acc"].astype('bool')
        # Predictions["evidence"]=Predictions["evidence"].()[0]
        acc = pd.Series([0,0],[True,False]) ##set defaults 
        acc.update(Predictions["Acc"].value_counts())


        # print(acc)
        # print((acc.loc[True] , acc.loc[False]))
        _Incorrects_missed = Predictions.loc[(Predictions['Acc'] == False)  & (Predictions["overlap"] == 1)] #all the predictions that the main exit got true and the branch got wrong
        if len(_Incorrects_missed) > 0 :
            mean = _Incorrects_missed.groupby("Acc")["evidence"].mean().iloc[0]
            std = _Incorrects_missed.groupby("Acc")["evidence"].std().iloc[0]
        else:
            mean = Predictions.loc[(Predictions['Acc'] == False)].groupby("Acc")["evidence"].mean().iloc[0]
            std = Predictions.loc[(Predictions['Acc'] == False)].groupby("Acc")["evidence"].std().iloc[0]

        print("mean",mean , " std",std)
        
        correct_rows = Predictions.loc[Predictions['Acc'] == True]
        incorrect_rows = Predictions.loc[Predictions['Acc'] == False]
        
        E_threshold = -1 #-1 is null for threshold
        if thresholds is not None:
            try:
                E_threshold = thresholds[i]
            except:
                print("threshold not supplied for branch {}, using test data".format(i))
                
        if Evidence:
            if E_threshold ==-1:
                E_threshold = mean + std
            Accepted = Predictions.loc[(Predictions["evidence"] >= E_threshold)]
            Rejected = Predictions.loc[(Predictions["evidence"] < E_threshold)]
        else: 
            if E_threshold ==-1:
                E_threshold = mean - std
            Accepted = Predictions.loc[(Predictions["evidence"] <= E_threshold)]
            Rejected = Predictions.loc[(Predictions["evidence"] > E_threshold)]
        
        rollOver_indices = Rejected.index
        Incorrects_overlap = Accepted.loc[(Accepted['Acc'] == False) & (Accepted["overlap"] == 0)].count().iloc[0]
        Outputs = Outputs.append(pd.DataFrame({"Branch Name":output_names[i],
                "Predictions": len(Predictions.index),
                "test_accuracy": test_accuracy,
                "Accuracy":(acc.loc[True] /  (acc.loc[True] + acc.loc[False])),
                "E_Threshold":E_threshold,
                # "Overlap_Threshold":non_overlapping_incorrects_threshold,
                "acceptance_rate":Accepted.shape[0]/(Predictions.shape[0]),
                "accepted_correct":Accepted.loc[(Predictions['Acc'] == True)].shape[0],
                "accepted_incorrect":Accepted.loc[(Predictions['Acc'] == False)].shape[0],
                "accepted_accuracy":(Accepted.loc[(Accepted['Acc'] == True)].shape[0])/ Accepted.shape[0],
                "overlap_adjusted_accuracy":(Accepted.loc[(Accepted['Acc'] == True)].count()[0] + Incorrects_overlap) / Predictions.loc[(Predictions["evidence"] >E_threshold)].count()[0],
                "M(T) B(F)":Accepted.loc[(Accepted["overlap"] == 1)].count().iloc[0],
                "M(F) B(T)":Accepted.loc[(Accepted["overlap"] ==-1)].count().iloc[0],
                "M(F) B(F) overlap":Incorrects_overlap,
                },index=[i]))
#         print("TT",Accepted.loc[(Accepted["Acc"] ==True) & (Accepted["overlap"] == 0)])
#         print("TF",Accepted.loc[(Accepted["overlap"] == 1)])
#         print("FT",Accepted.loc[(Accepted["overlap"] == -1)])
#         print("FF",Accepted.loc[(Accepted["Acc"] ==False) & (Accepted["overlap"] == 0)])
        axs[round(int(i/2)), round(i%2)]
    # fig, axs = plt.subplots(1, 2)
        # axs[round(int(i/2)), round(i%2)].suptitle('Horizontally stacked subplots')
        axs[round(int(i/2)), round(i%2)].scatter(correct_rows['label'],correct_rows['evidence'],c ='r',marker='+')
        axs[round(int(i/2)), round(i%2)].scatter(incorrect_rows['label']+.3,incorrect_rows['evidence'],c ='k',marker='x')
        axs[round(int(i/2)), round(i%2)].plot(np.repeat(E_threshold,11),'b--')
        axs[round(int(i/2)), round(i%2)].title.set_text(output_names[i])
    
    for ax in fig.axes:
            # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axLine, axLabel = ax.get_legend_handles_labels()
            lines=(axLine)
            labels=(axLabel)
    fig.text(0.5, 0.01, 'Items Exit at Branch', ha='center', va='center')
    fig.text(0.01, 0.5, 'Accuracy %', ha='center', va='center', rotation='vertical')
    # fig.legend(lines, labels,bbox_to_anchor=(1., 1), loc=2,borderaxespad=0.,frameon=True)
    # fig.set_size_inches(10, 10)
    plt.show()
    return Outputs


def displayEvidence(branch_predictions, output_names=["main_exit","branch_1","branch_2","branch_3"], Evidence = True):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    series=[]
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()
    Outputs=pd.DataFrame()
    for i, Predictions in enumerate(branch_predictions):

        acc = pd.Series([0,0],[True,False]) ##set defaults 
        acc.update(Predictions["Acc"].astype('bool').value_counts())

        mean = Predictions.loc[(Predictions['Acc'] == False)].groupby("Acc")["evidence"].mean().iloc[0]
        std = Predictions.loc[(Predictions['Acc'] == False)].groupby("Acc")["evidence"].std().iloc[0]
        
        correct_rows = Predictions.loc[Predictions['Acc'] == True]
        incorrect_rows = Predictions.loc[Predictions['Acc'] == False]
        if Evidence:
            E_threshold = mean + std
            Incorrects_overlap = Predictions.loc[(Predictions['Acc'] == False)  & (Predictions["evidence"] > E_threshold) & (Predictions["overlap"] == 0)].count().iloc[0]
            Outputs = Outputs.append(pd.DataFrame({"Branch Name":output_names[i],
                    "Accuracy":(acc.loc[True] /  (acc.loc[True] + acc.loc[False])),
                    "E_Threshold":E_threshold,
                    # "Overlap_Threshold":non_overlapping_incorrects_threshold,
                    "acceptance_rate":Predictions.loc[(Predictions["evidence"] > E_threshold)].sort_values("evidence").shape[0]/(Predictions.count().iloc[0]),
                    "accepted_correct":Predictions.loc[(Predictions['Acc'] == True)  & (Predictions["evidence"] > E_threshold)].sort_values("evidence").shape[0],
                    "accepted_incorrect":Predictions.loc[(Predictions['Acc'] == False)  & (Predictions["evidence"] > E_threshold)].sort_values("evidence").shape[0],
                    "accepted_accuracy":(Predictions.loc[(Predictions['Acc'] == True)  & (Predictions["evidence"] > E_threshold)].sort_values("evidence").shape[0])/ Predictions.loc[(Predictions["evidence"] > E_threshold)].count()[0],
                    "overlap_adjusted_accuracy":(Predictions.loc[(Predictions['Acc'] == True)  & (Predictions["evidence"] > E_threshold)].sort_values("evidence").shape[0] + Incorrects_overlap) / Predictions.loc[(Predictions["evidence"] > E_threshold)].count()[0],
                    "rejected_correct":Predictions.loc[(Predictions['Acc'] == True)  & (Predictions["evidence"] < E_threshold)].sort_values("evidence").shape[0],
                    "rejected_incorrect":Predictions.loc[(Predictions['Acc'] == False)  & (Predictions["evidence"] < E_threshold)].sort_values("evidence").shape[0],
                    "Incorrects_overlap":Incorrects_overlap,
                    },index=[i]))
        else:
            print("mean",mean , " std",std)
            E_threshold = mean - std
            Incorrects_overlap = Predictions.loc[(Predictions['Acc'] == False)  & (Predictions["evidence"] < E_threshold) & (Predictions["overlap"] == 0)].count().iloc[0]
            # if i ==1 or i == 0 :
                # print(Predictions)
                # print(Predictions.loc[ (Predictions['Acc'] == True)  & (Predictions["overlap"] == 0) ])
            # print(Predictions.loc[(Predictions['Acc'] == True)  & (Predictions["evidence"] < E_threshold) ].count())
            Outputs = Outputs.append(pd.DataFrame({"Branch Name":output_names[i],
                    "Accuracy":(acc.loc[True] /  (acc.loc[True] + acc.loc[False])),
                    "E_Threshold":E_threshold,
                    # "Overlap_Threshold":non_overlapping_incorrects_threshold,
                    "acceptance_rate":Predictions.loc[(Predictions["evidence"] < E_threshold)].sort_values("evidence").shape[0]/(Predictions.count().iloc[0]),
                    "accepted_correct":Predictions.loc[(Predictions['Acc'] == True)  & (Predictions["evidence"] < E_threshold)].sort_values("evidence").shape[0],
                    "accepted_incorrect":Predictions.loc[(Predictions['Acc'] == False)  & (Predictions["evidence"] < E_threshold)].sort_values("evidence").shape[0],
                    "accepted_accuracy":(Predictions.loc[(Predictions['Acc'] == True)  & (Predictions["evidence"] < E_threshold)].sort_values("evidence").shape[0])/ Predictions.loc[(Predictions["evidence"] < E_threshold)].count()[0],
                    "overlap_adjusted_accuracy":(Predictions.loc[(Predictions['Acc'] == True)  & (Predictions["evidence"] < E_threshold)].sort_values("evidence").shape[0] + Incorrects_overlap) / Predictions.loc[(Predictions["evidence"] < E_threshold)].count()[0],
                    "rejected_correct":Predictions.loc[(Predictions['Acc'] == True)  & (Predictions["evidence"] > E_threshold)].sort_values("evidence").shape[0],
                    "rejected_incorrect":Predictions.loc[(Predictions['Acc'] == False)  & (Predictions["evidence"] > E_threshold)].sort_values("evidence").shape[0],
                    "Incorrects_overlap":Incorrects_overlap,
                    },index=[i]))
        axs[round(int(i/2)), round(i%2)]
    # fig, axs = plt.subplots(1, 2)
        # axs[round(int(i/2)), round(i%2)].suptitle('Horizontally stacked subplots')
        axs[round(int(i/2)), round(i%2)].scatter(correct_rows['label'],correct_rows['evidence'],c ='r',marker='+')
        axs[round(int(i/2)), round(i%2)].scatter(incorrect_rows['label']+.3,incorrect_rows['evidence'],c ='k',marker='x')
        axs[round(int(i/2)), round(i%2)].plot(np.repeat(E_threshold,11),'b--')
        axs[round(int(i/2)), round(i%2)].title.set_text("evidence")
    
    for ax in fig.axes:
            # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axLine, axLabel = ax.get_legend_handles_labels()
            lines=(axLine)
            labels=(axLabel)
    fig.text(0.5, 0.01, 'Items Exit at Branch', ha='center', va='center')
    fig.text(0.01, 0.5, 'Accuracy %', ha='center', va='center', rotation='vertical')
    # fig.legend(lines, labels,bbox_to_anchor=(1., 1), loc=2,borderaxespad=0.,frameon=True)
    # fig.set_size_inches(10, 10)
    plt.show()
    return Outputs

def eval_branches( model, dataset, count = 1, options="accuracy"):
    """ evaulate func for checking how well a branched model is performing.
        function may be moved to eval_model.py in the future.
    """ 
    num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels

    (train_images, train_labels), (test_images, test_labels) = dataset
    
    print("ALEXNET {}".format(brevis.ALEXNET))
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

# def evalBranchMatrix( model, dataset):