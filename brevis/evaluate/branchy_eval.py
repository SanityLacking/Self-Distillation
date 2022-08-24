""" Evaulation Tools for branchynet models.
    Contains data graphing functions for visualizing the model, finding knee points, and graphing results.
"""
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

from brevis.utils import *


class branchyEval:
    def branchConfusionMatrix(predictions, labels):
        """ takes an array of predictions with multiple outputs and maps them to their labels. 
            with this you can see 
            Similar in concept to a standard confusion matrix of predictions and labels
        """
        matrix = np.array()
        for item in predictions:
            pass
        print("")

        return


    #Visualize Model
    def visualize_model(model,name=""):
        # tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        if name == "":
            name = "model_plot.png"
        else: 
            name = name + ".png"
        #plot_model(model, to_file=name, show_shapes=True, show_layer_names=True)

        

    def KneeGraph(pred, labels, entropy, num_outputs, classes, output_names=[]):
        """ generate a matrix of entropy values for all classes and outputs
            pred: list of all predicted labels
            labels: list of all actual labels. must match pred in size and shape
            classes: list of all classes, for example [0,1,2,3]
            output_names: list of names for each of the outputs. applies names to outputs in the same order as pred and labels.

        """    
        #graph the accuracy rate vs the entropy threshold.
        #get series of entropy values, series of 
        results = np.equal(pred, labels)
        pred = np.array(pred)
        labels = np.array(labels)
        entropy = np.array(entropy)
        classCount = {}
    #     results = pred
        labelClasses=classes
        transpose_results = np.transpose(results)
        transpose_preds = np.transpose(pred) #per exit rather then per input
        transpose_entropy = np.transpose(entropy) #per exit rather then per input
        transpose_labels = np.transpose(labels)
    #     print(transpose_results)
    #     print(transpose_preds)
    #     print(transpose_entropy)
    #     print(transpose_labels)
        # %matplotlib inline
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure()
        # ax = plt.axes()
        series=[]
        fig, axs = plt.subplots(2, 2)
        fig.tight_layout()
        df =  pd.DataFrame() 
    #     plt.subplots(2, 2, sharex='all', sharey='all')
        for i, branch in enumerate(transpose_entropy):
            series_branch=[]
            
            for j, ent in enumerate(branch):
                series_entropy = {}
                series_entropy["entropy"] = ent
    #             print("entropy: {}".format(ent))
    #             print(np.where(branch <= ent))
                series_entropy["pred"] = transpose_preds[i][np.where(branch <= ent)]
    #             print(series_entropy["pred"] )
                series_entropy["labels"] = transpose_labels[i][np.where(branch <= ent)]
    #             print(series_entropy["labels"] )
                series_entropy["truth"] = transpose_results[i][np.where(branch <= ent)]
    #             print(series_entropy["truth"] )
                series_entropy["accuracy"] = transpose_results[i][np.where(branch <= ent)].sum()/len(transpose_results[i])
    #             print(series_entropy["accuracy"])
                series_branch.append(series_entropy)
            df = pd.DataFrame(series_branch)
            df = df.sort_values(by=["entropy"])
            axs[round(int(i/2)), round(i%2)].plot(df["entropy"],df["accuracy"])
    #         axs[round(int(i/2)), round(i%2)].set_xlim([0,2])
            axs[round(int(i/2)), round(i%2)].set_ylim([0,1])
            if len(output_names) >= i:
                axs[round(int(i/2)), round(i%2)].title.set_text("branch: {}".format(output_names[i]))
            else:
                axs[round(int(i/2)), round(i%2)].title.set_text("branch: {}".format(i))
            series.append(series_branch)
        plt.show()
        return pd.DataFrame(series)

        
    def KneeGraphClasses(pred, labels, entropy, num_outputs, classes, output_names=[]):
        """ generate a matrix of entropy values for all classes and outputs
            pred: list of all predicted labels
            labels: list of all actual labels. must match pred in size and shape
            classes: list of all classes, for example [0,1,2,3]
            output_names: list of names for each of the outputs. applies names to outputs in the same order as pred and labels.

        """    
        #graph the accuracy rate vs the entropy threshold.
        #get series of entropy values, series of 
        resultsDict = {}
        results = np.equal(pred, labels)
        pred = np.array(pred)
        labels = np.array(labels)
        entropy = np.array(entropy)
        classCount = {}
    #     results = pred
        labelClasses=classes
        transpose_results = np.transpose(results) #truths
        transpose_preds = np.transpose(pred) #per exit rather then per input
        transpose_entropy = np.transpose(entropy) #per exit rather then per input
        transpose_labels = np.transpose(labels)
    #     print(transpose_results)
    #     print(transpose_preds)
    #     print(transpose_entropy)
    #     print(transpose_labels)
        # %matplotlib inline
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure()
        series=[]
        fig, axs = plt.subplots(2, 2)
        fig.tight_layout()
        df =  pd.DataFrame() 
    #     print(transpose_entropy)
    #     for branch in branches:
    #          for class in classes:
    #                 for entropy in entropies:
    #                     classAccuracy = sum of truth labels /count of truth labels where entropies <= entropy 
        returnData = []
        for i, branchEntropy in enumerate(transpose_entropy):
            print("branch {}: {}".format(i,branchEntropy))
            classEntropy = {}
            for j, labelClass in enumerate(labelClasses):
                classEntropy[labelClass] = []
    #             print("class {}".format(labelClass))
                for k, entropy in enumerate(branchEntropy):
                    # if there are no entries for a label class, this would produce an accuracy of NaN, so instead skip. 
                    if math.isnan(transpose_results[i][np.where((branchEntropy <= entropy) & (transpose_labels[i] == labelClass))].sum()/len(transpose_labels[i][np.where((branchEntropy <= entropy) & (transpose_labels[i] == labelClass))])) :
                        continue
                    seriesEntropy = {}
                    seriesEntropy["entropy"] = entropy
                    seriesEntropy["pred"] = transpose_preds[i][np.where((branchEntropy <= entropy) & (transpose_preds[i] == labelClass))]
                    seriesEntropy["accuracy"] = transpose_results[i][np.where((branchEntropy <= entropy) & (transpose_labels[i] == labelClass))].sum()/len(transpose_labels[i][np.where(transpose_labels[i] == labelClass)])
                    classEntropy[labelClass].append(seriesEntropy)
                df = pd.DataFrame(classEntropy[labelClass],columns=["entropy","pred","accuracy"])
                df = df.sort_values(by=["entropy"])
    #             print(df)
                axs[round(int(i/2)), round(i%2)].plot(df["entropy"],df["accuracy"], label="Class: {}".format(labelClass),alpha=0.8)        
            if len(output_names) >= i:
                axs[round(int(i/2)), round(i%2)].title.set_text("branch: {}".format(output_names[i]))
            else:
                axs[round(int(i/2)), round(i%2)].title.set_text("branch: {}".format(i))
            returnData.append(classEntropy)
            
        lines = []
        labels = []
        for ax in fig.axes:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axLine, axLabel = ax.get_legend_handles_labels()
            lines=(axLine)
            labels=(axLabel)
        
            # add a big axes, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("Entropy")
        plt.ylabel("Accuracy")

        fig.legend(lines, labels,bbox_to_anchor=(1., 1), loc=2,borderaxespad=0.,frameon=True)
        plt.show()
        returnData = pd.DataFrame(returnData)
        return returnData

    def KneeGraphPredictedClasses(pred, labels, entropy, num_outputs, classes, output_names=[]):
        """ generate a matrix of entropy values for all classes and outputs
            pred: list of all predicted labels
            labels: list of all actual labels. must match pred in size and shape
            classes: list of all classes, for example [0,1,2,3]
            output_names: list of names for each of the outputs. applies names to outputs in the same order as pred and labels.

        """    
        #graph the accuracy rate vs the entropy threshold.
        #get series of entropy values, series of 
        resultsDict = {}
        results = np.equal(pred, labels)
        pred = np.array(pred)
        labels = np.array(labels)
        entropy = np.array(entropy)
        classCount = {}
    #     results = pred
        labelClasses=classes
        transpose_results = np.transpose(results) #truths
        transpose_preds = np.transpose(pred) #per exit rather then per input
        transpose_entropy = np.transpose(entropy) #per exit rather then per input
        transpose_labels = np.transpose(labels)
    #     print(transpose_results)
    #     print(transpose_preds)
    #     print(transpose_entropy)
    #     print(transpose_labels)
        # %matplotlib inline
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure()
        series=[]
        fig, axs = plt.subplots(2, 2)
        fig.tight_layout()
        df =  pd.DataFrame() 
    #     print(transpose_entropy)
    #     for branch in branches:
    #          for class in classes:
    #                 for entropy in entropies:
    #                     classAccuracy = sum of truth labels /count of truth labels where entropies <= entropy 
        returnData = []
        for i, branchEntropy in enumerate(transpose_entropy):
            print("branch {}: {}".format(i,branchEntropy))
            classEntropy = {}
            for j, labelClass in enumerate(labelClasses):
                classEntropy[labelClass] = []
                print("class {}".format(labelClass))
                for k, entropy in enumerate(branchEntropy):
                    # if there are no entries for a label class, this would produce an accuracy of NaN, so instead skip. 
    #                 print("sum is: {} len is: {}, combined is: {}".format(transpose_results[i][np.where((branchEntropy <= entropy) & (transpose_labels[i] == labelClass))].sum(), len(transpose_labels[i][np.where((transpose_labels[i] == labelClass))]),transpose_results[i][np.where((branchEntropy <= entropy) & (transpose_labels[i] == labelClass))].sum()/len(transpose_labels[i][np.where((transpose_labels[i] == labelClass))])))
                    if len(transpose_preds[i][np.where((transpose_preds[i] == labelClass))])==0 :
    #                     print("skip!")
                        continue
                    seriesEntropy = {}
                    seriesEntropy["entropy"] = entropy
                    seriesEntropy["pred"] = transpose_preds[i][np.where((branchEntropy <= entropy) & (transpose_preds[i] == labelClass))] # select all where labelClass is PREDICTED
                    seriesEntropy["labels"] = transpose_labels[i][np.where((branchEntropy <= entropy) & (transpose_preds[i] == labelClass))] # select all where labelClass is PREDICTED
                    seriesEntropy["accuracy"] = transpose_results[i][np.where((branchEntropy <= entropy) & (transpose_labels[i] == labelClass))].sum()/len(seriesEntropy["pred"])
                    classEntropy[labelClass].append(seriesEntropy)
                df = pd.DataFrame(classEntropy[labelClass],columns=["entropy","pred","labels","accuracy"])
                df = df.sort_values(by=["entropy"])
                print(df)
                axs[round(int(i/2)), round(i%2)].plot(df["entropy"],df["accuracy"], label="Class: {}".format(labelClass),alpha=0.8)        
            if len(output_names) >= i:
                axs[round(int(i/2)), round(i%2)].title.set_text("branch: {}".format(output_names[i]))
            else:
                axs[round(int(i/2)), round(i%2)].title.set_text("branch: {}".format(i))
                
            returnData.append(classEntropy)
        lines = []
        labels = []
        for ax in fig.axes:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axLine, axLabel = ax.get_legend_handles_labels()
            lines=(axLine)
            labels=(axLabel)
        
            # add a big axes, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("Entropy")
        plt.ylabel("Accuracy")

        fig.legend(lines, labels,bbox_to_anchor=(1., 1), loc=2,borderaxespad=0.,frameon=True)
    #     plt.show()
        #######
        df = pd.DataFrame(columns=["entropy"])
        for i, branch in enumerate(returnData):
            branchOutput = []
            for j, classList in branch.items():
                entropy =np.empty([0,len(classList)])
                eList = []
                aList = []
                for k in classList:
                    eList.append(k["entropy"])
                    aList.append(k["accuracy"])
                e = pd.DataFrame({"entropy":eList,"branch{}_class{}_accuracy".format(i,j):aList})
                # print(e)
                # df.join(e.set_index('entropy'), on='entropy')
                df = pd.merge(df,e,on="entropy",how="outer")
                # df = pd.concat([df,e])
        print("-------")
        print(df)
        df.to_csv("graph_output_outer.csv")
        return returnData

    def entropyConfusionMatrix(pred, labels, entropy, num_outputs, classes, output_names=[]):
        """ generate a matrix of entropy values for all classes and outputs
            pred: list of all predicted labels
            labels: list of all actual labels. must match pred in size and shape
            classes: list of all classes, for example [0,1,2,3]
            output_names: list of names for each of the outputs. applies names to outputs in the same order as pred and labels.

        """    
    #     print(pred)
    #     print(labels)
        resultsDict = {}
        results = []
        pred = np.array(pred)
        labels = np.array(labels)
        entropy = np.array(entropy)
        classCount = {}
        results = pred
        labelClasses=classes
        from sklearn.metrics import confusion_matrix
        ### initialize the dictionary
        for i, labelClass in enumerate(labelClasses):    
            resultsDict[labelClass] ={}
            for j in range(num_outputs):
                resultsDict[labelClass][j] = []
    #         resultsDict[labelClass] = [0]*num_outputs
            classCount[labelClass] = 0
    #     print(resultsDict)
        ###loop through results 
        transpose_preds = np.transpose(results) #per exit rather then per input
        transpose_labels = np.transpose(labels)
        for i, item in enumerate(transpose_preds):
            print("exit:{}".format(i))
            df_confusion = pd.crosstab(item, transpose_labels[i], rownames=['Actual'], colnames=['Predicted'], margins=True)
            print(df_confusion)
    #         print(confusion_matrix(item,transpose_labels[i]))
        return

    def entropyMatrix(entropy, labels, num_outputs, classes, output_names=[]):
        """ generate a matrix of entropy values for all classes and outputs
            entropy: list of all predicted labels
            labels: list of all actual labels. must match pred in size and shape
            classes: list of all classes, for example [0,1,2,3]
            output_names: list of names for each of the outputs. applies names to outputs in the same order as pred and labels.

        """    
        # print(entropy)
        # print(labels)
        resultsDict = {}
        results = np.array(entropy)
        labels = np.array(labels)
        classCount = {}
        labelClasses=classes
        
        ### initialize the dictionary
        for i, labelClass in enumerate(labelClasses):    
            resultsDict[labelClass] ={}
            for j in range(num_outputs):
                resultsDict[labelClass][j] = []
    #         resultsDict[labelClass] = [0]*num_outputs
            classCount[labelClass] = 0
    #     print(resultsDict)
        ###loop through results 
        for i, item in enumerate(results):
            for j, branch in enumerate(item):
    #             print("{},{}".format(i, j))
    #             if branch == True: 
                resultsDict[labels[i][j]][j].append(branch)
            classCount[labels[i][0]] += 1
        # print(classCount)
        
        resultsDict = pd.DataFrame.from_dict(resultsDict,orient="index")
        for column in resultsDict:
            resultsDict[column] = [np.array(x).mean() for x in resultsDict[column].values]
        renameDict={}
        for i, name in enumerate(output_names):
            renameDict[i] = name
        print("rename:{}".format(renameDict))
        if len(renameDict) > 0:
            print("rename!")
            resultsDict = resultsDict.rename(renameDict,axis ="columns")
        resultsDict["itemCount"] = pd.Series(classCount)
        return resultsDict

    def throughputMatrix(pred, labels, num_outputs, classes, output_names=[]):
        """ generate a dictionary of lists comparing the correctly labeled predictions against the outputs for each class.        
            pred: list of all predicted labels
            labels: list of all actual labels. must match pred in size and shape
            classes: list of all classes, for example [0,1,2,3]
            output_names: list of names for each of the outputs. applies names to outputs in the same order as pred and labels.
        """    
        resultsDict = {}
        results = []
        #get truth matrix of the predictions/labels
        pred = np.array(pred)
        labels = np.array(labels)
        classCount = {}
        results = np.equal(pred, labels)
        labelClasses=classes
        # print("----")
        ### initialize the dictionary
        for i, labelClass in enumerate(labelClasses):    
            resultsDict[labelClass] ={}
            for j in range(num_outputs):
                resultsDict[labelClass][j] = 0
            classCount[labelClass] = 0
        ###loop through results 
        for i, item in enumerate(results):
            for j, branch in enumerate(item):
                if branch == True: 
                    resultsDict[labels[i][j]][j] += 1
            classCount[labels[i][0]] += 1
        resultsDict = pd.DataFrame.from_dict(resultsDict,orient="index")
        renameDict={}
        for i, name in enumerate(output_names):
            renameDict[i] = name
        print("rename:{}".format(renameDict))
        if len(renameDict) > 0:
            print("rename!")
            resultsDict = resultsDict.rename(renameDict,axis ="columns")
        resultsDict["itemCount"] = pd.Series(classCount)
        return resultsDict

        
    def find_neighbours(df, value, colname=""):
        """ find the closest matches to a value in a dataframe, if there are multiple matches, use the match that has the highest 
            count value, aka the most number of counts.
        """
        
        if isinstance(df, pd.DataFrame):
            if (df[colname].notna().sum()) <= 0:
                # no valid values were found, return 0 
                print("no Valid values were found for df")
                return 0
            exactmatch = df[df[colname] == value]
            if not exactmatch.empty:
                return exactmatch["count"].argmax()
            else:
    #             pd.set_option('display.max_rows', None) 
    #             print(df)
    #             pd.set_option('display.max_rows', 100) 
                try:            
                    lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
                except ValueError:
                    lowerneighbour_ind = 0
                try:
                    upperneighbour_ind = df[df[colname] > value][colname].idxmin()
                except ValueError:
                    upperneighbour_ind = 0
                    
    #             print("lowerneighbour_ind {}".format(lowerneighbour_ind))
    #             print("upperneighbour_ind {}".format(upperneighbour_ind))
                
                neighbours = df.iloc[[lowerneighbour_ind,upperneighbour_ind]]
    #             print(neighbours)
    #             print(neighbours["count"].idxmax())

    #             neighbours = neighbours["accuracy"].sub(value).abs().idxmin()
    #             print(neighbours)
                return neighbours["count"].idxmax()
        else:
            print("input is not a Dataframe, {}".format(type(df)))
            return None
        
    
def exitAccuracy(results, labels, classes=[]):
    """ find the accuracy scores of the main network exit for each class
        if classes is empty, return the average accuracy for all labels
    """    
    print(len(classes))
    classAcc = {}
    correct =[]
    count = []
    percentage = []
    if len(classes) > 0:
        for i, labelClass in enumerate(classes):            
            correct.append(results[np.where(labels==labelClass)].sum())
            count.append(len(labels[np.where(labels == labelClass)]))
            percentage.append(results[np.where(labels==labelClass)].sum()/len(labels[np.where(labels == labelClass)]))
            classAcc[labelClass] = results[np.where(labels==labelClass)].sum()/len(labels[np.where(labels == labelClass)])
    else: 
        classAcc["all"] = results.sum()/len(labels)
    return classAcc

def findMainExitAccuracies(pred, labels, num_outputs, labelClasses=[], output_names=[],graph=True):
        """ find the accuracy scores of the main network exit for each class
            if classes is empty, return the average accuracy for all labels
        """
        results = np.equal(pred, labels)
        pred = np.array(pred)
        print(pred.dtype)
        labels = np.array(labels)
        
        transpose_results = np.transpose(results) #truths
        transpose_preds = np.transpose(pred) #per exit rather then per input
        transpose_labels = np.transpose(labels)
        # %matplotlib inline
        if graph==True:
            plt.style.use('seaborn-whitegrid')
            fig = plt.figure()
            fig.tight_layout()
        df =  pd.DataFrame() 
        classAcc= exitAccuracy(transpose_results[0],transpose_labels[0],labelClasses)
        print(classAcc)
        fig, ax = plt.subplots()
        ticks = []
        if graph==True:
            for i, x in enumerate(classAcc):
                ticks.append(x)
                print(x)                
                plt.bar(x, classAcc[x], label="Class: {}".format(x),alpha=0.8) 
                if type(x) != str:
                    plt.text(x-.3, classAcc[x]-0.025, "{}".format(round(classAcc[x],2)), color='black', va='center', fontweight='bold')
                else: 
                    plt.text(x, classAcc[x]-0.025, "{}".format(round(classAcc[x],2)), color='black', va='center', fontweight='bold')
#                 plt.bar(x, .74, label="Avg Accuracy",alpha=0.8,bottom=classAcc[x]) 
            if len(labelClasses) > 0:
                plt.hlines(.74,-.5,len(classAcc.keys())-.5,label ="Accuracy", linestyles="dashed", alpha=0.5)
                plt.text(len(classAcc.keys())-.5, .74, ' Average Accuracy', ha='left', va='center')
            
    #     print(sum(classAcc.values())/len(classAcc.values()))
        if graph==True:
            plt.xticks(ticks)
            plt.title("Class Label Accuracy")
            plt.ylabel("Accuracy %")
            plt.xlabel("Label Class #")
            plt.show()
        return classAcc
    

    
def findThreshold(pred, labels, entropy, num_outputs, classes, output_names=[],mainBranchNum=0,graph=False):
    """    Find and Mark the threshold points for each class.
        mainbranchNum: the position in the pred array of the main exit, defaults to the first exit.
    """    
    resultsDict = {}
    results = np.equal(pred, labels)
    pred = np.array(pred)
    labels = np.array(labels)
    entropy = np.array(entropy)
    classCount = {}
    labelClasses=classes
    transpose_results = np.transpose(results) #truths
    transpose_preds = np.transpose(pred) #per exit rather then per input
    transpose_entropy = np.transpose(entropy) #per exit rather then per input
    transpose_labels = np.transpose(labels)
    if graph:
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure()
        series=[]
        fig, axs = plt.subplots(2, 2)
        fig.tight_layout()
    df =  pd.DataFrame() 
    returnData = []
    
    ##find the main exit accuracy levels to compare the branches to.
    ##assume the first branch is the main branch to match too
    
    mainAcc = branchyEval.findMainExitAccuracies(pred, labels, num_outputs, classes, output_names, graph=False)
    thresholdPoints={}
    colors = cm.rainbow(np.linspace(0, 1, len(labelClasses)))
    for i, branchEntropy in enumerate(transpose_entropy):
        print("branch {}".format(i))
        classEntropy = {}
        thresholdPoints[output_names[i]] = {}
        for j, labelClass in enumerate(labelClasses):
            classEntropy[labelClass] = []
            for k, entropy in enumerate(branchEntropy):
#                 print("sum is: {} len is: {}, combined is: {}".format(transpose_results[i][np.where((branchEntropy <= entropy) & (transpose_labels[i] == labelClass))].sum(), len(transpose_labels[i][np.where((transpose_labels[i] == labelClass))]),transpose_results[i][np.where((branchEntropy <= entropy) & (transpose_labels[i] == labelClass))].sum()/len(transpose_labels[i][np.where((transpose_labels[i] == labelClass))])))
                if len(transpose_preds[i][np.where((transpose_preds[i] == labelClass))])==0 :
#                     print("skip!")
                    continue
                seriesEntropy = {}
                seriesEntropy["entropy"] = entropy
                seriesEntropy["pred"] = transpose_preds[i][np.where((branchEntropy <= entropy) & (transpose_preds[i] == labelClass))] # select all where labelClass is PREDICTED
                seriesEntropy["labels"] = transpose_labels[i][np.where((branchEntropy <= entropy) & (transpose_preds[i] == labelClass))] # select all where labelClass is PREDICTED
                seriesEntropy["accuracy"] = transpose_results[i][np.where((branchEntropy <= entropy) & (transpose_labels[i] == labelClass))].sum()/len(transpose_labels[i][np.where((branchEntropy <= entropy) & (transpose_labels[i] == labelClass))])
                seriesEntropy["count"] = len(transpose_results[i][np.where((branchEntropy <= entropy) & (transpose_labels[i] == labelClass))])
                classEntropy[labelClass].append(seriesEntropy)
            df = pd.DataFrame(classEntropy[labelClass],columns=["entropy","pred","labels","accuracy","count"])
            threshold_idx = find_neighbours(df[["accuracy","count"]],mainAcc[labelClass],"accuracy")
            thresholdPoints[output_names[i]][labelClass] = df.iloc[threshold_idx][["accuracy","count","entropy"]].to_dict()
            maxRowidx = df["count"].idxmax()
#             print("nearest value to {} is {} at {} with {} counts ".format(mainAcc[labelClass],df["accuracy"][threshold_idx],threshold_idx, df["count"][threshold_idx]))
#             print("The max count value is {} with {} counts".format(float(df.iloc[maxRowidx]["accuracy"]), int(df.iloc[maxRowidx]["count"])))
            if graph:
                axs[round(int(i/2)), round(i%2)].plot(df.iloc[threshold_idx]["count"],df.iloc[threshold_idx]["accuracy"],marker='v', markersize=5, color=colors[j])
                df = df.sort_values(by=["count"])
                axs[round(int(i/2)), round(i%2)].plot(df["count"],df["accuracy"], label="Class: {}".format(labelClass), color=colors[j], alpha=0.8)        

                axs[round(int(i/2)), round(i%2)].plot()

                axs[round(int(i/2)), round(i%2)].set_ylim([0, 1])
        if graph:
            if len(output_names) >= i:
                axs[round(int(i/2)), round(i%2)].title.set_text("branch: {}".format(output_names[i]))
            else:
                axs[round(int(i/2)), round(i%2)].title.set_text("branch: {}".format(i))
            
        returnData.append(classEntropy)
    lines = []
    labels = []
    if graph:
        for ax in fig.axes:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axLine, axLabel = ax.get_legend_handles_labels()
            lines=(axLine)
            labels=(axLabel)

        # Set common labels
        fig.text(0.5, 0.01, 'Items Exit at Branch', ha='center', va='center')
        fig.text(0.01, 0.5, 'Accuracy %', ha='center', va='center', rotation='vertical')
        fig.legend(lines, labels,bbox_to_anchor=(1., 1), loc=2,borderaxespad=0.,frameon=True)
#     df.to_csv("graph_output.csv")
    return thresholdPoints