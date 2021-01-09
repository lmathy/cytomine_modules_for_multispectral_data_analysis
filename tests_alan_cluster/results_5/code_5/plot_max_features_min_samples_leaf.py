import os
import numpy
import pandas
import sklearn
import sklearn.ensemble
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



def get_precision_epithelium(dataFrame):
    return dataFrame["0"][0]


def get_recall_epithelium(dataFrame):
    return dataFrame["0"][1]


def get_f1_score_epithelium(dataFrame):
    return dataFrame["0"][2]



def get_precision_stroma(dataFrame):
    return dataFrame["1"][0]


def get_recall_stroma(dataFrame):
    return dataFrame["1"][1]


def get_f1_score_stroma(dataFrame):
    return dataFrame["1"][2]



def get_precision_blood(dataFrame):
    return dataFrame["2"][0]


def get_recall_blood(dataFrame):
    return dataFrame["2"][1]


def get_f1_score_blood(dataFrame):
    return dataFrame["2"][2]



def get_precision_necrosis(dataFrame):
    return dataFrame["3"][0]


def get_recall_necrosis(dataFrame):
    return dataFrame["3"][1]


def get_f1_score_necrosis(dataFrame):
    return dataFrame["3"][2]



def get_accuracy(dataFrame):
    return dataFrame["accuracy"][0]



def get_precision_macro_average(dataFrame):
    return dataFrame["macro avg"][0]


def get_recall_macro_average(dataFrame):
    return dataFrame["macro avg"][1]


def get_f1_score_macro_average(dataFrame):
    return dataFrame["macro avg"][2]



def get_precision_weighted_average(dataFrame):
    return dataFrame["weighted avg"][0]


def get_recall_weighted_average(dataFrame):
    return dataFrame["weighted avg"][1]


def get_f1_score_weighted_average(dataFrame):
    return dataFrame["weighted avg"][2]



if __name__ == "__main__":
    FOLDERDATAFRAMES = "./histology_classification_data_frames"
    METRICFUNCTION = get_f1_score_weighted_average
    
    
    modelAlgorithmList = []
    
    i = 5
    while i <= 50:
        
        j = 1
        while j <= 50:
            
            tupleToAdd = (sklearn.ensemble.RandomForestClassifier, {"n_estimators" : 500, "max_features" : i, "min_samples_leaf" : j, "n_jobs" : -1, "random_state" : 0})
            modelAlgorithmList.append(tupleToAdd)
            
            if j == 1:
                j = 5
            
            else:
                j += 5
        
        
        i += 5
    
    
    
    j = 1
    while j <= 50:
        
        tupleToAdd = (sklearn.ensemble.RandomForestClassifier, {"n_estimators" : 500, "max_features" : None, "min_samples_leaf" : j, "n_jobs" : -1, "random_state" : 0})
        modelAlgorithmList.append(tupleToAdd)
        
        if j == 1:
            j = 5
        
        else:
            j += 5
    
    
    
    maxFeaturesPlot = []
    minSamplesLeafPlot = []
    metricValues = []
    
    maxFeaturesPlot2 = []
    minSamplesLeafPlot2 = []
    metricValues2 = []
    
    idModel = 0
    
    for modelAlgorithm, additionalParametersModelAlgorithm in modelAlgorithmList:
        if idModel < 110:
            maxFeaturesPlot.append(additionalParametersModelAlgorithm["max_features"])
            minSamplesLeafPlot.append(additionalParametersModelAlgorithm["min_samples_leaf"])
            
            dataFrame = pandas.read_csv(os.path.join(FOLDERDATAFRAMES, str(idModel) + ".csv"))
            
            metricValues.append(METRICFUNCTION(dataFrame))
        
        
        else:
            maxFeaturesPlot2.append(additionalParametersModelAlgorithm["max_features"])
            minSamplesLeafPlot2.append(additionalParametersModelAlgorithm["min_samples_leaf"])
            
            dataFrame = pandas.read_csv(os.path.join(FOLDERDATAFRAMES, str(idModel) + ".csv"))
            
            metricValues2.append(METRICFUNCTION(dataFrame))
        
        
        idModel += 1
    
    
    
    minCMap = min([*metricValues, *metricValues2])
    maxCMap = max([*metricValues, *metricValues2])
    
    
    fig = plt.figure()
    ax = plt.axes(projection ="3d")
    
    scatterPlot = ax.scatter3D(maxFeaturesPlot, minSamplesLeafPlot, metricValues)
    
    ax.set_xlabel("maximum number of features to consider")
    ax.set_ylabel("minimum number of samples per leaf")
    
    plt.show()
    plt.close()
    
    
    dictionaryDataFrame = dict()
    dictionaryDataFrame["max_features"] = maxFeaturesPlot
    dictionaryDataFrame["min_samples_leaf"] = minSamplesLeafPlot
    dictionaryDataFrame["metric_values"] = metricValues
    dataFramePlot = pandas.DataFrame(dictionaryDataFrame)
    
    dataFramePlot.plot.scatter(x = "max_features", y = "min_samples_leaf", c = "metric_values", cmap = "viridis")
    
    plt.xlabel("maximum number of features to consider")
    plt.ylabel("minimum number of samples per leaf")
    
    plt.gcf().get_axes()[1].set_ylabel("")
    
    plt.show()
    plt.close()
    
    
    plt.plot(minSamplesLeafPlot2, metricValues2)
    plt.xlabel("minimum number of samples per leaf")
    plt.show()
    plt.close()
    
