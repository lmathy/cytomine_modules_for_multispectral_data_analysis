import os
import numpy
import pandas
import matplotlib.pyplot as plt



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
    IDSTOTAKE = numpy.arange(0, 15)
    INDEX = numpy.arange(1, 30, 2)
    XLABEL = "subwindow width and height"
    
    
    precisionEpithelium = []
    recallEpithelium = []
    f1ScoreEpithelium = []
    
    precisionStroma = []
    recallStroma = []
    f1ScoreStroma = []
    
    precisionBlood = []
    recallBlood = []
    f1ScoreBlood = []
    
    precisionNecrosis = []
    recallNecrosis = []
    f1ScoreNecrosis = []
    
    accuracy = []
    
    precisionMacroAverage = []
    recallMacroAverage = []
    f1ScoreMacroAverage = []
    
    precisionWeightedAverage = []
    recallWeightedAverage = []
    f1ScoreWeightedAverage = []
    
    
    for identifier in IDSTOTAKE:
        dataFrame = pandas.read_csv(os.path.join(FOLDERDATAFRAMES, str(identifier) + ".csv"))
        
        
        precisionEpithelium.append(get_precision_epithelium(dataFrame))
        recallEpithelium.append(get_recall_epithelium(dataFrame))
        f1ScoreEpithelium.append(get_f1_score_epithelium(dataFrame))
        
        precisionStroma.append(get_precision_stroma(dataFrame))
        recallStroma.append(get_recall_stroma(dataFrame))
        f1ScoreStroma.append(get_f1_score_stroma(dataFrame))
        
        precisionBlood.append(get_precision_blood(dataFrame))
        recallBlood.append(get_recall_blood(dataFrame))
        f1ScoreBlood.append(get_f1_score_blood(dataFrame))
        
        precisionNecrosis.append(get_precision_necrosis(dataFrame))
        recallNecrosis.append(get_recall_necrosis(dataFrame))
        f1ScoreNecrosis.append(get_f1_score_necrosis(dataFrame))
        
        accuracy.append(get_accuracy(dataFrame))
        
        precisionMacroAverage.append(get_precision_macro_average(dataFrame))
        recallMacroAverage.append(get_recall_macro_average(dataFrame))
        f1ScoreMacroAverage.append(get_f1_score_macro_average(dataFrame))
        
        precisionWeightedAverage.append(get_precision_weighted_average(dataFrame))
        recallWeightedAverage.append(get_recall_weighted_average(dataFrame))
        f1ScoreWeightedAverage.append(get_f1_score_weighted_average(dataFrame))
    
    
    
    print("precision, recall, and f1 score for epithelium class.")
    
    dictionaryPlot = dict()
    dictionaryPlot["precision"] = precisionEpithelium
    dictionaryPlot["recall"] = recallEpithelium
    dictionaryPlot["f1 score"] = f1ScoreEpithelium
    
    dataFramePlot = pandas.DataFrame(dictionaryPlot, index = INDEX)
    dataFramePlot.plot()
    plt.xlabel(XLABEL)
    plt.show()
    plt.close()
    
    
    print("precision, recall, and f1 score for stroma class.")
    
    dictionaryPlot = dict()
    dictionaryPlot["precision"] = precisionStroma
    dictionaryPlot["recall"] = recallStroma
    dictionaryPlot["f1 score"] = f1ScoreStroma
    
    dataFramePlot = pandas.DataFrame(dictionaryPlot, index = INDEX)
    dataFramePlot.plot()
    plt.xlabel(XLABEL)
    plt.show()
    plt.close()
    
    
    print("precision, recall, and f1 score for blood class.")
    
    dictionaryPlot = dict()
    dictionaryPlot["precision"] = precisionBlood
    dictionaryPlot["recall"] = recallBlood
    dictionaryPlot["f1 score"] = f1ScoreBlood
    
    dataFramePlot = pandas.DataFrame(dictionaryPlot, index = INDEX)
    dataFramePlot.plot()
    plt.xlabel(XLABEL)
    plt.show()
    plt.close()
    
    
    print("precision, recall, and f1 score for necrosis class.")
    
    dictionaryPlot = dict()
    dictionaryPlot["precision"] = precisionNecrosis
    dictionaryPlot["recall"] = recallNecrosis
    dictionaryPlot["f1 score"] = f1ScoreNecrosis
    
    dataFramePlot = pandas.DataFrame(dictionaryPlot, index = INDEX)
    dataFramePlot.plot()
    plt.xlabel(XLABEL)
    plt.show()
    plt.close()
    
    
    print("global accuracy.")
    
    dictionaryPlot = dict()
    dictionaryPlot["accuracy"] = accuracy
    
    dataFramePlot = pandas.DataFrame(dictionaryPlot, index = INDEX)
    dataFramePlot.plot()
    plt.xlabel(XLABEL)
    plt.show()
    plt.close()
    
    
    print("macro average precision, recall, and f1 score.")
    
    dictionaryPlot = dict()
    dictionaryPlot["precision"] = precisionMacroAverage
    dictionaryPlot["recall"] = recallMacroAverage
    dictionaryPlot["f1 score"] = f1ScoreMacroAverage
    
    dataFramePlot = pandas.DataFrame(dictionaryPlot, index = INDEX)
    dataFramePlot.plot()
    plt.xlabel(XLABEL)
    plt.show()
    plt.close()
    
    
    print("weighted average precision, recall, and f1 score.")
    
    dictionaryPlot = dict()
    dictionaryPlot["precision"] = precisionWeightedAverage
    dictionaryPlot["recall"] = recallWeightedAverage
    dictionaryPlot["f1 score"] = f1ScoreWeightedAverage
    
    dataFramePlot = pandas.DataFrame(dictionaryPlot, index = INDEX)
    dataFramePlot.plot()
    plt.xlabel(XLABEL)
    plt.show()
    plt.close()
    
