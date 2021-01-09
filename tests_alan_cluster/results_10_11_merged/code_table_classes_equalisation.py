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
    ALGORITHMLIST = ["no resamplers", "cluster centroids", "cluster centroids", "condensed nearest neighbour", "condensed nearest neighbour", "edited nearest neighbours", "edited nearest neighbours", "edited nearest neighbours", "edited nearest neighbours", "instance hardness threshold", "near miss", "near miss", "near miss", "random under sampler", "tomek links", "tomek links", "borderline SMOTE", "borderline SMOTE", "random over sampler", "SMOTE", "SMOTE edited nearest neighbours", "SMOTE edited nearest neighbours", "SMOTE edited nearest neighbours", "SMOTE edited nearest neighbours", "SMOTE tomek", "SMOTE tomek"]
    PARAMETERSLIST = ["/", "soft voting strategy", "hard voting strategy", "resample all classes but the minority class", "resample all classes", "\makecell{resample all classes but the minority class \\\\ all strategy}", "\makecell{resample all classes but the minority class \\\\ mode strategy}", "\makecell{resample all classes \\\\ all strategy}", "\makecell{resample all classes \\\\ mode strategy}", "\makecell{random forest with 500 trees, \\\\ 15 features tested at each node and \\\\ the leafs have 10 samples minimum}", "version 1", "version 2", "version 3", "/", "resample all classes but the minority class", "resample all classes", "borderline-1 algorithm", "borderline-2 algorithm", "/", "/", "\makecell{resample all classes but the minority class \\\\ all strategy}", "\makecell{resample all classes but the minority class \\\\ mode strategy}", "\makecell{resample all classes \\\\ all strategy}", "\makecell{resample all classes \\\\ mode strategy}", "resample all classes but the minority class", "resample all classes"]
    
    IDSTOTAKE = numpy.arange(0, 26)
    FOLDERDATAFRAMES = "./histology_classification_data_frames"
    
    i = 0
    stringResult = "\\hline\n"
    for identifier in IDSTOTAKE:
        dataFrame = pandas.read_csv(os.path.join(FOLDERDATAFRAMES, str(identifier) + ".csv"))
        
        stringResult += ALGORITHMLIST[i] + " & " + PARAMETERSLIST[i] + " & " + str(round(get_f1_score_epithelium(dataFrame) * 100, 2)) + " & " + str(round(get_f1_score_stroma(dataFrame) * 100, 2)) + " & " + str(round(get_f1_score_blood(dataFrame) * 100, 2)) + " & " + str(round(get_f1_score_necrosis(dataFrame) * 100, 2)) + " & " + str(round(get_f1_score_weighted_average(dataFrame) * 100, 2)) + " \\\\\n\\hline\n"
        
        i += 1
    
    print(stringResult)

