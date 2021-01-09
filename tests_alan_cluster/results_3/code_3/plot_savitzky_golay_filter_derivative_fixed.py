import os
import numpy
import pandas
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
    SMOOTHINGPOINTSLIST = numpy.arange(3, 20, 2)
    ORDERPOLYNOMIALFITLIST = numpy.arange(0, 6, 1)
    ADDITIONALPARAMETERSSAVITZKYGOLAYFILTERLIST = [{"deriv" : 0, "mode" : "constant", "cval" : 0.0}, {"deriv" : 1, "mode" : "constant", "cval" : 0.0}, {"deriv" : 2, "mode" : "constant", "cval" : 0.0}, {"deriv" : 3, "mode" : "constant", "cval" : 0.0}, {"deriv" : 4, "mode" : "constant", "cval" : 0.0}, {"deriv" : 5, "mode" : "constant", "cval" : 0.0}]
    ORDERDERIVATIVEFIXED = 2
    METRICFUNCTION = get_f1_score_weighted_average
    
    
    
    smoothingPointsPlot = []
    orderPolynomialFitPlot = []
    orderDerivativePlot = []
    metricValues = []
    
    idModel = 0
    escape = False
    
    for smoothingPoints in SMOOTHINGPOINTSLIST:
        for orderPolynomialFit in ORDERPOLYNOMIALFITLIST:
            for additionalParametersSavitzkyGolayFilter in ADDITIONALPARAMETERSSAVITZKYGOLAYFILTERLIST:
                
                if ((smoothingPoints % 2) == 0) or (orderPolynomialFit >= smoothingPoints):
                    escape = True
                
                elif "deriv" in additionalParametersSavitzkyGolayFilter:
                    if additionalParametersSavitzkyGolayFilter["deriv"] > orderPolynomialFit:
                        escape = True
                
                
                if escape:
                    escape = False
                    idModel += 1
                    continue
                
                
                
                if additionalParametersSavitzkyGolayFilter["deriv"] == ORDERDERIVATIVEFIXED:
                    smoothingPointsPlot.append(smoothingPoints)
                    orderPolynomialFitPlot.append(orderPolynomialFit)
                    orderDerivativePlot.append(additionalParametersSavitzkyGolayFilter["deriv"])
                    
                    
                    dataFrame = pandas.read_csv(os.path.join(FOLDERDATAFRAMES, str(idModel) + ".csv"))
                    
                    metricValues.append(METRICFUNCTION(dataFrame))
                
                
                
                idModel += 1
    
    
    
    dictionaryDataFrame = dict()
    dictionaryDataFrame["smoothing_points"] = smoothingPointsPlot
    dictionaryDataFrame["order_polynomial_fit"] = orderPolynomialFitPlot
    dictionaryDataFrame["metric_values"] = metricValues
    dataFramePlot = pandas.DataFrame(dictionaryDataFrame)
    
    dataFramePlot.plot.scatter(x = "smoothing_points", y = "order_polynomial_fit", c = "metric_values", cmap = "viridis")
    
    plt.xlabel("number of smoothing points")
    plt.ylabel("polynomial order")
    
    plt.gcf().get_axes()[1].set_ylabel("")
    
    plt.show()
    plt.close()
    
