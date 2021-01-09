import numpy
import scipy.io
import scipy.signal
import scipy.special
import matplotlib.pyplot as plt
import os
import skimage.io
import sklearn.ensemble
import sklearn.discriminant_analysis
import sklearn.decomposition
import sklearn.metrics
import sklearn.linear_model
import sklearn.feature_selection
import sklearn.tree
import pickle
import copy
import imblearn.under_sampling
import imblearn.over_sampling
import imblearn.combine
import imblearn.ensemble
import statsmodels.multivariate.pca
import time
import pandas


def plot_feature_importances(modelPath, permutationImportancesPathTrainingSet, permutationImportancesPathTestSet, useModelFeatureImportances, alphaValue, numberFeaturesToShowBarPlot):
    
    with open(modelPath, "rb") as f:
        modelInformations = pickle.load(f)
    
    waveNumbers = modelInformations.waveNumbers.reshape((-1,))
    
    
    with open(permutationImportancesPathTrainingSet, "rb") as f:
        permutationImportancesTrainingSetInformations = pickle.load(f)
    
    
    with open(permutationImportancesPathTestSet, "rb") as f:
        permutationImportancesTestSetInformations = pickle.load(f)
    
    
    #print("sum reduction of impurity importances: {}".format(modelInformations.model.feature_importances_.sum()))
    #print("sum training set: {}".format(permutationImportancesTrainingSetInformations.importances_mean.sum()))
    #print("sum test set: {}".format(permutationImportancesTestSetInformations.importances_mean.sum()))
    
    
    dictionaryPlotCumulatedImportances = dict()
    
    
    if useModelFeatureImportances:
        
        print("plot of the reduction of impurity importances")
        
        dictionaryPlot = dict()
        dictionaryPlot["reduction of impurity"] = modelInformations.model.feature_importances_
        dataFramePlot = pandas.DataFrame(dictionaryPlot, index = waveNumbers)
        dataFramePlot.plot()
        plt.xlabel("wave numbers")
        plt.ylabel("importance")
        plt.show()
        plt.close()
    
    
    print("plot of the cumulated reduction of impurity importances. The features are sorted from most important to least important.")
    
    featureImportancesSortedIndices = modelInformations.model.feature_importances_.argsort()
    featureImportancesSortedIndicesReversed = featureImportancesSortedIndices[::-1]
    featureImportancesSorted = modelInformations.model.feature_importances_[featureImportancesSortedIndicesReversed]
    featureImportancesSortedCumulativeSum = featureImportancesSorted.cumsum()
    waveNumbersSorted = waveNumbers[featureImportancesSortedIndicesReversed]
    
    #dictionaryPlotCumulatedImportances["reduction of impurity importances"] = featureImportancesSortedCumulativeSum
    
    """
    proportionListIndices = []
    for proportion in numpy.arange(0.0, 1.01, 0.1):
        proportionListIndices.append(round(len(waveNumbers) * proportion - 1))
    """
    
    
    plt.figure()
    #plt.plot(range(len(waveNumbersSorted)), featureImportancesSortedCumulativeSum)
    #plt.xticks(proportionListIndices, numpy.around(numpy.arange(0.0, 1.01, 0.1), 1))
    plt.plot(numpy.arange(1 / len(waveNumbers), 1 + 1 / (2 * len(waveNumbers)), 1 / len(waveNumbers)), featureImportancesSortedCumulativeSum)
    plt.xlabel("Proportion of most important features")
    plt.ylabel("cumulated importance")
    plt.show()
    plt.close()
    
    
    
    i = 0
    while i < 2:
        
        if i == 0:
            importancesMeanTrainingSet = permutationImportancesTrainingSetInformations.importances_mean
            importancesMeanTestSet = permutationImportancesTestSetInformations.importances_mean
        
        elif i == 1:
            importancesMeanTrainingSetWithoutNegative = permutationImportancesTrainingSetInformations.importances_mean
            importancesMeanTrainingSetWithoutNegative[importancesMeanTrainingSetWithoutNegative < 0] = 0
            importancesMeanTrainingSetNormalized = importancesMeanTrainingSetWithoutNegative / numpy.sum(importancesMeanTrainingSetWithoutNegative)
            importancesMeanTrainingSet = importancesMeanTrainingSetNormalized
            
            importancesMeanTestSetWithoutNegative = permutationImportancesTestSetInformations.importances_mean
            importancesMeanTestSetWithoutNegative[importancesMeanTestSetWithoutNegative < 0] = 0
            importancesMeanTestSetNormalized = importancesMeanTestSetWithoutNegative / numpy.sum(importancesMeanTestSetWithoutNegative)
            importancesMeanTestSet = importancesMeanTestSetNormalized
            
        
        
        if i == 0:
            print("plot of the permutations importances")
        
        elif i == 1:
            print("plot of the permutations importances. Permutation importances are normalized.")
        
        dictionaryPlot = dict()
        dictionaryPlot["permutation importance on the training set"] = importancesMeanTrainingSet
        dictionaryPlot["permutation importance on the test set"] = importancesMeanTestSet
        dataFramePlot = pandas.DataFrame(dictionaryPlot, index = waveNumbers)
        dataFramePlot.plot()
        plt.xlabel("wave numbers")
        plt.ylabel("importance")
        plt.show()
        plt.close()
        
        
        if useModelFeatureImportances:
            if i == 0:
                print("plot of the permutations importances and the reduction of impurity importances")
            
            elif i == 1:
                print("plot of the permutations importances and the reduction of impurity importances. Permutation importances are normalized.")
            
            dictionaryPlot = dict()
            dictionaryPlot["permutation importance on the training set"] = importancesMeanTrainingSet
            dictionaryPlot["permutation importance on the test set"] = importancesMeanTestSet
            
            dictionaryPlot["reduction of impurity"] = modelInformations.model.feature_importances_
            
            dataFramePlot = pandas.DataFrame(dictionaryPlot, index = waveNumbers)
            dataFramePlot.plot()
            plt.xlabel("wave numbers")
            plt.ylabel("importance")
            plt.show()
            plt.close()
        
        
        if i == 0:
            print("plot of the permutations importances standard deviations")
            
            dictionaryPlot = dict()
            dictionaryPlot["permutation importance standard deviations on the training set"] = permutationImportancesTrainingSetInformations.importances_std
            dictionaryPlot["permutation importance standard deviations on the test set"] = permutationImportancesTestSetInformations.importances_std
            dataFramePlot = pandas.DataFrame(dictionaryPlot, index = waveNumbers)
            dataFramePlot.plot()
            plt.xlabel("wave numbers")
            plt.ylabel("importance")
            plt.show()
            plt.close()
        
        
        if i == 0:
            print("plot of the permutations importances with standard deviations shown with bands")
            
            dictionaryPlot = dict()
            dictionaryPlot["permutation importance on the training set"] = importancesMeanTrainingSet
            dictionaryPlot["permutation importance on the test set"] = importancesMeanTestSet
            dataFramePlot = pandas.DataFrame(dictionaryPlot, index = waveNumbers)
            dataFramePlot.plot()
            
            plt.fill_between(waveNumbers, importancesMeanTrainingSet - permutationImportancesTrainingSetInformations.importances_std, importancesMeanTrainingSet + permutationImportancesTrainingSetInformations.importances_std, alpha = alphaValue)
            
            plt.fill_between(waveNumbers, importancesMeanTestSet - permutationImportancesTestSetInformations.importances_std, importancesMeanTestSet + permutationImportancesTestSetInformations.importances_std, alpha = alphaValue)
            
            plt.xlabel("wave numbers")
            plt.ylabel("importance")
            plt.show()
            plt.close()
        
        
        if i == 0:
            if useModelFeatureImportances:
                print("plot of the permutations importances with standard deviations shown with bands. The reduction of impurity importances are added.")
                
                dictionaryPlot = dict()
                dictionaryPlot["permutation importance on the training set"] = importancesMeanTrainingSet
                dictionaryPlot["permutation importance on the test set"] = importancesMeanTestSet
                
                dictionaryPlot["reduction of impurity"] = modelInformations.model.feature_importances_
                
                dataFramePlot = pandas.DataFrame(dictionaryPlot, index = waveNumbers)
                dataFramePlot.plot()
                
                plt.fill_between(waveNumbers, importancesMeanTrainingSet - permutationImportancesTrainingSetInformations.importances_std, importancesMeanTrainingSet + permutationImportancesTrainingSetInformations.importances_std, alpha = alphaValue)
                
                plt.fill_between(waveNumbers, importancesMeanTestSet - permutationImportancesTestSetInformations.importances_std, importancesMeanTestSet + permutationImportancesTestSetInformations.importances_std, alpha = alphaValue)
                
                plt.xlabel("wave numbers")
                plt.ylabel("importance")
                plt.show()
                plt.close()
        
        
        if useModelFeatureImportances:
            if i == 0:
                print("bar plot of the {} most important features according to the reduction of impurity feature importances".format(numberFeaturesToShowBarPlot))
            
            elif i == 1:
                print("bar plot of the {} most important features according to the reduction of impurity feature importances. Permutation importances are normalized.".format(numberFeaturesToShowBarPlot))
            
            featureImportancesSortedIndices = modelInformations.model.feature_importances_.argsort()
            featureImportancesSortedIndicesReversed = featureImportancesSortedIndices[::-1]
            featureImportancesSorted = modelInformations.model.feature_importances_[featureImportancesSortedIndicesReversed]
            featureImportancesSortedFiltered = featureImportancesSorted[0:numberFeaturesToShowBarPlot]
            waveNumbersSorted = waveNumbers[featureImportancesSortedIndicesReversed]
            waveNumbersSortedFiltered = waveNumbersSorted[0:numberFeaturesToShowBarPlot]
            
            plt.figure()
            plt.bar(range(len(waveNumbersSortedFiltered)), featureImportancesSortedFiltered, color = "r", align = "center")
            plt.xticks(range(len(waveNumbersSortedFiltered)), waveNumbersSortedFiltered)
            plt.xlim([-1, len(waveNumbersSortedFiltered)])
            plt.show()
            plt.close()
            
            
            if i == 0:
                print("bar plot of the {} least important features according to the reduction of impurity feature importances".format(numberFeaturesToShowBarPlot))
            
            elif i == 1:
                print("bar plot of the {} least important features according to the reduction of impurity feature importances. Permutation importances are normalized.".format(numberFeaturesToShowBarPlot))
            
            featureImportancesSortedIndices = modelInformations.model.feature_importances_.argsort()
            featureImportancesSorted = modelInformations.model.feature_importances_[featureImportancesSortedIndices]
            featureImportancesSortedFiltered = featureImportancesSorted[0:numberFeaturesToShowBarPlot]
            waveNumbersSorted = waveNumbers[featureImportancesSortedIndices]
            waveNumbersSortedFiltered = waveNumbersSorted[0:numberFeaturesToShowBarPlot]
            
            plt.figure()
            plt.bar(range(len(waveNumbersSortedFiltered)), featureImportancesSortedFiltered, color = "r", align = "center")
            plt.xticks(range(len(waveNumbersSortedFiltered)), waveNumbersSortedFiltered)
            plt.xlim([-1, len(waveNumbersSortedFiltered)])
            plt.show()
            plt.close()
            
            
            """
            print("bar plot of the cumulated reduction of impurity importances. The features are sorted from most important to least important.")
            
            featureImportancesSortedIndices = modelInformations.model.feature_importances_.argsort()
            featureImportancesSortedIndicesReversed = featureImportancesSortedIndices[::-1]
            featureImportancesSorted = modelInformations.model.feature_importances_[featureImportancesSortedIndicesReversed]
            featureImportancesSortedCumulativeSum = featureImportancesSorted.cumsum()
            waveNumbersSorted = waveNumbers[featureImportancesSortedIndicesReversed]
            
            proportionListIndices = []
            for proportion in numpy.arange(0.0, 1.01, 0.1):
                proportionListIndices.append(round(len(waveNumbers) * proportion - 1))
            
            
            plt.figure()
            #plt.bar(range(len(waveNumbersSorted)), featureImportancesSortedCumulativeSum, color = "r", align = "center")
            plt.plot(range(len(waveNumbersSorted)), featureImportancesSortedCumulativeSum)
            plt.xticks(proportionListIndices, numpy.around(numpy.arange(0.0, 1.01, 0.1), 1))
            #plt.xlim([-1, len(waveNumbersSorted)])
            plt.show()
            plt.close()
            """
            
        
        if i == 0:
            print("bar plot of the {} most important features according to the permutation importances for the training set".format(numberFeaturesToShowBarPlot))
        
        elif i == 1:
            print("bar plot of the {} most important features according to the permutation importances for the training set. Permutation importances are normalized.".format(numberFeaturesToShowBarPlot))
        
        featureImportancesSortedIndices = importancesMeanTrainingSet.argsort()
        featureImportancesSortedIndicesReversed = featureImportancesSortedIndices[::-1]
        featureImportancesSorted = importancesMeanTrainingSet[featureImportancesSortedIndicesReversed]
        featureImportancesSortedFiltered = featureImportancesSorted[0:numberFeaturesToShowBarPlot]
        waveNumbersSorted = waveNumbers[featureImportancesSortedIndicesReversed]
        waveNumbersSortedFiltered = waveNumbersSorted[0:numberFeaturesToShowBarPlot]
        
        plt.figure()
        plt.bar(range(len(waveNumbersSortedFiltered)), featureImportancesSortedFiltered, color = "r", align = "center")
        plt.xticks(range(len(waveNumbersSortedFiltered)), waveNumbersSortedFiltered)
        plt.xlim([-1, len(waveNumbersSortedFiltered)])
        plt.show()
        plt.close()
        
        
        if i == 0:
            print("bar plot of the {} least important features according to the permutation importances for the training set".format(numberFeaturesToShowBarPlot))
        
        elif i == 1:
            print("bar plot of the {} least important features according to the permutation importances for the training set. Permutation importances are normalized".format(numberFeaturesToShowBarPlot))
        
        featureImportancesSortedIndices = importancesMeanTrainingSet.argsort()
        featureImportancesSorted = importancesMeanTrainingSet[featureImportancesSortedIndices]
        featureImportancesSortedFiltered = featureImportancesSorted[0:numberFeaturesToShowBarPlot]
        waveNumbersSorted = waveNumbers[featureImportancesSortedIndices]
        waveNumbersSortedFiltered = waveNumbersSorted[0:numberFeaturesToShowBarPlot]
        
        plt.figure()
        plt.bar(range(len(waveNumbersSortedFiltered)), featureImportancesSortedFiltered, color = "r", align = "center")
        plt.xticks(range(len(waveNumbersSortedFiltered)), waveNumbersSortedFiltered)
        plt.xlim([-1, len(waveNumbersSortedFiltered)])
        plt.show()
        plt.close()
        
        
        if i == 0:
            print("bar plot of the {} most important features according to the permutation importances for the test set".format(numberFeaturesToShowBarPlot))
        
        elif i == 1:
            print("bar plot of the {} most important features according to the permutation importances for the test set. Permutation importances are normalized.".format(numberFeaturesToShowBarPlot))
        
        featureImportancesSortedIndices = importancesMeanTestSet.argsort()
        featureImportancesSortedIndicesReversed = featureImportancesSortedIndices[::-1]
        featureImportancesSorted = importancesMeanTestSet[featureImportancesSortedIndicesReversed]
        featureImportancesSortedFiltered = featureImportancesSorted[0:numberFeaturesToShowBarPlot]
        waveNumbersSorted = waveNumbers[featureImportancesSortedIndicesReversed]
        waveNumbersSortedFiltered = waveNumbersSorted[0:numberFeaturesToShowBarPlot]
        
        plt.figure()
        plt.bar(range(len(waveNumbersSortedFiltered)), featureImportancesSortedFiltered, color = "r", align = "center")
        plt.xticks(range(len(waveNumbersSortedFiltered)), waveNumbersSortedFiltered)
        plt.xlim([-1, len(waveNumbersSortedFiltered)])
        plt.show()
        plt.close()
        
        
        if i == 0:
            print("bar plot of the {} least important features according to the permutation importances for the test set".format(numberFeaturesToShowBarPlot))
        
        elif i == 1:
            print("bar plot of the {} least important features according to the permutation importances for the test set. Permutation importances are normalized.".format(numberFeaturesToShowBarPlot))
        
        featureImportancesSortedIndices = importancesMeanTestSet.argsort()
        featureImportancesSorted = importancesMeanTestSet[featureImportancesSortedIndices]
        featureImportancesSortedFiltered = featureImportancesSorted[0:numberFeaturesToShowBarPlot]
        waveNumbersSorted = waveNumbers[featureImportancesSortedIndices]
        waveNumbersSortedFiltered = waveNumbersSorted[0:numberFeaturesToShowBarPlot]
        
        plt.figure()
        plt.bar(range(len(waveNumbersSortedFiltered)), featureImportancesSortedFiltered, color = "r", align = "center")
        plt.xticks(range(len(waveNumbersSortedFiltered)), waveNumbersSortedFiltered)
        plt.xlim([-1, len(waveNumbersSortedFiltered)])
        plt.show()
        plt.close()
        
        
        if i == 1:
            print("plot of the cumulated permutation importances on the training set. The features are sorted from most important to least important. Permutation importances are normalized.")
            
            featureImportancesSortedIndices = importancesMeanTrainingSet.argsort()
            featureImportancesSortedIndicesReversed = featureImportancesSortedIndices[::-1]
            featureImportancesSorted = importancesMeanTrainingSet[featureImportancesSortedIndicesReversed]
            featureImportancesSortedCumulativeSum = featureImportancesSorted.cumsum()
            waveNumbersSorted = waveNumbers[featureImportancesSortedIndicesReversed]
            
            dictionaryPlotCumulatedImportances["permutation importances on the training set"] = featureImportancesSortedCumulativeSum
            
            """
            proportionListIndices = []
            for proportion in numpy.arange(0.0, 1.01, 0.1):
                proportionListIndices.append(round(len(waveNumbers) * proportion - 1))
            """
            
            
            plt.figure()
            #plt.plot(range(len(waveNumbersSorted)), featureImportancesSortedCumulativeSum)
            #plt.xticks(proportionListIndices, numpy.around(numpy.arange(0.0, 1.01, 0.1), 1))
            plt.plot(numpy.arange(1 / len(waveNumbers), 1 + 1 / (2 * len(waveNumbers)), 1 / len(waveNumbers)), featureImportancesSortedCumulativeSum)
            plt.xlabel("Proportion of most important features")
            plt.ylabel("cumulated importance")
            plt.show()
            plt.close()
            
            
            print("plot of the cumulated permutation importances on the test set. The features are sorted from most important to least important. Permutation importances are normalized.")
            
            featureImportancesSortedIndices = importancesMeanTestSet.argsort()
            featureImportancesSortedIndicesReversed = featureImportancesSortedIndices[::-1]
            featureImportancesSorted = importancesMeanTestSet[featureImportancesSortedIndicesReversed]
            featureImportancesSortedCumulativeSum = featureImportancesSorted.cumsum()
            waveNumbersSorted = waveNumbers[featureImportancesSortedIndicesReversed]
            
            dictionaryPlotCumulatedImportances["permutation importances on the test set"] = featureImportancesSortedCumulativeSum
            
            """
            proportionListIndices = []
            for proportion in numpy.arange(0.0, 1.01, 0.1):
                proportionListIndices.append(round(len(waveNumbers) * proportion - 1))
            """
            
            
            plt.figure()
            #plt.plot(range(len(waveNumbersSorted)), featureImportancesSortedCumulativeSum)
            #plt.xticks(proportionListIndices, numpy.around(numpy.arange(0.0, 1.01, 0.1), 1))
            plt.plot(numpy.arange(1 / len(waveNumbers), 1 + 1 / (2 * len(waveNumbers)), 1 / len(waveNumbers)), featureImportancesSortedCumulativeSum)
            plt.xlabel("Proportion of most important features")
            plt.ylabel("cumulated importance")
            plt.show()
            plt.close()
            
            
            featureImportancesSortedIndices = modelInformations.model.feature_importances_.argsort()
            featureImportancesSortedIndicesReversed = featureImportancesSortedIndices[::-1]
            featureImportancesSorted = modelInformations.model.feature_importances_[featureImportancesSortedIndicesReversed]
            featureImportancesSortedCumulativeSum = featureImportancesSorted.cumsum()
            waveNumbersSorted = waveNumbers[featureImportancesSortedIndicesReversed]
            
            dictionaryPlotCumulatedImportances["reduction of impurity importances"] = featureImportancesSortedCumulativeSum
            
            
            print("plot of the cumulated reduction of impurity importances and the permutation importances on the training and the test set. The features are sorted from most important to least important. Permutation importances are normalized.")
            
            #dataFramePlot = pandas.DataFrame(dictionaryPlotCumulatedImportances, index = numpy.arange(0, len(waveNumbers), 1))
            dataFramePlot = pandas.DataFrame(dictionaryPlotCumulatedImportances, index = numpy.arange(1 / len(waveNumbers), 1 + 1 / (2 * len(waveNumbers)), 1 / len(waveNumbers)))
            dataFramePlot.plot()
            
            """
            proportionListIndices = []
            for proportion in numpy.arange(0.0, 1.01, 0.1):
                proportionListIndices.append(round(len(waveNumbers) * proportion - 1))
            
            plt.xticks(proportionListIndices, numpy.around(numpy.arange(0.0, 1.01, 0.1), 1))
            """
            
            plt.xlabel("Proportion of most important features")
            plt.ylabel("cumulated importance")
            plt.show()
            plt.close()
        
        
        i += 1




class ModelInformations:
    """
    This class allows to store the preprocessing parameters, the preprocessed dataset, and the trained model and to save them to files.
    
    Variables:
    ----------
    overlayPath: The folder where to find all overlay files.
    dataCubePath: The folder where to find all datacube files, i.e. files where spectra for each pixels are stored.
    kFoldIndicesPath: Path to a file that stores an array of values with one value for each overlay file in the alphabetical order. The value associated to a file allows to assign the file to either the training set or the test set. If the value in the array is 1, the corresponding file is in the test set. Otherwise, it is in the training set.
    colours: The numpy array with 3 columns and a number of rows equal to the number of classes. Each colour represents a class and each row represents a colour. Each row contains an array of 3 values representing the RGB intensities of the colour. The row position of a colour determines the value assigned to the corresponding class.
    peakWaveQualityTest: This corresponds to the peak to detect when using quality test to remove pixels that do not corresponds to tissues. If set to None, the quality test is not done.
    troughWaveQualityTest: This corresponds to the trough to detect when using quality test to remove pixels that do not corresponds to tissues. If set to None, the quality test is not done.
    minimumQualityTest: The minimum absorbance to obtain to pass the quality test. If set to None, the quality test is not done.
    maximumQualityTest: The maximum absorbance to obtain to pass the quality test. If set to None, the quality test is not done.
    noiseReductionFirst: Boolean determining the order in which the noise reduction and the range selection operations must be done. True means noise reduction is done first. False means range selection is done first.
    methodNoiseReduction: A string indicating the algorithm to use for noise reduction. Possible values are: sklearn_pca, sklearn_kernel_pca, sklearn_sparse_pca, sklearn_truncated_svd, sklearn_incremental_pca, sklearn_mini_batch_sparse_pca, statsmodels_pca, bassan_nipals. None value is also possible and means no noise reduction if numberPrincipalComponents is also None. Otherwise, an exception is raised.
    numberPrincipalComponents: Number of principal components to retain when performing principal components based noise reduction. If set to None, the noise reduction is not done.
    inverseTransform: boolean telling if the noise reduction must get back the original data denoised. True means to come back to the original features, False means to keep the features obtained by principal component analysis. If set to False, range selection and savitzky golay filter cannot be done, i.e. intervalsToSelect must be None and smoothingPoints or orderPolynomialFit or both must be None.
    additionalParametersNoiseReduction: a dictionary providing additional parameters necessary for the noise reduction method selected.
    pcaClass: The class doing principal component analysis necessary for noise reduction.
    intervalsToSelect: The intervals to keep, the rest is removed. If set to None, the range selection is not done.
    normalise: Boolean telling whether to normalise the spectra or not.
    methodSavitzkyGolayFilter: A string indicating the algorithm to use for Savitzky-Golay filter. Possible values are: savitzky_golay_filter, first_derivative. None value is also possible and means no Savitzky-Golay filter if smoothingPoints and orderPolynomialFit are also None. Otherwise, an exception is raised.
    smoothingPoints: The number of smoothing points to consider when computing the first derivative by performing Savitzky-Golay smoothing. If set to None, the first derivative is not done.
    orderPolynomialFit: The order of the polynomial used to fit when computing the first derivative by performing Savitzky-Golay smoothing. If set to None, the first derivative is not done.
    additionalParametersSavitzkyGolayFilter: a dictionary providing additional parameters necessary for the Savitzky-Golay filter method selected.
    dictionaryClassesMapping: The set where the keys are tuples representing the classes that must be mapped to the class given by their respective values.
    methodClassesEqualization: A string indicating the algorithm to use for classes equalization. Possible values are: random_under_sampler, random_over_sampler, tomek_links, cluster_centroids, smote, smote_tomek. None value is also possible and means no classes equalization.
    additionalParametersClassesEqualization: a dictionary providing additional parameters necessary for the classes equalization method selected.
    setInput: The spectra of all annotated pixels. Each row corresponds to one spectra.
    setOutput: The histological classes associated with the spectra whose row index in setInput is equal to the index in setOutput.
    waveNumbers: The list of wave numbers where the absorbance has been measured.
    model: The learning algorithm used already fitted.
    """
    
    
    def __init__(self, overlayPath, dataCubePath, kFoldIndicesPath, colours, peakWaveQualityTest, troughWaveQualityTest, minimumQualityTest, maximumQualityTest, noiseReductionFirst, methodNoiseReduction, numberPrincipalComponents, inverseTransform, additionalParametersNoiseReduction, pcaClass, intervalsToSelect, normalise, methodSavitzkyGolayFilter, smoothingPoints, orderPolynomialFit, additionalParametersSavitzkyGolayFilter, dictionaryClassesMapping, methodClassesEqualization, additionalParametersClassesEqualization, methodFeatureSelection, additionalParametersFeatureSelection, model, modelAlgorithm, seed, waveNumbers, widthSubwindow, heightSubwindow, methodSubwindow, uniqueValuesTrainingSetOutput, uniqueValuesTrainingSetOutputCounts, makeQualityTestTrainingSet, makeQualityTestTestSet, makeMirrors, makeRotations):
        """
        This method initializes the class ModelInformations.
        
        parameters:
        -----------
        overlayPath: The folder where to find all overlay files.
        dataCubePath: The folder where to find all datacube files, i.e. files where spectra for each pixels are stored.
        kFoldIndicesPath: Path to a file that stores an array of values with one value for each overlay file in the alphabetical order. The value associated to a file allows to assign the file to either the training set or the test set. If the value in the array is 1, the corresponding file is in the test set. Otherwise, it is in the training set.
        colours: The numpy array with 3 columns and a number of rows equal to the number of classes. Each colour represents a class and each row represents a colour. Each row contains an array of 3 values representing the RGB intensities of the colour. The row position of a colour determines the value assigned to the corresponding class.
        peakWaveQualityTest: This corresponds to the peak to detect when using quality test to remove pixels that do not corresponds to tissues. If set to None, the quality test is not done.
        troughWaveQualityTest: This corresponds to the trough to detect when using quality test to remove pixels that do not corresponds to tissues. If set to None, the quality test is not done.
        minimumQualityTest: The minimum absorbance to obtain to pass the quality test. If set to None, the quality test is not done.
        maximumQualityTest: The maximum absorbance to obtain to pass the quality test. If set to None, the quality test is not done.
        noiseReductionFirst: Boolean determining the order in which the noise reduction and the range selection operations must be done. True means noise reduction is done first. False means range selection is done first.
        methodNoiseReduction: A string indicating the algorithm to use for noise reduction. Possible values are: sklearn_pca, sklearn_kernel_pca, sklearn_sparse_pca, sklearn_truncated_svd, sklearn_incremental_pca, sklearn_mini_batch_sparse_pca, statsmodels_pca, bassan_nipals. None value is also possible and means no noise reduction if numberPrincipalComponents is also None. Otherwise, an exception is raised.
        numberPrincipalComponents: Number of principal components to retain when performing principal components based noise reduction. If set to None, the noise reduction is not done.
        inverseTransform: boolean telling if the noise reduction must get back the original data denoised. True means to come back to the original features, False means to keep the features obtained by principal component analysis. If set to False, range selection and savitzky golay filter cannot be done, i.e. intervalsToSelect must be None and smoothingPoints or orderPolynomialFit or both must be None.
        additionalParametersNoiseReduction: a dictionary providing additional parameters necessary for the noise reduction method selected.
        pcaClass: The class doing principal component analysis necessary for noise reduction.
        intervalsToSelect: The intervals to keep, the rest is removed. If set to None, the range selection is not done.
        normalise: Boolean telling whether to normalise the spectra or not.
        methodSavitzkyGolayFilter: A string indicating the algorithm to use for Savitzky-Golay filter. Possible values are: savitzky_golay_filter, first_derivative. None value is also possible and means no Savitzky-Golay filter if smoothingPoints and orderPolynomialFit are also None. Otherwise, an exception is raised.
        smoothingPoints: The number of smoothing points to consider when computing the first derivative by performing Savitzky-Golay smoothing. If set to None, the first derivative is not done.
        orderPolynomialFit: The order of the polynomial used to fit when computing the first derivative by performing Savitzky-Golay smoothing. If set to None, the first derivative is not done.
        additionalParametersSavitzkyGolayFilter: a dictionary providing additional parameters necessary for the Savitzky-Golay filter method selected.
        dictionaryClassesMapping: The set where the keys are tuples representing the classes that must be mapped to the class given by their respective values.
        methodClassesEqualization: A string indicating the algorithm to use for classes equalization. Possible values are: random_under_sampler, random_over_sampler, tomek_links, cluster_centroids, smote, smote_tomek. None value is also possible and means no classes equalization.
        additionalParametersClassesEqualization: a dictionary providing additional parameters necessary for the classes equalization method selected.
        setInput: The spectra of all annotated pixels. Each row corresponds to one spectra.
        setOutput: The histological classes associated with the spectra whose row index in setInput is equal to the index in setOutput.
        waveNumbers: The list of wave numbers where the absorbance has been measured.
        model: The learning algorithm used already fitted.
        """
        
        
        self.overlayPath = os.path.abspath(overlayPath)
        self.dataCubePath = os.path.abspath(dataCubePath)
        self.kFoldIndicesPath = os.path.abspath(kFoldIndicesPath)
        self.colours = colours
        self.peakWaveQualityTest = peakWaveQualityTest
        self.troughWaveQualityTest = troughWaveQualityTest
        self.minimumQualityTest = minimumQualityTest
        self.maximumQualityTest = maximumQualityTest
        self.noiseReductionFirst = noiseReductionFirst
        self.methodNoiseReduction = methodNoiseReduction
        self.numberPrincipalComponents = numberPrincipalComponents
        self.inverseTransform = inverseTransform
        self.additionalParametersNoiseReduction = additionalParametersNoiseReduction
        self.pcaClass = pcaClass
        self.intervalsToSelect = intervalsToSelect
        self.normalise = normalise
        self.methodSavitzkyGolayFilter = methodSavitzkyGolayFilter
        self.smoothingPoints = smoothingPoints
        self.orderPolynomialFit = orderPolynomialFit
        self.additionalParametersSavitzkyGolayFilter = additionalParametersSavitzkyGolayFilter
        self.dictionaryClassesMapping = dictionaryClassesMapping
        self.methodClassesEqualization = methodClassesEqualization
        self.additionalParametersClassesEqualization = additionalParametersClassesEqualization
        self.methodFeatureSelection = methodFeatureSelection
        self.additionalParametersFeatureSelection = additionalParametersFeatureSelection
        self.model = model
        self.modelAlgorithm = modelAlgorithm
        self.seed = seed
        self.waveNumbers = waveNumbers
        self.widthSubwindow = widthSubwindow
        self.heightSubwindow = heightSubwindow
        self.methodSubwindow = methodSubwindow
        self.uniqueValuesTrainingSetOutput = uniqueValuesTrainingSetOutput
        self.uniqueValuesTrainingSetOutputCounts = uniqueValuesTrainingSetOutputCounts
        self.makeQualityTestTrainingSet = makeQualityTestTrainingSet
        self.makeQualityTestTestSet = makeQualityTestTestSet
        self.makeMirrors = makeMirrors
        self.makeRotations = makeRotations
    
    
    def __str__(self):
        """
        This method creates a string representation of the preprocessing parameters and the trained model parameters.
        
        returns:
        --------
        The string representation of the preprocessing parameters and the trained model parameters.
        """
        
        string = "seed: " + str(self.seed) + "\n"
        string += "overlayPath: " + self.overlayPath + "\n"
        string += "dataCubePath: " + self.dataCubePath + "\n"
        string += "kFoldIndicesPath: " + self.kFoldIndicesPath + "\n"
        string += "colours: \n" + str(self.colours) + "\n"
        string += "peakWaveQualityTest: " + str(self.peakWaveQualityTest) + "\n"
        string += "troughWaveQualityTest: " + str(self.troughWaveQualityTest) + "\n"
        string += "minimumQualityTest: " + str(self.minimumQualityTest) + "\n"
        string += "maximumQualityTest: " + str(self.maximumQualityTest) + "\n"
        string += "makeQualityTestTrainingSet: " + str(self.makeQualityTestTrainingSet) + "\n"
        string += "makeQualityTestTestSet: " + str(self.makeQualityTestTestSet) + "\n"
        string += "noiseReductionFirst: " + str(self.noiseReductionFirst) + "\n"
        string += "methodNoiseReduction: " + str(self.methodNoiseReduction) + "\n"
        string += "numberPrincipalComponents: " + str(self.numberPrincipalComponents) + "\n"
        string += "inverseTransform: " + str(self.inverseTransform) + "\n"
        
        string += "\nadditionalParametersNoiseReduction:\n"
        
        for key, value in self.additionalParametersNoiseReduction.items():
            string += key + ": " + str(value) + "\n"
        
        
        if self.pcaClass is None:
            string += "\npcaClass informations: None\n"
        
        else:
            string += "\npcaClass informations:\n"
            
            for key, value in self.pcaClass.get_params().items():
                string += key + ": " + str(value) + "\n"
        
        
        string += "\nintervalsToSelect: \n" + str(self.intervalsToSelect) + "\n"
        string += "normalise: " + str(self.normalise) + "\n"
        string += "methodSavitzkyGolayFilter: " + str(self.methodSavitzkyGolayFilter) + "\n"
        string += "smoothingPoints: " + str(self.smoothingPoints) + "\n"
        string += "orderPolynomialFit: " + str(self.orderPolynomialFit) + "\n"
        
        string += "\nadditionalParametersSavitzkyGolayFilter:\n"
        
        for key, value in self.additionalParametersSavitzkyGolayFilter.items():
            string += key + ": " + str(value) + "\n"
        
        
        string += "\ndictionaryClassesMapping: \n" + str(self.dictionaryClassesMapping) + "\n"
        string += "methodClassesEqualization: " + str(self.methodClassesEqualization) + "\n"
        
        string += "\nadditionalParametersClassesEqualization:\n"
        
        for key, value in self.additionalParametersClassesEqualization.items():
            string += key + ": " + str(value) + "\n"
        
        
        if self.methodFeatureSelection is None:
            string += "\nmethodFeatureSelection: None\n"
        
        else:
            string += "\nmethodFeatureSelection: " + str(self.methodFeatureSelection) + "\n"
            
            string += "\nadditionalParametersFeatureSelection:\n"
            
            for key, value in self.additionalParametersFeatureSelection.items():
                string += key + ": " + str(value) + "\n"
        
        
        string += "\nwaveNumbers:\n"
        string += str(self.waveNumbers.reshape((-1,))) + "\n"
        
        
        string += "\nmodel: " + str(self.modelAlgorithm) + "\n"
        
        if self.model is None:
            string += "\nlearning model informations: None\n"
        
        else:
            string += "\nlearning model informations:\n"
            
            for key, value in self.model.get_params().items():
                string += key + ": " + str(value) + "\n"
        
        
        if self.widthSubwindow is None:
            string += "\nwidthSubwindow: None\n"
        
        else:
            string += "\nwidthSubwindow: " + str(self.widthSubwindow) + "\n"
        
        
        if self.heightSubwindow is None:
            string += "heightSubwindow: None\n"
        
        else:
            string += "heightSubwindow: " + str(self.heightSubwindow) + "\n"
        
        
        if self.methodSubwindow is None:
            string += "methodSubwindow: None\n"
        
        else:
            string += "methodSubwindow: " + str(self.methodSubwindow) + "\n"
        
        
        string += "\nclass distribution after potential subwindows creations:\n"
        
        for value, count in zip(self.uniqueValuesTrainingSetOutput, self.uniqueValuesTrainingSetOutputCounts):
            string += "class number: {}, number elements: {}\n".format(value, count)
        
        string += "total number elements: {}\n\n".format(self.uniqueValuesTrainingSetOutputCounts.sum())
        
        string += "makeMirrors: {}\n".format(str(self.makeMirrors))
        string += "makeRotations: {}\n\n".format(str(self.makeRotations))
        
        return string
    
    
    def save_to_pickle_file(self, fileName):
        """
        This method saves an instance of the ModelInformations class to a pickle file.
        
        parameter:
        -----------
        fileName: The file where the instance of the class ModelInformations must be stored.
        """
        
        
        if os.path.dirname(fileName) != "":
            os.makedirs(os.path.dirname(fileName), exist_ok = True)
        
        
        with open(fileName, "wb") as f:
            pickle.dump(self, f)
    
    
    def save_parameters_to_text_file(self, fileName):
        """
        This method saves the string representation of the preprocessing parameters and the trained model parameters to a text file.
        
        parameter:
        ----------
        fileName: The file where the string representation of the preprocessing parameters and the trained model parameters must be stored.
        """
        
        
        string = self.__str__()
        
        if os.path.dirname(fileName) != "":
            os.makedirs(os.path.dirname(fileName), exist_ok = True)
        
        with open(fileName, "w") as f:
            f.write(string)
    
    
    def save_both_to_pickle_and_to_text_file(self, fileName):
        """
        This method saves both an instance of the ModelInformations class to a pickle file and the string representation of the preprocessing parameters and the trained model parameters to a text file. The .pkl extension is added for the pickle file and the .txt extension is added for the text file. If there is already an extension in fileName string, the extension is removed.
        
        parameter:
        -----------
        fileName: The files (one for the pickle file, the other for the human-readable text file) where the preprocessing parameters, the preprocessed dataset, and the trained model are stored. If there is already an extension in fileName string, the extension is removed.
        """
        
        
        baseName = os.path.splitext(fileName)[0]
        
        self.save_to_pickle_file(baseName + ".pkl")
        self.save_parameters_to_text_file(baseName + ".txt")




if __name__ == "__main__":
    
    MODELPATH = "./histology_classification_models/0.pkl"
    PERMUTATIONIMPORTANCESPATHTRAININGSET = "results_model_analysis/0_training_set.pkl"
    PERMUTATIONIMPORTANCESPATHTESTSET = "results_model_analysis/0_test_set.pkl"
    USEMODELFEATUREIMPORTANCES = True
    ALPHAVALUE = 0.1
    NUMBERFEATURESTOSHOWBARPLOT = 20
    
    
    plot_feature_importances(MODELPATH, PERMUTATIONIMPORTANCESPATHTRAININGSET, PERMUTATIONIMPORTANCESPATHTESTSET, USEMODELFEATUREIMPORTANCES, ALPHAVALUE, NUMBERFEATURESTOSHOWBARPLOT)

