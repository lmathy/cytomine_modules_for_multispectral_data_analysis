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



def model_for_identifying_histology(overlayPath, dataCubePath, kFoldIndicesPath, colours, peakWaveQualityTest, troughWaveQualityTest, minimumQualityTest, maximumQualityTest, noiseReductionFirstList, noiseReductionList, numberPrincipalComponentsList, intervalsToSelectList, normaliseList, methodSavitzkyGolayFilterList, smoothingPointsList, orderPolynomialFitList, additionalParametersSavitzkyGolayFilterList, dictionaryClassesMapping, classesEqualizationList, featureSelectionList, modelAlgorithmList, testParametersList, fileNameText, folderDataFrame, seed):
    """
    This function extract the histology classes for different pixels in overlay files. Then, the corresponding spectra of these pixels are extracted in order to form the dataset. Finally, a model is trained in order to detect the histology classes.
    
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
    intervalsToSelect: The intervals to keep, the rest is removed. If set to None, the range selection is not done.
    normalise: Boolean telling whether to normalise the spectra or not.
    methodSavitzkyGolayFilter: A string indicating the algorithm to use for Savitzky-Golay filter. Possible values are: savitzky_golay_filter, first_derivative. None value is also possible and means no Savitzky-Golay filter if smoothingPoints and orderPolynomialFit are also None. Otherwise, an exception is raised.
    smoothingPoints: The number of smoothing points to consider when computing the first derivative by performing Savitzky-Golay smoothing. If set to None, the first derivative is not done.
    orderPolynomialFit: The order of the polynomial used to fit when computing the first derivative by performing Savitzky-Golay smoothing. If set to None, the first derivative is not done.
    additionalParametersSavitzkyGolayFilter: a dictionary providing additional parameters necessary for the Savitzky-Golay filter method selected.
    dictionaryClassesMapping: The set where the keys are tuples representing the classes that must be mapped to the class given by their respective values.
    methodClassesEqualization: A string indicating the algorithm to use for classes equalization. Possible values are: random_under_sampler, random_over_sampler, tomek_links, cluster_centroids, smote, smote_tomek. None value is also possible and means no classes equalization.
    additionalParametersClassesEqualization: a dictionary providing additional parameters necessary for the classes equalization method selected.
    model: The learning algorithm used.
    acceptanceThreshold: The minimum threshold that must be reached in order to accept a classification for one pixel. It means that a minimum proportions of estimators must agree on the class of a given pixel. If it is not the case, the classification for the pixel is not considered as reliable and is rejected.
    fileName: The files (one for the pickle file, the other for the human-readable text file) where the preprocessing parameters, the preprocessed dataset, and the trained model are stored. The .pkl extension is added for the pickle file and the .txt extension is added for the text file.
    """
    
    
    if os.path.dirname(fileNameText) != "":
        os.makedirs(os.path.dirname(fileNameText), exist_ok = True)
    
    
    if (folderDataFrame != ".") and (folderDataFrame != ""):
        os.makedirs(folderDataFrame, exist_ok = True)
    
    
    
    trainSetNames, validationSetNames, testSetNames = get_overlay_files_list_and_separate_overlay_files_training_validation_test_set(overlayPath, kFoldIndicesPath)
    
    trainSetNames += validationSetNames
    
    
    spectraTrainingSet, waveNumbers, trainingSetInput, trainingSetOutput = create_dataset(overlayPath, dataCubePath, trainSetNames, colours)
    
    
    print("merging classes in progress")
    
    trainingSetOutput = merge_classes(trainingSetOutput, dictionaryClassesMapping)
    
    
    trainingSetInput, trainingSetOutput = quality_test(trainingSetInput, trainingSetOutput, waveNumbers, peakWaveQualityTest, troughWaveQualityTest, minimumQualityTest, maximumQualityTest)
    
    
    
    
    spectraTestSet, waveNumbers, testSetInput, testSetOutput = create_dataset(overlayPath, dataCubePath, testSetNames, colours)
    
    print("merging classes in progress")
    
    testSetOutput = merge_classes(testSetOutput, dictionaryClassesMapping)
    
    
    testSetInput, testSetOutput = quality_test(testSetInput, testSetOutput, waveNumbers, peakWaveQualityTest, troughWaveQualityTest, minimumQualityTest, maximumQualityTest)
    
    
    
    trainingSetInputCopy = copy.deepcopy(trainingSetInput)
    trainingSetOutputCopy = copy.deepcopy(trainingSetOutput)
    testSetInputCopy = copy.deepcopy(testSetInput)
    testSetOutputCopy = copy.deepcopy(testSetOutput)
    waveNumbersCopy = copy.deepcopy(waveNumbers)
    
    idModel = 0
    escape = False
    
    with open(fileNameText, "w") as fileText:
        
        for noiseReductionFirst in noiseReductionFirstList:
            for methodNoiseReduction, inverseTransform, additionalParametersNoiseReduction in noiseReductionList:
                for numberPrincipalComponents in numberPrincipalComponentsList:
                    for intervalsToSelect in intervalsToSelectList:
                        for normalise in normaliseList:
                            for methodSavitzkyGolayFilter in methodSavitzkyGolayFilterList:
                                for smoothingPoints in smoothingPointsList:
                                    for orderPolynomialFit in orderPolynomialFitList:
                                        for additionalParametersSavitzkyGolayFilter in additionalParametersSavitzkyGolayFilterList:
                                            for methodClassesEqualization, additionalParametersClassesEqualization in classesEqualizationList:
                                                for methodFeatureSelection, additionalParametersFeatureSelection in featureSelectionList:
                                                    for modelAlgorithm, additionalParametersModelAlgorithm in modelAlgorithmList:
                                                        
                                                        if ((smoothingPoints % 2) == 0) or (orderPolynomialFit >= smoothingPoints):
                                                            escape = True
                                                        
                                                        elif "deriv" in additionalParametersSavitzkyGolayFilter:
                                                            if additionalParametersSavitzkyGolayFilter["deriv"] > orderPolynomialFit:
                                                                escape = True
                                                        
                                                        
                                                        if escape:
                                                            modelInformations = ModelInformations(overlayPath, dataCubePath, kFoldIndicesPath, colours, peakWaveQualityTest, troughWaveQualityTest, minimumQualityTest, maximumQualityTest, noiseReductionFirst, methodNoiseReduction, numberPrincipalComponents, inverseTransform, additionalParametersNoiseReduction, None, intervalsToSelect, normalise, methodSavitzkyGolayFilter, smoothingPoints, orderPolynomialFit, additionalParametersSavitzkyGolayFilter, dictionaryClassesMapping, methodClassesEqualization, additionalParametersClassesEqualization, methodFeatureSelection, additionalParametersFeatureSelection, None, modelAlgorithm, seed, waveNumbers)
                                                            
                                                            
                                                            fileText.write("idModel: {}\n".format(idModel))
                                                            fileText.write(str(modelInformations))
                                                            fileText.write("\n\n")
                                                            
                                                            escape = False
                                                            idModel += 1
                                                            
                                                            continue
                                                        
                                                        
                                                        
                                                        trainingSetInput = copy.deepcopy(trainingSetInputCopy)
                                                        trainingSetOutput = copy.deepcopy(trainingSetOutputCopy)
                                                        testSetInput = copy.deepcopy(testSetInputCopy)
                                                        testSetOutput = copy.deepcopy(testSetOutputCopy)
                                                        waveNumbers = copy.deepcopy(waveNumbersCopy)
                                                        
                                                        
                                                        trainingSetInput, trainingSetOutput, waveNumbers, pcaClass = make_preprocessing(trainingSetInput, trainingSetOutput, waveNumbers, noiseReductionFirst, methodNoiseReduction, numberPrincipalComponents, inverseTransform, additionalParametersNoiseReduction, intervalsToSelect, normalise, methodSavitzkyGolayFilter, smoothingPoints, orderPolynomialFit, additionalParametersSavitzkyGolayFilter)
                                                        
                                                        
                                                        print("classes equalization in progress")
                                                        
                                                        trainingSetInput, trainingSetOutput = classes_equalization(trainingSetInput, trainingSetOutput, methodClassesEqualization, additionalParametersClassesEqualization)
                                                        
                                                        
                                                        if not (methodFeatureSelection is None):
                                                            print("feature selection in progress")
                                                            
                                                            trainingSetInput, featureSelector = feature_selection(trainingSetInput, trainingSetOutput, methodFeatureSelection, additionalParametersFeatureSelection)
                                                            
                                                            trainingSetInput = featureSelector.transform(trainingSetInput)
                                                            waveNumbers = waveNumbers[featureSelector.get_support()]
                                                        
                                                        
                                                        print("training the model")
                                                        
                                                        model = modelAlgorithm(**additionalParametersModelAlgorithm)
                                                        model.fit(trainingSetInput, trainingSetOutput)
                                                        
                                                        
                                                        waveNumbers = copy.deepcopy(waveNumbersCopy)
                                                        
                                                        
                                                        testSetInput, testSetOutput, waveNumbers, pcaClass = make_preprocessing(testSetInput, testSetOutput, waveNumbers, noiseReductionFirst, "reuse", numberPrincipalComponents, inverseTransform, {"pca" : pcaClass}, intervalsToSelect, normalise, methodSavitzkyGolayFilter, smoothingPoints, orderPolynomialFit, additionalParametersSavitzkyGolayFilter)
                                                        
                                                        
                                                        
                                                        if not (methodFeatureSelection is None):
                                                            testSetInput = featureSelector.transform(testSetInput)
                                                            waveNumbers = waveNumbers[featureSelector.get_support()]
                                                        
                                                        
                                                        
                                                        for methodPrediction, acceptanceThreshold in testParametersList:
                                                            
                                                            print("testing the model")
                                                            
                                                            testSetInputClassified, testSetOutputClassified, indexesPredicted, testSetInputRejected, testSetOutputRejected, indexesPredictedAllEstimators = predict_with_acceptance_threshold(model, testSetInput, testSetOutput, acceptanceThreshold, methodPrediction)
                                                            
                                                            
                                                            modelInformations = ModelInformations(overlayPath, dataCubePath, kFoldIndicesPath, colours, peakWaveQualityTest, troughWaveQualityTest, minimumQualityTest, maximumQualityTest, noiseReductionFirst, methodNoiseReduction, numberPrincipalComponents, inverseTransform, additionalParametersNoiseReduction, pcaClass, intervalsToSelect, normalise, methodSavitzkyGolayFilter, smoothingPoints, orderPolynomialFit, additionalParametersSavitzkyGolayFilter, dictionaryClassesMapping, methodClassesEqualization, additionalParametersClassesEqualization, methodFeatureSelection, additionalParametersFeatureSelection, model, modelAlgorithm, seed, waveNumbers)
                                                            
                                                            
                                                            fileText.write("idModel: {}\n".format(idModel))
                                                            fileText.write(str(modelInformations))
                                                            
                                                            fileText.write(sklearn.metrics.classification_report(testSetOutputClassified, indexesPredicted))
                                                            
                                                            
                                                            fileText.write("\nconfusion matrix not normalized:")
                                                            
                                                            confusionMatrix = sklearn.metrics.confusion_matrix(testSetOutputClassified, indexesPredicted, normalize = None)
                                                            
                                                            fileText.write("\n")
                                                            fileText.write(str(confusionMatrix))
                                                            fileText.write("\n")
                                                            
                                                            
                                                            fileText.write("confusion matrix normalized according to the true classes:")
                                                            
                                                            confusionMatrix = sklearn.metrics.confusion_matrix(testSetOutputClassified, indexesPredicted, normalize = "true")
                                                            
                                                            fileText.write("\n")
                                                            fileText.write(str(confusionMatrix))
                                                            fileText.write("\n")
                                                            
                                                            
                                                            fileText.write("confusion matrix normalized according to the predicted classes:")
                                                            
                                                            confusionMatrix = sklearn.metrics.confusion_matrix(testSetOutputClassified, indexesPredicted, normalize = "pred")
                                                            
                                                            fileText.write("\n")
                                                            fileText.write(str(confusionMatrix))
                                                            fileText.write("\n")
                                                            
                                                            
                                                            fileText.write("confusion matrix normalized according to the whole population:")
                                                            
                                                            confusionMatrix = sklearn.metrics.confusion_matrix(testSetOutputClassified, indexesPredicted, normalize = "all")
                                                            
                                                            fileText.write("\n")
                                                            fileText.write(str(confusionMatrix))
                                                            fileText.write("\n\n\n")
                                                            
                                                            
                                                            
                                                            dataFrame = pandas.DataFrame(sklearn.metrics.classification_report(testSetOutputClassified, indexesPredicted, output_dict = True))
                                                            dataFrame.to_csv(os.path.join(folderDataFrame, str(idModel)) + ".csv")
                                                            
                                                            
                                                            idModel += 1




def get_overlay_files_list_and_separate_overlay_files_training_validation_test_set(overlayPath, kFoldIndicesPath):
    """
    This function gets the list of overlay file names in alphabetical order and separates the overlay files names into three lists corresponding to the training set, the validation set, and the test set.
    
    parameters:
    ----------
    overlayPath: The folder where to find all overlay files.
    kFoldIndicesPath: Path to a file that stores an array of values with one value for each overlay file in the alphabetical order. The value associated to a file allows to assign the file to either the training set, the validation set, or the test set.
    
    returns:
    --------
    trainSetNames: the list of overlay files assigned to the training set.
    validationSetNames: the list of overlay files assigned to the validation set.
    testSetNames: The list of overlay files assigned to the test set.
    """
    
    
    overlayFiles = get_overlay_files_list(overlayPath)
    
    trainSetNames, validationSetNames, testSetNames = separate_overlay_files_training_validation_test_set(overlayFiles, kFoldIndicesPath)
    
    return trainSetNames, validationSetNames, testSetNames



def get_overlay_files_list(overlayPath):
    """
    This function returns the list of overlay file names in alphabetical order.
    
    parameter:
    ----------
    overlayPath: The folder where to find all overlay files.
    
    returns:
    --------
    The list of overlay file names.
    """
    
    
    overlayFiles = []
    for root, directoryList, fileList in os.walk(overlayPath):
        for file in fileList:
            if ".png" in file:
                overlayFiles.append(file)
    
    
    overlayFiles.sort()
    
    return overlayFiles



def separate_overlay_files_training_validation_test_set(overlayFiles, kFoldIndicesPath):
    """
    This function separates the overlay files names into three lists corresponding to the training set, the validation set, and the test set.
    
    parameters:
    -----------
    overlayFiles: The list of overlay file names.
    kFoldIndicesPath: Path to a file that stores an array of values with one value for each overlay file in the alphabetical order. The value associated to a file allows to assign the file to either the training set, the validation set, or the test set.
    
    returns:
    --------
    trainSetNames: the list of overlay files assigned to the training set.
    validationSetNames: the list of overlay files assigned to the validation set.
    testSetNames: The list of overlay files assigned to the test set.
    """
    
    
    kFoldIndicesStructure = scipy.io.loadmat(kFoldIndicesPath)
    kFoldIndices = kFoldIndicesStructure["kfoldIndices"]
    
    trainSetNames = []
    validationSetNames = []
    testSetNames = []
    
    i = 0
    while i < kFoldIndices.shape[0]:
        if kFoldIndices[i, 0] == 1:
            testSetNames.append(overlayFiles[i])
        
        elif kFoldIndices[i, 0] == 4:
            validationSetNames.append(overlayFiles[i])
        
        else:
            trainSetNames.append(overlayFiles[i])
        
        i += 1
    
    
    return trainSetNames, validationSetNames, testSetNames



def create_dataset(overlayPath, dataCubePath, setNames, colours):
    """
    This function creates a dataset of spectra with their respective classes.
    
    parameters:
    -----------
    overlayPath: The folder where to find all overlay files.
    dataCubePath: The folder where to find all datacube files, i.e. files where spectra for each pixels are stored.
    setNames: The list of overlay file names.
    colours: The numpy array with 3 columns and a number of rows equal to the number of classes. Each colour represents a class and each row represents a colour. Each row contains an array of 3 values representing the RGB intensities of the colour. The row position of a colour determines the value assigned to the corresponding class.
    
    returns:
    --------
    spectraSet: A list of lists where each inner list corresponds to a data cube file and the indexes of the inner list correspond to the histological classes. At each index of the inner list is stored an numpy array storing the spectra of the corresponding histological class in the corresponding files.
    waveNumbers: The list of wave numbers where the absorbance has been measured.
    setInput: The spectra of all annotated pixels. Each row corresponds to one spectra.
    setOutput: The histological classes associated with the spectra whose row index in setInput is equal to the index in setOutput.
    """
    
    
    annotationsPositionsSet = get_annotations_positions(overlayPath, setNames, colours)
    
    dataCubeFiles = get_cube_name(setNames)
    
    print("loading data cube and concatenating each class")
    
    spectraSet, waveNumbers, setInput, setOutput = load_each_data_cube_and_catenate_each_class(dataCubeFiles, dataCubePath, annotationsPositionsSet)
    
    return spectraSet, waveNumbers, setInput, setOutput



def get_annotations_positions(overlayPath, setNames, colours):
    """
    This function computes the positions where annotations are made on images and returns a list of list where each inner list corresponds to a file and the inner list contains the positions for each histological class, each histological class corresponding to an index in the inner list. At each index of the inner list is stored an numpy array storing the positions of the corresponding histological class in the corresponding files. The numpy array is a 2D matrix with two columns corresponding to the rows and columns of the pixels labeled with an histological class. The number of rows is dependent upon the number of pixels having the given histological class in the corresponding file.
    
    parameters:
    -----------
    overlayPath: The folder where to find all overlay files.
    setNames: The list of overlay files names whose annotations must be extracted.
    colours: The numpy array with 3 columns and a number of rows equal to the number of classes. Each colour represents a class and each row represents a colour. Each row contains an array of 3 values representing the RGB intensities of the colour. The row position of a colour determines the value assigned to the corresponding class.
    
    returns:
    --------
    The annotations positions
    """
    
    
    annotationsPositions = []
    
    for name in setNames:
        mask = skimage.io.imread(os.path.join(overlayPath, name))
        redValues, greenValues, blueValues = convert_mask_to_RGB(mask)
        indicesClasses = find_colour_positions(redValues, greenValues, blueValues, colours)
        
        annotationsPositions.append(indicesClasses)
    
    
    return annotationsPositions



def convert_mask_to_RGB(mask):
    """
    This function returns the 3 RGB channels from an image.
    
    parameters:
    -----------
    mask: The image pixel values
    
    returns:
    --------
    The red, green, and blue channel respectively.
    """
    
    
    return mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]



def find_colour_positions(redValues, greenValues, blueValues, colours):
    """
    This function returns the pixels where the colour corresponds to an annotation for an histology class.
    
    parameters:
    -----------
    redValues: The red intensities of the pixels of an image.
    greenValues: The green intensities of the pixels of an image.
    blueValues: The blue intensities of the pixels of an image.
    colours: The numpy array with 3 columns and a number of rows equal to the number of classes. Each colour represents a class and each row represents a colour. Each row contains an array of 3 values representing the RGB intensities of the colour. The row position of a colour determines the value assigned to the corresponding class.
    
    returns:
    --------
    A list where each index stores an numpy array storing the positions of the corresponding histological class. The order of the histological classes is given by the colours array order. The numpy array is a 2D matrix with two columns corresponding to the rows and columns of the pixels labeled with an histological class. The number of rows is dependent upon the number of pixels having the given histological class.
    """
    
    
    numberColours = len(colours)
    indicesClasses = []
    
    i = 0
    while i < numberColours:
        indicesSelected = numpy.argwhere((redValues == colours[i, 0]) & (greenValues == colours[i, 1]) & (blueValues == colours[i, 2]))
        indicesClasses.append(indicesSelected)
        
        i += 1
    
    return indicesClasses



def get_cube_name(setNames):
    """
    This function returns a list of the data cube file names corresponding to the given overlay file names.
    
    parameter:
    ----------
    setNames: The list of overlay file names.
    
    returns:
    --------
    The list of data cube file names.
    """
    
    
    return [string.replace(".png", ".mat") for string in setNames]



def load_each_data_cube_and_catenate_each_class(dataCubeFiles, dataCubePath, annotationsPositionsSet):
    """
    This function creates the dataset used either for training or for testing.
    
    parameters:
    -----------
    dataCubeFiles: The list of data cube file names
    dataCubePath: The folder where to find all datacube files, i.e. files where spectra for each pixels are stored.
    annotationsPositionsSet: A list of lists where each inner list corresponds to a data cube file and the indexes of the inner list correspond to the histological classes. At each index of the inner list is stored an numpy array storing the positions of the corresponding histological class in the corresponding files. The numpy array is a 2D matrix with two columns corresponding to the rows and columns of the pixels labeled with an histological class. The number of rows is dependent upon the number of pixels having the given histological class in the corresponding file.
    
    returns:
    --------
    spectraTrainingSet: A list of lists where each inner list corresponds to a data cube file and the indexes of the inner list correspond to the histological classes. At each index of the inner list is stored an numpy array storing the spectra of the corresponding histological class in the corresponding files.
    waveNumbers: The list of wave numbers where the absorbance has been measured.
    spectraAllClassConcatenated: The spectra of all annotated pixels. Each row corresponds to one spectra.
    spectraCorrespondingClass: The histological classes associated with the spectra whose row index in spectraAllClassConcatenated is equal to the index in spectraCorrespondingClass.
    """
    
    
    spectraAllClass = []
    i = 0
    while i < len(annotationsPositionsSet[0]):
        spectraAllClass.append(0)
        
        i += 1
    
    
    waveNumbers = []
    spectraTrainingSet = []
    i = 0
    while i < len(annotationsPositionsSet):
        fileName = dataCubeFiles[i]
        dataCubeStruct = scipy.io.loadmat(os.path.join(dataCubePath, fileName))
        dataCubeImage = dataCubeStruct["image"]
        
        if i == 0:
            waveNumbers = dataCubeStruct["wn"]
        
        
        spectraEachColor = []
        j = 0
        while j < len(annotationsPositionsSet[0]):
            rowIndexes = [element[0] for element in annotationsPositionsSet[i][j]]
            columnIndexes = [element[1] for element in annotationsPositionsSet[i][j]]
            spectraSelected = dataCubeImage[rowIndexes, columnIndexes, :]
            
            spectraEachColor.append(spectraSelected)
            
            if not isinstance(spectraAllClass[j], numpy.ndarray):
                spectraAllClass[j] = spectraSelected
            
            else:
                spectraAllClass[j] = numpy.vstack((spectraAllClass[j], spectraSelected))
            
            j += 1
        
        
        spectraTrainingSet.append(spectraEachColor)
        
        i += 1
    
    
    spectraAllClassConcatenated = spectraAllClass[0]
    spectraCorrespondingClass = numpy.full((spectraAllClass[0].shape[0],), 0)
    i = 1
    while i < len(annotationsPositionsSet[0]):
        spectraAllClassConcatenated = numpy.vstack((spectraAllClassConcatenated, spectraAllClass[i]))
        
        spectraCorrespondingClass = numpy.concatenate((spectraCorrespondingClass, numpy.full((spectraAllClass[i].shape[0],), i)))
        
        i += 1
    
    
    return spectraTrainingSet, waveNumbers, spectraAllClassConcatenated, spectraCorrespondingClass



def make_preprocessing(setInput, setOutput, waveNumbers, noiseReductionFirst, methodNoiseReduction, numberPrincipalComponents, inverseTransform, additionalParametersNoiseReduction, intervalsToSelect, normalise, methodSavitzkyGolayFilter, smoothingPoints, orderPolynomialFit, additionalParametersSavitzkyGolayFilter):
    """
    This function makes the preprocessing operations on the data before feeding them to the learning algorithm.
    
    parameters:
    -----------
    setInput: The spectra of all annotated pixels. Each row corresponds to one spectra.
    setOutput: The histological classes associated with the spectra whose row index in setInput is equal to the index in setOutput.
    waveNumbers: The list of wave numbers where the absorbance has been measured.
    peakWaveQualityTest: This corresponds to the peak to detect when using quality test to remove pixels that do not corresponds to tissues. If set to None, the quality test is not done.
    troughWaveQualityTest: This corresponds to the trough to detect when using quality test to remove pixels that do not corresponds to tissues. If set to None, the quality test is not done.
    minimumQualityTest: The minimum absorbance to obtain to pass the quality test. If set to None, the quality test is not done.
    maximumQualityTest: The maximum absorbance to obtain to pass the quality test. If set to None, the quality test is not done.
    noiseReductionFirst: Boolean determining the order in which the noise reduction and the range selection operations must be done. True means noise reduction is done first. False means range selection is done first.
    methodNoiseReduction: A string indicating the algorithm to use for noise reduction. Possible values are: sklearn_pca, sklearn_kernel_pca, sklearn_sparse_pca, sklearn_truncated_svd, sklearn_incremental_pca, sklearn_mini_batch_sparse_pca, statsmodels_pca, bassan_nipals. None value is also possible and means no noise reduction.
    numberPrincipalComponents: Number of principal components to retain when performing principal components based noise reduction. If set to None, the noise reduction is not done.
    inverseTransform: boolean telling if the noise reduction must get back the original data denoised. True means to come back to the original features, False means to keep the features obtained by principal component analysis. If set to False, range selection and savitzky golay filter cannot be done, i.e. intervalsToSelect must be None and smoothingPoints or orderPolynomialFit or both must be None.
    additionalParametersNoiseReduction: a dictionary providing additional parameters necessary for the noise reduction method selected.
    intervalsToSelect: The intervals to keep, the rest is removed. If set to None, the range selection is not done.
    normalise: Boolean telling whether to normalise the spectra or not.
    methodSavitzkyGolayFilter: A string indicating the algorithm to use for Savitzky-Golay filter. Possible values are: savitzky_golay_filter, first_derivative. None value is also possible and means no Savitzky-Golay filter.
    smoothingPoints: The number of smoothing points to consider when computing the first derivative by performing Savitzky-Golay smoothing. If set to None, the first derivative is not done.
    orderPolynomialFit: The order of the polynomial used to fit when computing the first derivative by performing Savitzky-Golay smoothing. If set to None, the first derivative is not done.
    additionalParametersSavitzkyGolayFilter: a dictionary providing additional parameters necessary for the Savitzky-Golay filter method selected.
    
    returns:
    --------
    setInput: The preprocessed spectra of all annotated pixels. Each row corresponds to one spectra.
    setOutput: The preprocessed histological classes associated with the spectra whose row index in setInput is equal to the index in setOutput.
    waveNumbers: The preprocessed list of wave numbers where the absorbance has been measured.
    """
    
    
    """
    print("quality test in progress")
    
    setInput, setOutput = quality_test(setInput, setOutput, waveNumbers, peakWaveQualityTest, troughWaveQualityTest, minimumQualityTest, maximumQualityTest)
    """
    
    
    if noiseReductionFirst:
        print("noise reduction in progress")
        
        setInput, pcaClass = noise_reduction(setInput, methodNoiseReduction, numberPrincipalComponents, inverseTransform, additionalParametersNoiseReduction)
        
        
        print("selecting range")
        
        setInput, waveNumbers = select_range(setInput, intervalsToSelect, waveNumbers)
    
    
    else:
        print("selecting range")
        
        setInput, waveNumbers = select_range(setInput, intervalsToSelect, waveNumbers)
        
        
        print("noise reduction in progress")
        
        setInput, pcaClass = noise_reduction(setInput, methodNoiseReduction, numberPrincipalComponents, inverseTransform, additionalParametersNoiseReduction)
    
    
    
    print("normalising vector")
    
    setInput = vector_normalisation(setInput, normalise)
    
    
    print("Savitzky Golay filter in progress")
    
    setInput = savitzky_golay_filter(setInput, methodSavitzkyGolayFilter, waveNumbers, smoothingPoints, orderPolynomialFit, additionalParametersSavitzkyGolayFilter)
    
    
    return setInput, setOutput, waveNumbers, pcaClass



def quality_test(setInput, setOutput, waveNumbers, peakWave, troughWave, minimum, maximum):
    """
    This function tests the spectra in order to keep spectra that correspond to a specific property, e.g. to keep spectra that correspond to tissue. In order to do that, the closest wave numbers are found according to the ones given and the difference between the two values at the specified wave numbers must be in a specified interval.
    
    parameters:
    -----------
    setInput: The matrix where each row contains the spectrum of a pixel.
    setOutput: An array whose values correspond to the classes of the spectra in setInput at the rows corresponding to the indexes in setOutput.
    waveNumbers: An array storing the wavenumbers of all spectra in setInput.
    peakWave: The peak to test.
    troughWave: The trough to test.
    minimum: The lower bound of the interval.
    maximum: The upper bound of the interval.
    
    returns:
    --------
    setInput: The matrix where each spectrum has passed the test.
    setOutput: The corresponding classes.
    """
    
    
    if (peakWave is None) and (troughWave is None) and (minimum is None) and (maximum is None):
        return copy.deepcopy(setInput), copy.deepcopy(setOutput)
    
    elif (peakWave is None) or (troughWave is None) or (minimum is None) or (maximum is None):
        raise ValueError("Parameters for quality_test function are not valid.")
    
    
    indexPeak = numpy.argmin(numpy.abs(waveNumbers - peakWave))
    indexTrough = numpy.argmin(numpy.abs(waveNumbers - troughWave))
    
    keptValues = numpy.argwhere(((setInput[:, indexPeak] - setInput[:, indexTrough]) >= minimum) & ((setInput[:, indexPeak] - setInput[:, indexTrough]) <= maximum))
    keptValues = keptValues.flatten()
    
    setInput = setInput[keptValues, :]
    setOutput = setOutput[keptValues]
    
    return setInput, setOutput



def noise_reduction(setToDenoise, method, numberComponentsToKeep, inverseTransform, kwargs):
    """
    This function uses principal component-based noise reduction in order to improve signal-to-noise.
    
    parameters:
    -----------
    setToReduce: The matrix of spectra to process. Each row contains the spectrum of a pixel.
    method: A string indicating the algorithm to use for noise reduction. Possible values are: sklearn_pca, sklearn_kernel_pca, sklearn_sparse_pca, sklearn_truncated_svd, sklearn_incremental_pca, sklearn_mini_batch_sparse_pca, statsmodels_pca, bassan_nipals. None value is also possible and means no noise reduction if numberComponentsToKeep is also None. Otherwise, an exception is raised.
    numberComponentsToKeep: The number of principal components to keep.
    inverseTransform: boolean telling if the noise reduction must get back the original data denoised. True means to come back to the original features, False means to keep the features obtained by principal component analysis. If set to False, range selection and savitzky golay filter cannot be done.
    kwargs: Additional parameters necessary for the noise reduction method selected. 
    
    returns:
    --------
    The processed matrix of spectra.
    """
    
    
    if (method is None):
        setDenoised = copy.deepcopy(setToDenoise)
        pca = None
    
    
    elif method == "reuse":
        pca = kwargs["pca"]
        
        if (pca is None):
            setDenoised = copy.deepcopy(setToDenoise)
            pca = None
        
        else:
            setDimensionalityReduced = pca.transform(setToDenoise)
            
            if inverseTransform == True:
                setDenoised = pca.inverse_transform(setDimensionalityReduced)
            
            else:
                setDenoised = setDimensionalityReduced
        
        
        return setDenoised, pca
    
    
    
    elif method == "bassan_nipals":
        setDenoised = noise_reduction_bassan_nipals(setToDenoise, numberComponentsToKeep, inverseTransform)
        pca = None
    
    
    elif method == "statsmodels_pca":
        pca = statsmodels.multivariate.pca.PCA(setToDenoise, numberComponentsToKeep, **kwargs)
        
        if inverseTransform == True:
            setDenoised = pca.scores @ pca.loadings.T
        
        else:
            setDenoised = pca.scores
    
    
    else:
        if method == "sklearn_pca":
            pca = sklearn.decomposition.PCA(numberComponentsToKeep, **kwargs)
        
        elif method == "sklearn_kernel_pca":
            pca = sklearn.decomposition.KernelPCA(numberComponentsToKeep, **kwargs)
        
        elif method == "sklearn_sparse_pca":
            pca = sklearn.decomposition.SparsePCA(numberComponentsToKeep, **kwargs)
        
        elif method == "sklearn_truncated_svd":
            pca = sklearn.decomposition.TruncatedSVD(numberComponentsToKeep, **kwargs)
        
        elif method == "sklearn_incremental_pca":
            pca = sklearn.decomposition.IncrementalPCA(numberComponentsToKeep, **kwargs)
        
        elif method == "sklearn_mini_batch_sparse_pca":
            pca = sklearn.decomposition.MiniBatchSparsePCA(numberComponentsToKeep, **kwargs)
        
        elif callable(method):
            pca = method(numberComponentsToKeep, **kwargs)
        
        else:
            raise ValueError("Method parameter for noise_reduction function is not valid.")
        
        pca.fit(setToDenoise)
        setDimensionalityReduced = pca.transform(setToDenoise)
        
        if inverseTransform == True:
            setDenoised = pca.inverse_transform(setDimensionalityReduced)
        
        else:
            setDenoised = setDimensionalityReduced
    
    
    
    return setDenoised, pca



def noise_reduction_bassan_nipals(setToReduce, numberComponentsToKeep, inverseTransform):
    """
    This function uses principal component-based noise reduction in order to improve signal-to-noise with the Bassan nipals method.
    
    parameters:
    -----------
    setToReduce: The matrix of spectra to process. Each row contains the spectrum of a pixel.
    numberComponentsToKeep: The number of principal components to keep.
    inverseTransform: boolean telling if the noise reduction must get back the original data denoised. True means to come back to the original features, False means to keep the features obtained by principal component analysis. If set to False, range selection and savitzky golay filter cannot be done.
    
    returns:
    --------
    The processed matrix of spectra.
    """
    
    
    if setToReduce.shape[0] < 4097:
        tol = 1e-2
    
    else:
        tol = 1e-1
    
    Tprinc = numpy.ones((setToReduce.shape[0], numberComponentsToKeep))
    Pprinc = numpy.ones((setToReduce.shape[1], numberComponentsToKeep))
    
    j = 0
    while j < numberComponentsToKeep:
        T = Tprinc[:,j].reshape((-1, 1))
        d = 1.0
        
        while d > tol:
            P = setToReduce.T @ T
            P = P / numpy.linalg.norm(P)
            Told = T
            denominator = (P.T @ P)[0, 0]
            T = (setToReduce @ P) / denominator
            d = numpy.linalg.norm(Told - T)
        
        
        setToReduce = setToReduce - (T @ P.T)
        Tprinc[:,j] = T.reshape((-1,))
        Pprinc[:,j] = P.reshape((-1,))
        
        j += 1
    
    if inverseTransform == True:
        result = Tprinc[:, 0:numberComponentsToKeep] @ Pprinc[:, 0:numberComponentsToKeep].T
        
        if not isinstance(result, numpy.ndarray):
            result = numpy.array([[result]])
        
        
        return result
    
    
    return Tprinc[:, 0:numberComponentsToKeep]



def select_range(setInput, intervals, waveNumbers):
    """
    This function allows to select some ranges to extract from all spectra given. The spectra ranges that are outside the extracted ranges are removed. For each range, the lower and upper bounds of the intervals are adapted to choose the closest wave number in the waveNumbers array.
    
    parameters:
    -----------
    setInput: The matrix where each row contains the spectrum of a pixel.
    intervals: A numpy array with 2 columns and a number of rows equal to the number of intervals to extract. Each row corresponds to an interval to select. If intervals value is None, then no range selections are done.
    waveNumbers: An array storing the wavenumbers of all spectra in setInput.
    
    returns:
    --------
    setInputIntermediary: The matrix setInput with spectra values only in the range given in intervals.
    waveNumbersIntermediary: The wave numbers that are in the range extracted.
    """
    
    
    if intervals is None:
        return copy.deepcopy(setInput), copy.deepcopy(waveNumbers)
    
    
    setInputIntermediary = None
    waveNumbersIntermediary = None
    
    i = 0
    while i < intervals.shape[0]:
        indexStart = numpy.argmin(numpy.abs(waveNumbers - intervals[i, 0]))
        indexEnd = numpy.argmin(numpy.abs(waveNumbers - intervals[i, 1]))
        
        block = setInput[:, indexStart:(indexEnd + 1)]
        waveNumbersBlock = waveNumbers[indexStart:(indexEnd + 1), :]
        
        if i == 0:
            setInputIntermediary = block
            waveNumbersIntermediary = waveNumbersBlock
        
        else:
            setInputIntermediary = numpy.hstack((setInputIntermediary, block))
            waveNumbersIntermediary = numpy.vstack((waveNumbersIntermediary, waveNumbersBlock))
        
        
        i += 1
    
    
    return setInputIntermediary, waveNumbersIntermediary



def vector_normalisation(inputSet, normalise):
    """
    This function normalises each spectra in inputSet.
    
    parameter:
    ----------
    inputSet: The matrix where each row contains the spectrum of a pixel.
    normalise: Boolean telling wheter to normalize or not. True means normalisation must been done.
    
    returns:
    --------
    The matrix of spectra normalised.
    """
    
    
    if not normalise:
        return copy.deepcopy(inputSet)
    
    
    squares = inputSet ** 2
    sumOfSquares = numpy.sum(squares, axis = 1)
    divisor = sumOfSquares ** (1/2)
    divisor[divisor == 0.0] = 1.0
    
    multiplier = 1.0 / divisor
    multiplier = multiplier.reshape((-1, 1))
    inputSet = inputSet * multiplier
    
    return inputSet



def savitzky_golay_filter(intensity, method, waveNumbers, smoothingPoints, orderPolynomialFit, kwargs):
    """
    This function applies a Savitzky-Golay filter.
    
    parameters:
    -----------
    intensity: The matrix where each row contains the spectrum of a pixel.
    method: A string indicating the algorithm to use for Savitzky-Golay filter. Possible values are: savitzky_golay_filter, first_derivative. None value is also possible and means no Savitzky-Golay filter if smoothingPoints and orderPolynomialFit are also None. Otherwise, an exception is raised.
    waveNumbers: An array storing the wavenumbers of all spectra in intensity.
    smoothingPoints: The window size when performing Savitzky-Golay smoothing.
    orderPolynomialFit: The degree of the polynomial fit.
    kwargs: Additional parameters necessary for the Savitzky-Golay filter method selected.
    
    returns:
    --------
    The matrix of spectra with Savitzky-Golay filter applied.
    """
    
    
    if (method is None):
        return copy.deepcopy(intensity)
    
    
    elif method == "first_derivative":
        result = first_derivative(waveNumbers, intensity, smoothingPoints, orderPolynomialFit)
    
    elif method == "savitzky_golay_filter":
        dx = (waveNumbers[-1, 0] - waveNumbers[0, 0]) / waveNumbers.shape[0]
    
        result = scipy.signal.savgol_filter(intensity, smoothingPoints, orderPolynomialFit, delta = dx, **kwargs)
    
    elif callable(method):
        result = method(intensity, smoothingPoints, orderPolynomialFit, **kwargs)
    
    else:
        raise ValueError("Method parameter for savitzky_golay_filter function is not valid.")
    
    
    return result



def first_derivative(waveNumbers, intensity, smoothingPoints, orderPolynomialFit):
    """
    This function computes the first derivative of the spectra by performing Savitzky-Golay smoothing.
    
    parameters:
    -----------
    waveNumbers: An array storing the wavenumbers of all spectra in intensity.
    intensity: The matrix where each row contains the spectrum of a pixel.
    smoothingPoints: The window size when performing Savitzky-Golay smoothing.
    orderPolynomialFit: The degree of the polynomial fit.
    
    returns:
    --------
    The first derivative of the spectra by performing Savitzky-Golay smoothing.
    """
    
    
    g = numpy.array([scipy.signal.savgol_coeffs(smoothingPoints, orderPolynomialFit, deriv = d, use = "dot") for d in range(orderPolynomialFit + 1)]).T / scipy.special.factorial(numpy.arange(orderPolynomialFit + 1))
    
    dx = (waveNumbers[-1, 0] - waveNumbers[0, 0]) / waveNumbers.shape[0]
    halfWin = ((smoothingPoints + 1) // 2) - 1
    SG1 = numpy.zeros(intensity.shape)
    
    r = 0
    while r < intensity.shape[0]:
        
        n = ((smoothingPoints + 1) // 2) - 1
        while n < (intensity.shape[1] - ((smoothingPoints + 1) // 2)):
            SG1[r, n] = g[:, 1] @ intensity[r, (n - halfWin):(n + halfWin + 1)]
            
            n += 1
        
        
        r += 1
    
    
    SG1 = SG1 / dx
    
    return SG1



def merge_classes(setClasses, mappingDictionary):
    """
    This function allows to merge some classes into another class.
    
    parameters:
    -----------
    setClasses: The array storing the class values.
    mappingDictionary: The set where the keys are tuples representing the classes that must be mapped to the class given by their respective values.
    
    returns:
    --------
    The array storing the class values where some classes are merged into another class.
    """
    
    
    newSetClasses = copy.deepcopy(setClasses)
    
    for initialClasses, finalClass in mappingDictionary.items():
        for individualClass in initialClasses:
            newSetClasses[setClasses == individualClass] = finalClass
    
    
    return newSetClasses



def classes_equalization(setInput, setOutput, method, kwargs):
    """
    This function modifies the dataset such that each class in the dataset has a same number of samples.
    
    parameters:
    -----------
    setInput: The matrix where each row contains the spectrum of a pixel.
    setOutput: An array whose values correspond to the classes of the spectra in setInput at the rows corresponding to the indexes in setOutput.
    methodClassesEqualization: A string indicating the algorithm to use for classes equalization. Possible values are: random_under_sampler, random_over_sampler, tomek_links, cluster_centroids, smote, smote_tomek. None value is also possible and means no classes equalization.
    kwargs: Additional parameters necessary for the classes equalization method selected.
    
    returns:
    --------
    setInput: The equalized matrix where each row contains the spectrum of a pixel.
    setOutput: The corresponding array whose values correspond to the classes of the spectra in setInput at the rows corresponding to the indexes in setOutput.
    """
    
    
    if method == "random_under_sampler":
        sampler = imblearn.under_sampling.RandomUnderSampler(**kwargs)
        setInput, setOutput = sampler.fit_resample(setInput, setOutput)
    
    elif method == "random_over_sampler":
        sampler = imblearn.over_sampling.RandomOverSampler(**kwargs)
        setInput, setOutput = sampler.fit_resample(setInput, setOutput)
    
    elif method == "tomek_links":
        sampler = imblearn.under_sampling.TomekLinks(**kwargs)
        setInput, setOutput = sampler.fit_resample(setInput, setOutput)
    
    elif method == "cluster_centroids":
        sampler = imblearn.under_sampling.ClusterCentroids(**kwargs)
        setInput, setOutput = sampler.fit_resample(setInput, setOutput)
    
    elif method == "smote":
        sampler = imblearn.over_sampling.SMOTE(**kwargs)
        setInput, setOutput = sampler.fit_resample(setInput, setOutput)
    
    elif method == "smote_tomek":
        sampler = imblearn.combine.SMOTETomek(**kwargs)
        setInput, setOutput = sampler.fit_resample(setInput, setOutput)
    
    elif callable(method):
        sampler = method(**kwargs)
        setInput, setOutput = sampler.fit_resample(setInput, setOutput)
    
    elif not (method is None):
        raise ValueError("Method parameter for classes_equalization function is not valid.")
    
    
    return setInput, setOutput



def feature_selection(setInput, setOutput, methodFeatureSelection, additionalParametersFeatureSelection):
    if methodFeatureSelection is None:
        return setInput, None
    
    
    featureSelector = methodFeatureSelection(**additionalParametersFeatureSelection)
    featureSelector.fit(setInput, setOutput)
    
    
    return setInput, featureSelector
    



def predict_with_acceptance_threshold(model, testSetInput, testSetOutput, acceptanceThreshold = None, methodPrediction = "voting_hard"):
    """
    This function makes prediction of spectra classes based on a trained model. Only the prediction where a minimum proportion of estimators agrees on the class are kept, if the acceptance threshold is provided.
    
    parameters:
    -----------
    model: The trained model.
    testSetInput: The matrix where each row contains the spectrum of a pixel.
    testSetOutput: An array whose values correspond to the classes of the spectra in testSetInput at the rows corresponding to the indexes in testSetOutput.
    acceptanceThreshold: The minimum proportion of estimators that must agree for a class prediciton of a spectrum to be kept. If None, then the whole testSetInput is classified and indexesPredictedAllEstimators output is None.
    
    returns:
    --------
    testSetInputClassified: The matrix where each row contains the spectrum of a pixel and the proportion of estimators agreeing on the class of these spectra is above the acceptanceThreshold.
    testSetOutputClassified: An array whose values correspond to the real classes of the spectra in testSetInputClassified at the rows corresponding to the indexes in testSetOutputClassified.
    indexesPredicted: An array whose values correspond to the predicted classes of the spectra in testSetInputClassified at the rows corresponding to the indexes in testSetOutputClassified.
    testSetInputRejected: The matrix where each row contains the spectrum of a pixel and the proportion of estimators agreeing on the class of these spectra is strictly below the acceptanceThreshold.
    testSetOutputRejected: An array whose values correspond to the real classes of the spectra in testSetInputRejected at the rows corresponding to the indexes in testSetOutputRejected.
    indexesPredictedAllEstimators: A matrix where the number of rows is equal to the total number of spectra and the number of columns is equal to the number of estimators in the trained model. The values in the matrix give the predicted classes for the corresponding spectra and estimators.
    """
    
    
    if acceptanceThreshold is None:
        testSetInputClassified = numpy.array(copy.deepcopy(testSetInput))
        testSetOutputClassified = numpy.array(copy.deepcopy(testSetOutput))
        indexesPredicted = model.predict(testSetInputClassified)
        testSetInputRejected = numpy.array([])
        testSetOutputRejected = numpy.array([])
        indexesPredictedAllEstimators = None
        
        return testSetInputClassified, testSetOutputClassified, indexesPredicted, testSetInputRejected, testSetOutputRejected, indexesPredictedAllEstimators
    
    
    elif methodPrediction == "predict_proba":
        classesPredicted = model.predict(testSetInput)
        classesProbabilities = model.predict_proba(testSetInput)
        
        classesMaxProbability = numpy.amax(classesProbabilities, axis = 1)
        
        testSetInputClassified = testSetInput[classesMaxProbability >= acceptanceThreshold]
        testSetOutputClassified = testSetOutput[classesMaxProbability >= acceptanceThreshold]
        indexesPredicted = classesPredicted[classesMaxProbability >= acceptanceThreshold]
        testSetInputRejected = testSetInput[classesMaxProbability < acceptanceThreshold]
        testSetOutputRejected = testSetOutput[classesMaxProbability < acceptanceThreshold]
        indexesPredictedAllEstimators = None
        
        return testSetInputClassified, testSetOutputClassified, indexesPredicted, testSetInputRejected, testSetOutputRejected, indexesPredictedAllEstimators
    
    
    elif (not hasattr(model, "estimators_")):
        raise TypeError("The model given is not an ensemble method and the acceptance threshold is not None.")
    
    
    elif methodPrediction != "voting_hard":
        raise ValueError("The value of the variable methodPrediction is incorrect.")
    
    
    
    testSetInputClassified = []
    testSetInputRejected = []
    testSetOutputClassified = []
    testSetOutputRejected = []
    indexesPredicted = []
    
    
    indexesPredictedAllEstimators = [estimator.predict(testSetInput) for estimator in model.estimators_]
    indexesPredictedAllEstimators = numpy.array(indexesPredictedAllEstimators)
    indexesPredictedAllEstimators = indexesPredictedAllEstimators.T
    
    
    i = 0
    while i < indexesPredictedAllEstimators.shape[0]:
        
        uniqueValues, uniqueValuesCounts = numpy.unique(indexesPredictedAllEstimators[i, :], return_counts = True)
        
        uniqueValuesCounts = uniqueValuesCounts / indexesPredictedAllEstimators.shape[1]
        
        indexMax = uniqueValuesCounts.argmax()
        
        if uniqueValuesCounts[indexMax] < acceptanceThreshold:
            testSetInputRejected.append(testSetInput[i, :])
            testSetOutputRejected.append(testSetOutput[i])
        
        else:
            testSetInputClassified.append(testSetInput[i, :])
            testSetOutputClassified.append(testSetOutput[i])
            
            indexesPredicted.append(uniqueValues[indexMax])
        
        
        i += 1
    
    
    if testSetInputClassified == []:
        testSetInputClassified = numpy.array([])
    
    else:
        testSetInputClassified = numpy.vstack(testSetInputClassified)
    
    
    if testSetInputRejected == []:
        testSetInputRejected = numpy.array([])
    
    else:
        testSetInputRejected = numpy.vstack(testSetInputRejected)
    
    
    testSetOutputClassified = numpy.array(testSetOutputClassified)
    testSetOutputRejected = numpy.array(testSetOutputRejected)
    indexesPredicted = numpy.array(indexesPredicted)
    
    return testSetInputClassified, testSetOutputClassified, indexesPredicted, testSetInputRejected, testSetOutputRejected, indexesPredictedAllEstimators



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
    
    
    def __init__(self, overlayPath, dataCubePath, kFoldIndicesPath, colours, peakWaveQualityTest, troughWaveQualityTest, minimumQualityTest, maximumQualityTest, noiseReductionFirst, methodNoiseReduction, numberPrincipalComponents, inverseTransform, additionalParametersNoiseReduction, pcaClass, intervalsToSelect, normalise, methodSavitzkyGolayFilter, smoothingPoints, orderPolynomialFit, additionalParametersSavitzkyGolayFilter, dictionaryClassesMapping, methodClassesEqualization, additionalParametersClassesEqualization, methodFeatureSelection, additionalParametersFeatureSelection, model, modelAlgorithm, seed, waveNumbers):
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
    """
    OVERLAYPATH: The folder where to find all overlay files.
    DATACUBEPATH: The folder where to find all datacube files, i.e. files where spectra for each pixels are stored.
    KFOLDINDICESPATH: Path to a file that stores an array of values with one value for each overlay file in the alphabetical order. The value associated to a file allows to assign the file to either the training set or the test set. If the value in the array is 1, the corresponding file is in the test set. Otherwise, it is in the training set.
    COLOURS: The numpy array with 3 columns and a number of rows equal to the number of classes. Each colour represents a class and each row represents a colour. Each row contains an array of 3 values representing the RGB intensities of the colour. The row position of a colour determines the value assigned to the corresponding class.
    PEAKWAVEQUALITYTEST: This corresponds to the peak to detect when using quality test to remove pixels that do not corresponds to tissues. If set to None, the quality test is not done.
    TROUGHWAVEQUALITYTEST: This corresponds to the trough to detect when using quality test to remove pixels that do not corresponds to tissues. If set to None, the quality test is not done.
    MINIMUMQUALITYTEST: The minimum absorbance to obtain to pass the quality test. If set to None, the quality test is not done.
    MAXIMUMQUALITYTEST: The maximum absorbance to obtain to pass the quality test. If set to None, the quality test is not done.
    NOISEREDUCTIONFIRST: Boolean determining the order in which the noise reduction and the range selection operations must be done. True means noise reduction is done first. False means range selection is done first.
    METHODNOISEREDUCTION: A string indicating the algorithm to use for noise reduction. Possible values are: sklearn_pca, sklearn_kernel_pca, sklearn_sparse_pca, sklearn_truncated_svd, sklearn_incremental_pca, sklearn_mini_batch_sparse_pca, statsmodels_pca, bassan_nipals. None value is also possible and means no noise reduction if numberPrincipalComponents is also None. Otherwise, an exception is raised.
    NUMBERPRINCIPALCOMPONENTS: Number of principal components to retain when performing principal components based noise reduction. If set to None, the noise reduction is not done.
    INVERSETRANSFORM: boolean telling if the noise reduction must get back the original data denoised. True means to come back to the original features, False means to keep the features obtained by principal component analysis. If set to False, range selection and savitzky golay filter cannot be done, i.e. intervalsToSelect must be None and smoothingPoints or orderPolynomialFit or both must be None.
    ADDITIONALPARAMETERSNOISEREDUCTION: a dictionary providing additional parameters necessary for the noise reduction method selected.
    INTERVALSTOSELECT: The intervals to keep, the rest is removed. If set to None, the range selection is not done.
    NORMALISE: Boolean telling whether to normalise the spectra or not.
    METHODSAVITZKYGOLAYFILTER: A string indicating the algorithm to use for Savitzky-Golay filter. Possible values are: savitzky_golay_filter, first_derivative. None value is also possible and means no Savitzky-Golay filter if smoothingPoints and orderPolynomialFit are also None. Otherwise, an exception is raised.
    SMOOTHINGPOINTS: The number of smoothing points to consider when computing the first derivative by performing Savitzky-Golay smoothing. If set to None, the first derivative is not done.
    ORDERPOLYNOMIALFIT: The order of the polynomial used to fit when computing the first derivative by performing Savitzky-Golay smoothing. If set to None, the first derivative is not done.
    ADDITIONALPARAMETERSSAVITZKYGOLAYFILTER: a dictionary providing additional parameters necessary for the Savitzky-Golay filter method selected.
    DICTIONARYCLASSESMAPPING: The set where the keys are tuples representing the classes that must be mapped to the class given by their respective values.
    METHODCLASSESEQUALIZATION: A string indicating the algorithm to use for classes equalization. Possible values are: random_under_sampler, random_over_sampler, tomek_links, cluster_centroids, smote, smote_tomek. None value is also possible and means no classes equalization.
    ADDITIONALPARAMETERSCLASSESEQUALIZATION: a dictionary providing additional parameters necessary for the classes equalization method selected.
    NUMBERESTIMATORS: The number of estimators in the ensemble method.
    MAXIMUMFEATURES: The number of features to consider when looking for the best split in the random forest.
    MINIMUMSAMPLESPERLEAF: The minimum number of samples that must be in the left and right node when splitting a node. If it is not possible to respect this criterion, the node is not splitted.
    NUMBERJOBS: The number of jobs to use for fitting the random forest.
    ACCEPTANCETHRESHOLD: The minimum threshold that must be reached in order to accept a classification for one pixel. It means that a minimum proportions of estimators must agree on the class of a given pixel. If it is not the case, the classification for the pixel is not considered as reliable and is rejected.
    FILENAME: The files (one for the pickle file, the other for the human-readable text file) where the preprocessing parameters, the preprocessed dataset, and the trained model are stored. The .pkl extension is added for the pickle file and the .txt extension is added for the text file.
    """
    
    SEED = int(time.time()) % (2 ** 32)
    RANDOMSTATE = numpy.random.RandomState(SEED)
    
    OVERLAYPATH = "../../overlay"
    DATACUBEPATH = "../../br20832_cores"
    KFOLDINDICESPATH = "../../matlab/histology/kFoldInd3.mat"
    COLOURS = numpy.array([[103, 193, 66], [255, 0, 0], [165, 55, 156], [243, 143, 53], [194, 243, 174], [69, 120, 236]])
    PEAKWAVEQUALITYTEST = 1655
    TROUGHWAVEQUALITYTEST = 1760
    MINIMUMQUALITYTEST = 0.1
    MAXIMUMQUALITYTEST = 2
    NOISEREDUCTIONFIRSTLIST = [True]
    
    NOISEREDUCTIONLIST = [(None, True, dict())]
    
    #METHODNOISEREDUCTION = "sklearn_pca"
    
    NUMBERPRINCIPALCOMPONENTSLIST = [35]
    
    #INVERSETRANSFORM = True
    #ADDITIONALPARAMETERSNOISEREDUCTION = {"random_state" : RANDOMSTATE}
    
    INTERVALSTOSELECTLIST = [None]
    NORMALISELIST = [False]
    METHODSAVITZKYGOLAYFILTERLIST = ["savitzky_golay_filter"]
    SMOOTHINGPOINTSLIST = [13]
    ORDERPOLYNOMIALFITLIST = [4]
    ADDITIONALPARAMETERSSAVITZKYGOLAYFILTERLIST = [{"deriv" : 2, "mode" : "constant", "cval" : 0.0}]
    DICTIONARYCLASSESMAPPING = {(0, 1) : 0, (2, 3) : 1, (4,) : 2, (5,) : 3}
    
    CLASSESEQUALIZATIONLIST = [("random_under_sampler", {"random_state" : RANDOMSTATE})]
    
    #METHODCLASSESEQUALIZATION = "random_under_sampler"
    #ADDITIONALPARAMETERSCLASSESEQUALIZATION = {"random_state" : RANDOMSTATE}
    
    #METHODCLASSESEQUALIZATION = "cluster_centroids"
    #ADDITIONALPARAMETERSCLASSESEQUALIZATION = {"random_state" : RANDOMSTATE, "n_jobs" : -1}
    #METHODCLASSESEQUALIZATION = "tomek_links"
    #ADDITIONALPARAMETERSCLASSESEQUALIZATION = {"n_jobs" : -1}
    #METHODCLASSESEQUALIZATION = None
    #ADDITIONALPARAMETERSCLASSESEQUALIZATION = dict()
    #METHODFEATURESELECTION = sklearn.feature_selection.GenericUnivariateSelect
    #ADDITIONALPARAMETERSFEATURESELECTION = {"score_func" : sklearn.feature_selection.f_classif, "mode" : "k_best", "param" : 50}
    #METHODFEATURESELECTION = sklearn.feature_selection.SelectFromModel
    #ADDITIONALPARAMETERSFEATURESELECTION = {"estimator" : sklearn.ensemble.RandomForestClassifier(n_estimators = 500, max_features = 15, min_samples_leaf = 10, n_jobs = -1, random_state = RANDOMSTATE), "threshold" : -numpy.inf, "max_features" : 150}
    
    FEATURESELECTIONLIST = [(None, None)]
    
    #METHODFEATURESELECTION = None
    #ADDITIONALPARAMETERSFEATURESELECTION = None
    
    
    """
    NUMBERESTIMATORS = 500
    MAXIMUMFEATURES = 15
    MINIMUMSAMPLESPERLEAF = 10
    NUMBERJOBS = -1
    """
    
    #MODELALGORITHMLIST = [(sklearn.ensemble.AdaBoostClassifier, {"random_state" : RANDOMSTATE})]
    
    
    
    MODELALGORITHMLIST = []
    
    i = 5
    while i <= 50:
        
        j = 1
        while j <= 50:
            
            tupleToAdd = (sklearn.ensemble.ExtraTreesClassifier, {"n_estimators" : 500, "max_features" : i, "min_samples_leaf" : j, "n_jobs" : -1, "random_state" : RANDOMSTATE})
            MODELALGORITHMLIST.append(tupleToAdd)
            
            if j == 1:
                j = 5
            
            else:
                j += 5
        
        
        i += 5
    
    
    
    j = 1
    while j <= 50:
        
        tupleToAdd = (sklearn.ensemble.ExtraTreesClassifier, {"n_estimators" : 500, "max_features" : None, "min_samples_leaf" : j, "n_jobs" : -1, "random_state" : RANDOMSTATE})
        MODELALGORITHMLIST.append(tupleToAdd)
        
        if j == 1:
            j = 5
        
        else:
            j += 5
    
    
    #MODELALGORITHM = sklearn.ensemble.RandomForestClassifier
    #MODELALGORITHM = imblearn.ensemble.BalancedRandomForestClassifier
    #ADDITIONALPARAMETERSMODELALGORITHM = {"n_estimators" : NUMBERESTIMATORS, "max_features" : MAXIMUMFEATURES, "min_samples_leaf" : MINIMUMSAMPLESPERLEAF, "n_jobs" : NUMBERJOBS, "random_state" : RANDOMSTATE}
    
    TESTPARAMETERSLIST = [("voting_hard", None)]
    
    FILENAMETEXT = "histology_classification_all_results.txt"
    FOLDERDATAFRAME = "./histology_classification_data_frames"
    
    #ACCEPTANCETHRESHOLD = None
    #METHODPREDICTION = "voting_hard"
    #FILENAME = "random_forest_classifier"
    
    
    #model_for_identifying_histology(OVERLAYPATH, DATACUBEPATH, KFOLDINDICESPATH, COLOURS, PEAKWAVEQUALITYTEST, TROUGHWAVEQUALITYTEST, MINIMUMQUALITYTEST, MAXIMUMQUALITYTEST, NOISEREDUCTIONFIRST, METHODNOISEREDUCTION, NUMBERPRINCIPALCOMPONENTS, INVERSETRANSFORM, ADDITIONALPARAMETERSNOISEREDUCTION, INTERVALSTOSELECT, NORMALISE, METHODSAVITZKYGOLAYFILTER, SMOOTHINGPOINTS, ORDERPOLYNOMIALFIT, ADDITIONALPARAMETERSSAVITZKYGOLAYFILTER, DICTIONARYCLASSESMAPPING, METHODCLASSESEQUALIZATION, ADDITIONALPARAMETERSCLASSESEQUALIZATION, METHODFEATURESELECTION, ADDITIONALPARAMETERSFEATURESELECTION, MODELALGORITHM, ADDITIONALPARAMETERSMODELALGORITHM, ACCEPTANCETHRESHOLD, METHODPREDICTION, FILENAME, SEED)
    
    
    model_for_identifying_histology(OVERLAYPATH, DATACUBEPATH, KFOLDINDICESPATH, COLOURS, PEAKWAVEQUALITYTEST, TROUGHWAVEQUALITYTEST, MINIMUMQUALITYTEST, MAXIMUMQUALITYTEST, NOISEREDUCTIONFIRSTLIST, NOISEREDUCTIONLIST, NUMBERPRINCIPALCOMPONENTSLIST, INTERVALSTOSELECTLIST, NORMALISELIST, METHODSAVITZKYGOLAYFILTERLIST, SMOOTHINGPOINTSLIST, ORDERPOLYNOMIALFITLIST, ADDITIONALPARAMETERSSAVITZKYGOLAYFILTERLIST, DICTIONARYCLASSESMAPPING, CLASSESEQUALIZATIONLIST, FEATURESELECTIONLIST, MODELALGORITHMLIST, TESTPARAMETERSLIST, FILENAMETEXT, FOLDERDATAFRAME, SEED)

