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


import histology_classification_cython_code



def model_for_identifying_histology(overlayPath, dataCubePath, kFoldIndicesPath, colours, peakWaveQualityTest, troughWaveQualityTest, minimumQualityTest, maximumQualityTest, dictionaryClassesMapping, makeQualityTestTrainingSet, makeQualityTestTestSet, alphaValue):
    
    trainSetNames, validationSetNames, testSetNames = get_overlay_files_list_and_separate_overlay_files_training_validation_test_set(overlayPath, kFoldIndicesPath)
    
    trainSetNames += validationSetNames
    
    
    spectraTrainingSet, waveNumbers, trainingSetInput, trainingSetOutput, trainingSetFileNumbers, trainingSetPositions, annotationsPositionsTrainingSet, dataCubeFilesTrainingSet = create_dataset(overlayPath, dataCubePath, trainSetNames, colours)
    
    spectraTestSet, waveNumbers, testSetInput, testSetOutput, testSetFileNumbers, testSetPositions, annotationsPositionsTestSet, dataCubeFilesTestSet = create_dataset(overlayPath, dataCubePath, testSetNames, colours)
    
    waveNumbers = waveNumbers.reshape((-1,))
    
    
    if makeQualityTestTrainingSet:
        print("quality test in progress")
        
        trainingSetInput, trainingSetOutput, trainingSetFileNumbers, trainingSetPositions = quality_test(trainingSetInput, trainingSetOutput, waveNumbers, peakWaveQualityTest, troughWaveQualityTest, minimumQualityTest, maximumQualityTest, trainingSetFileNumbers, trainingSetPositions)
    
    
    if makeQualityTestTestSet:
        print("quality test in progress")
        
        testSetInput, testSetOutput, testSetFileNumbers, testSetPositions = quality_test(testSetInput, testSetOutput, waveNumbers, peakWaveQualityTest, troughWaveQualityTest, minimumQualityTest, maximumQualityTest, testSetFileNumbers, testSetPositions)
    
    
    trainingSetOutput = merge_classes(trainingSetOutput, dictionaryClassesMapping)
    testSetOutput = merge_classes(testSetOutput, dictionaryClassesMapping)
    
    
    
    
    print("plot of the mean spectrum per class for the training set")
    
    dictionaryPlot = dict()
    numberClassPair = [(0, "epithelium"), (1, "stroma"), (2, "blood"), (3, "necrosis")]
    
    for classNumber, classString in numberClassPair:
        maskFilter = (trainingSetOutput == classNumber)
        setInputFiltered = trainingSetInput[maskFilter, :]
        resultingMean = numpy.mean(setInputFiltered, axis = 0)
        
        dictionaryPlot[classString] = resultingMean
    
    
    dataFramePlot = pandas.DataFrame(dictionaryPlot, index = waveNumbers)
    dataFramePlot.plot()
    plt.xlabel("wave numbers")
    plt.ylabel("absorbance")
    plt.show()
    plt.close()
    
    
    print("plot of the mean spectrum per class for the test set")
    
    dictionaryPlot = dict()
    numberClassPair = [(0, "epithelium"), (1, "stroma"), (2, "blood"), (3, "necrosis")]
    
    for classNumber, classString in numberClassPair:
        maskFilter = (testSetOutput == classNumber)
        setInputFiltered = testSetInput[maskFilter, :]
        resultingMean = numpy.mean(setInputFiltered, axis = 0)
        
        dictionaryPlot[classString] = resultingMean
    
    
    dataFramePlot = pandas.DataFrame(dictionaryPlot, index = waveNumbers)
    dataFramePlot.plot()
    plt.xlabel("wave numbers")
    plt.ylabel("absorbance")
    plt.show()
    plt.close()
    
    
    
    
    
    print("plot of the standard deviations of spectra per class for the training set")
    
    dictionaryPlot = dict()
    numberClassPair = [(0, "epithelium"), (1, "stroma"), (2, "blood"), (3, "necrosis")]
    
    for classNumber, classString in numberClassPair:
        maskFilter = (trainingSetOutput == classNumber)
        setInputFiltered = trainingSetInput[maskFilter, :]
        resultingStd = numpy.std(setInputFiltered, axis = 0)
        
        dictionaryPlot[classString] = resultingStd
    
    
    dataFramePlot = pandas.DataFrame(dictionaryPlot, index = waveNumbers)
    dataFramePlot.plot()
    plt.xlabel("wave numbers")
    plt.ylabel("absorbance")
    plt.show()
    plt.close()
    
    
    print("plot of the standard deviations of spectra per class for the test set")
    
    dictionaryPlot = dict()
    numberClassPair = [(0, "epithelium"), (1, "stroma"), (2, "blood"), (3, "necrosis")]
    
    for classNumber, classString in numberClassPair:
        maskFilter = (testSetOutput == classNumber)
        setInputFiltered = testSetInput[maskFilter, :]
        resultingStd = numpy.std(setInputFiltered, axis = 0)
        
        dictionaryPlot[classString] = resultingStd
    
    
    dataFramePlot = pandas.DataFrame(dictionaryPlot, index = waveNumbers)
    dataFramePlot.plot()
    plt.xlabel("wave numbers")
    plt.ylabel("absorbance")
    plt.show()
    plt.close()
    
    
    
    
    
    
    print("plot of the mean spectrum per class for the training set with standard deviations bands")
    
    dictionaryPlot = dict()
    numberClassPair = [(0, "epithelium"), (1, "stroma"), (2, "blood"), (3, "necrosis")]
    
    for classNumber, classString in numberClassPair:
        maskFilter = (trainingSetOutput == classNumber)
        setInputFiltered = trainingSetInput[maskFilter, :]
        resultingMean = numpy.mean(setInputFiltered, axis = 0)
        
        dictionaryPlot[classString] = resultingMean
    
    
    dataFramePlot = pandas.DataFrame(dictionaryPlot, index = waveNumbers)
    dataFramePlot.plot()
    plt.xlabel("wave numbers")
    plt.ylabel("absorbance")
    
    
    for classNumber, classString in numberClassPair:
        maskFilter = (trainingSetOutput == classNumber)
        setInputFiltered = trainingSetInput[maskFilter, :]
        resultingStd = numpy.std(setInputFiltered, axis = 0)
        
        plt.fill_between(waveNumbers, dictionaryPlot[classString] - resultingStd, dictionaryPlot[classString] + resultingStd, alpha = alphaValue)
    
    
    plt.show()
    plt.close()
    
    
    print("plot of the mean spectrum per class for the test set with standard deviations bands")
    
    dictionaryPlot = dict()
    numberClassPair = [(0, "epithelium"), (1, "stroma"), (2, "blood"), (3, "necrosis")]
    
    for classNumber, classString in numberClassPair:
        maskFilter = (testSetOutput == classNumber)
        setInputFiltered = testSetInput[maskFilter, :]
        resultingMean = numpy.mean(setInputFiltered, axis = 0)
        
        dictionaryPlot[classString] = resultingMean
    
    
    dataFramePlot = pandas.DataFrame(dictionaryPlot, index = waveNumbers)
    dataFramePlot.plot()
    plt.xlabel("wave numbers")
    plt.ylabel("absorbance")
    
    
    for classNumber, classString in numberClassPair:
        maskFilter = (testSetOutput == classNumber)
        setInputFiltered = testSetInput[maskFilter, :]
        resultingStd = numpy.std(setInputFiltered, axis = 0)
        
        plt.fill_between(waveNumbers, dictionaryPlot[classString] - resultingStd, dictionaryPlot[classString] + resultingStd, alpha = alphaValue)
    
    
    plt.show()
    plt.close()
    
    
    
    
    numberClassPair = [(0, "epithelium"), (1, "stroma"), (2, "blood"), (3, "necrosis")]
    
    for classNumber, classString in numberClassPair:
        
        print("plot of the mean spectrum of the training and test set for the {} class".format(classString))
        
        dictionaryPlot = dict()
        
        maskFilter = (trainingSetOutput == classNumber)
        setInputFiltered = trainingSetInput[maskFilter, :]
        resultingMean = numpy.mean(setInputFiltered, axis = 0)
        
        dictionaryPlot["training set"] = resultingMean
        
        
        maskFilter = (testSetOutput == classNumber)
        setInputFiltered = testSetInput[maskFilter, :]
        resultingMean = numpy.mean(setInputFiltered, axis = 0)
        
        dictionaryPlot["test set"] = resultingMean
        
        
        dataFramePlot = pandas.DataFrame(dictionaryPlot, index = waveNumbers)
        dataFramePlot.plot()
        plt.xlabel("wave numbers")
        plt.ylabel("absorbance")
        plt.show()
        plt.close()
    
    
    
    numberClassPair = [(0, "epithelium"), (1, "stroma"), (2, "blood"), (3, "necrosis")]
    
    for classNumber, classString in numberClassPair:
        
        print("plot of the standard deviations of the spectra of the training and test set for the {} class".format(classString))
        
        dictionaryPlot = dict()
        
        maskFilter = (trainingSetOutput == classNumber)
        setInputFiltered = trainingSetInput[maskFilter, :]
        resultingStd = numpy.std(setInputFiltered, axis = 0)
        
        dictionaryPlot["training set"] = resultingStd
        
        
        maskFilter = (testSetOutput == classNumber)
        setInputFiltered = testSetInput[maskFilter, :]
        resultingStd = numpy.std(setInputFiltered, axis = 0)
        
        dictionaryPlot["test set"] = resultingStd
        
        
        dataFramePlot = pandas.DataFrame(dictionaryPlot, index = waveNumbers)
        dataFramePlot.plot()
        plt.xlabel("wave numbers")
        plt.ylabel("absorbance")
        plt.show()
        plt.close()





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
    
    spectraSet, waveNumbers, setInput, setOutput, setFileNumbers, setPositions = load_each_data_cube_and_catenate_each_class(dataCubeFiles, dataCubePath, annotationsPositionsSet)
    
    return spectraSet, waveNumbers, setInput, setOutput, setFileNumbers, setPositions, annotationsPositionsSet, dataCubeFiles



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
    spectraAllClassFileNumbers = []
    spectraAllClassPositions = []
    i = 0
    while i < len(annotationsPositionsSet[0]):
        spectraAllClass.append(0)
        
        spectraAllClassFileNumbers.append(0)
        spectraAllClassPositions.append(0)
        
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
                
                spectraAllClassFileNumbers[j] = numpy.full((spectraSelected.shape[0],), i)
                spectraAllClassPositions[j] = copy.deepcopy(annotationsPositionsSet[i][j])
                
            
            else:
                spectraAllClass[j] = numpy.vstack((spectraAllClass[j], spectraSelected))
                
                spectraAllClassFileNumbers[j] = numpy.concatenate((spectraAllClassFileNumbers[j], numpy.full((spectraSelected.shape[0],), i)))
                spectraAllClassPositions[j] = numpy.vstack((spectraAllClassPositions[j], copy.deepcopy(annotationsPositionsSet[i][j])))
                
            
            j += 1
        
        
        spectraTrainingSet.append(spectraEachColor)
        
        i += 1
    
    
    spectraAllClassConcatenated = spectraAllClass[0]
    spectraCorrespondingClass = numpy.full((spectraAllClass[0].shape[0],), 0)
    
    spectraCorrespondingFileNumber = spectraAllClassFileNumbers[0]
    spectraCorrespondingPosition = spectraAllClassPositions[0]
    
    i = 1
    while i < len(annotationsPositionsSet[0]):
        spectraAllClassConcatenated = numpy.vstack((spectraAllClassConcatenated, spectraAllClass[i]))
        
        spectraCorrespondingClass = numpy.concatenate((spectraCorrespondingClass, numpy.full((spectraAllClass[i].shape[0],), i)))
        
        spectraCorrespondingFileNumber = numpy.concatenate((spectraCorrespondingFileNumber, spectraAllClassFileNumbers[i]))
        spectraCorrespondingPosition = numpy.vstack((spectraCorrespondingPosition, spectraAllClassPositions[i]))
        
        i += 1
    
    
    return spectraTrainingSet, waveNumbers, spectraAllClassConcatenated, spectraCorrespondingClass, spectraCorrespondingFileNumber, spectraCorrespondingPosition



def quality_test(setInput, setOutput, waveNumbers, peakWave, troughWave, minimum, maximum, setFileNumbers, setPositions):
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
    
    setFileNumbers = setFileNumbers[keptValues]
    setPositions = setPositions[keptValues, :]
    
    return setInput, setOutput, setFileNumbers, setPositions



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



if __name__ == "__main__":
    """
    See master_thesis_code_explanations.pdf for details on the parameters.
    """
    
    
    OVERLAYPATH = "path/to/overlay/folder"
    DATACUBEPATH = "path/to/datacube/folder"
    KFOLDINDICESPATH = "./splitting_file.mat"
    COLOURS = numpy.array([[103, 193, 66], [255, 0, 0], [165, 55, 156], [243, 143, 53], [194, 243, 174], [69, 120, 236]])
    PEAKWAVEQUALITYTEST = 1655
    TROUGHWAVEQUALITYTEST = 1760
    MINIMUMQUALITYTEST = 0.1
    MAXIMUMQUALITYTEST = 2
    MAKEQUALITYTESTTRAININGSET = True
    MAKEQUALITYTESTTESTSET = True
    
    DICTIONARYCLASSESMAPPING = {(0, 1) : 0, (2, 3) : 1, (4,) : 2, (5,) : 3}
    
    ALPHAVALUE = 0.1
    
    model_for_identifying_histology(OVERLAYPATH, DATACUBEPATH, KFOLDINDICESPATH, COLOURS, PEAKWAVEQUALITYTEST, TROUGHWAVEQUALITYTEST, MINIMUMQUALITYTEST, MAXIMUMQUALITYTEST, DICTIONARYCLASSESMAPPING, MAKEQUALITYTESTTRAININGSET, MAKEQUALITYTESTTESTSET, ALPHAVALUE)

