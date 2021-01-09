#cython: language_level=3

import numpy


cimport libc.stdio
cimport cython







#@cython.boundscheck(False)
#@cython.wraparound(False)
def create_set_subwindows(setInput, setOutput, setFileNumbers, setPositions, completeSetInput, completeSetOutput, completeSetFileNumbers, completeSetPositions, widthSubwindow, heightSubwindow, methodSubwindow):
    
    #print("setInput shape: {}".format(setInput.shape))
    #print("setOutput shape: {}".format(setOutput.shape))
    #print("setFileNumbers shape: {}".format(setFileNumbers.shape))
    #print("setPositions shape: {}".format(setPositions.shape))
    #print("completeSetInput shape: {}".format(completeSetInput.shape))
    #print("completeSetOutput shape: {}".format(completeSetOutput.shape))
    #print("completeSetFileNumbers shape: {}".format(completeSetFileNumbers.shape))
    #print("completeSetPositions shape: {}".format(completeSetPositions.shape))
    
    
    indicesSorted = numpy.lexsort((completeSetPositions[:, 1], completeSetPositions[:, 0], completeSetFileNumbers))
    
    completeSetInput = completeSetInput[indicesSorted, :]
    completeSetOutput = completeSetOutput[indicesSorted]
    completeSetFileNumbers = completeSetFileNumbers[indicesSorted]
    completeSetPositions = completeSetPositions[indicesSorted, :]
    
    
    setInput = numpy.ascontiguousarray(setInput.astype(numpy.single))
    setOutput = numpy.ascontiguousarray(setOutput.astype(numpy.intc))
    setFileNumbers = numpy.ascontiguousarray(setFileNumbers.astype(numpy.intc))
    setPositions = numpy.ascontiguousarray(setPositions.astype(numpy.intc))
    
    completeSetInput = numpy.ascontiguousarray(completeSetInput.astype(numpy.single))
    completeSetOutput = numpy.ascontiguousarray(completeSetOutput.astype(numpy.intc))
    completeSetFileNumbers = numpy.ascontiguousarray(completeSetFileNumbers.astype(numpy.intc))
    completeSetPositions = numpy.ascontiguousarray(completeSetPositions.astype(numpy.intc))
    
    
    cdef float[:, ::1] setInputView = setInput
    cdef int[::1] setOutputView = setOutput
    cdef int[::1] setFileNumbersView = setFileNumbers
    cdef int[:, ::1] setPositionsView = setPositions
    
    cdef float[:, ::1] completeSetInputView = completeSetInput
    cdef int[::1] completeSetOutputView = completeSetOutput
    cdef int[::1] completeSetFileNumbersView = completeSetFileNumbers
    cdef int[:, ::1] completeSetPositionsView = completeSetPositions
    
    cdef int widthSubwindowCVariable = widthSubwindow
    cdef int heightSubwindowCVariable = heightSubwindow
    
    
    cdef Py_ssize_t i = 0
    cdef int j = 0
    cdef int k = 0
    cdef Py_ssize_t m = 0
    
    cdef Py_ssize_t numberRowsSets = setPositions.shape[0]
    cdef Py_ssize_t numberRowsCompleteSets = completeSetPositions.shape[0]
    cdef Py_ssize_t numberColumnsInputSets = setInput.shape[1]
    
    positionCentralPixel = numpy.zeros((2,), dtype = numpy.intc)
    
    cdef int[::1] positionCentralPixelView = positionCentralPixel
    
    cdef int minimumRow = 0
    cdef int maximumRow = 0
    cdef int minimumColumn = 0
    cdef int maximumColumn = 0
    
    
    cdef Py_ssize_t beginIndexRow = 0
    cdef Py_ssize_t endIndexRow = 0
    cdef Py_ssize_t currentIndexRow = 0
    cdef Py_ssize_t columnIndex = 0
    
    cdef int currentFileNumber = 0
    
    cdef int errorOccured = 0
    
    
    cdef int methodSubwindowCode = 0
    
    if methodSubwindow == "subwindow_center":
        methodSubwindowCode = 0
    
    elif methodSubwindow == "subwindow_multioutput":
        methodSubwindowCode = 1
    
    else:
        return None, None
    
    
    finalSetInput = numpy.zeros((setInput.shape[0], setInput.shape[1] * widthSubwindow * heightSubwindow), dtype = numpy.single)
    
    cdef float[:, ::1] finalSetInputView = finalSetInput
    
    
    finalSetOutput = 0
    finalSetOutputMultioutput = 0
    
    if methodSubwindow == "subwindow_center":
        finalSetOutput = numpy.zeros((setInput.shape[0],), dtype = numpy.intc)
        finalSetOutputMultioutput = numpy.zeros((1, 1), dtype = numpy.intc)
    
    elif methodSubwindow == "subwindow_multioutput":
        finalSetOutputMultioutput = numpy.zeros((setInput.shape[0], widthSubwindow * heightSubwindow), dtype = numpy.intc)
        finalSetOutput = numpy.zeros((1,), dtype = numpy.intc)
    
    
    cdef int[::1] finalSetOutputView = finalSetOutput
    cdef int[:, ::1] finalSetOutputMultioutputView = finalSetOutputMultioutput
    
    
    #libc.stdio.printf("numberRowsSets: %zd, %zd\n", numberRowsSets, numberRowsCompleteSets)
    
    while (i < numberRowsSets) and (errorOccured == 0):
        #libc.stdio.printf("i: %zd\n", i)
        
        if methodSubwindowCode == 0:
            finalSetOutputView[i] = setOutputView[i]
        
        positionCentralPixelView = setPositionsView[i, :]
        
        if (heightSubwindowCVariable % 2) == 1:
            minimumRow = positionCentralPixelView[0] - (heightSubwindowCVariable // 2)
            maximumRow = positionCentralPixelView[0] + (heightSubwindowCVariable // 2)
        
        else:
            minimumRow = positionCentralPixelView[0] - (heightSubwindowCVariable // 2) + 1
            maximumRow = positionCentralPixelView[0] + (heightSubwindowCVariable // 2)
        
        
        if (widthSubwindowCVariable % 2) == 1:
            minimumColumn = positionCentralPixelView[1] - (widthSubwindowCVariable // 2)
            maximumColumn = positionCentralPixelView[1] + (widthSubwindowCVariable // 2)
        
        else:
            minimumColumn = positionCentralPixelView[1] - (widthSubwindowCVariable // 2) + 1
            maximumColumn = positionCentralPixelView[1] + (widthSubwindowCVariable // 2)
        
        
        currentFileNumber = setFileNumbersView[i]
        j = minimumRow
        while (j <= maximumRow) and (errorOccured == 0):
            
            k = minimumColumn
            while (k <= maximumColumn) and (errorOccured == 0):
                
                beginIndexRow = 0
                endIndexRow = numberRowsCompleteSets
                while beginIndexRow != endIndexRow:
                    
                    currentIndexRow = (endIndexRow + beginIndexRow) // 2
                    
                    if completeSetFileNumbersView[currentIndexRow] == currentFileNumber:
                        if completeSetPositionsView[currentIndexRow, 0] == j:
                            if completeSetPositionsView[currentIndexRow, 1] == k:
                                
                                m = 0
                                while m < numberColumnsInputSets:
                                
                                    columnIndex = ((j - minimumRow) * widthSubwindowCVariable + (k - minimumColumn)) * numberColumnsInputSets + m
                                    finalSetInputView[i, columnIndex] = completeSetInputView[currentIndexRow, m]
                                    
                                    m += 1
                                
                                
                                if methodSubwindowCode == 1:
                                    columnIndex = ((j - minimumRow) * widthSubwindowCVariable + (k - minimumColumn))
                                    finalSetOutputMultioutputView[i, columnIndex] = completeSetOutputView[currentIndexRow]
                                
                                break
                            
                            elif completeSetPositionsView[currentIndexRow, 1] < k:
                                beginIndexRow = currentIndexRow + 1
                            
                            else:
                                endIndexRow = currentIndexRow
                        
                        elif completeSetPositionsView[currentIndexRow, 0] < j:
                            beginIndexRow = currentIndexRow + 1
                        
                        else:
                            endIndexRow = currentIndexRow
                    
                    elif completeSetFileNumbersView[currentIndexRow] < currentFileNumber:
                        beginIndexRow = currentIndexRow + 1
                    
                    else:
                        endIndexRow = currentIndexRow
                
                
                if beginIndexRow == endIndexRow:
                    errorOccured = 1
                
                
                k += 1
            
            j += 1
        
        i += 1
    
    
    if errorOccured == 1:
        return None, None
    
    elif methodSubwindow == "subwindow_multioutput":
        return finalSetInput, finalSetOutputMultioutput
    
    
    return finalSetInput, finalSetOutput





#@cython.boundscheck(False)
#@cython.wraparound(False)
def create_final_predictions_multioutput(setFileNumbers, setPositions, finalPredictions, completeSetFileNumbers, completeSetPositions, widthSubwindow, heightSubwindow, methodPrediction):
    
    indicesSorted = numpy.lexsort((completeSetPositions[:, 1], completeSetPositions[:, 0], completeSetFileNumbers))
    
    
    if methodPrediction == "hard":
        finalPredictionsHard = finalPredictions[indicesSorted, :]
    
    elif methodPrediction == "soft":
        finalPredictionsSoft = [predictions[indicesSorted, :] for predictions in finalPredictions]
    
    
    completeSetFileNumbers = completeSetFileNumbers[indicesSorted]
    completeSetPositions = completeSetPositions[indicesSorted, :]
    
    
    setFileNumbers = numpy.ascontiguousarray(setFileNumbers.astype(numpy.intc))
    setPositions = numpy.ascontiguousarray(setPositions.astype(numpy.intc))
    
    
    if methodPrediction == "hard":
        finalPredictionsHard = numpy.ascontiguousarray(finalPredictionsHard.astype(numpy.intc))
        finalPredictionsSoft = numpy.zeros((1, 1, 1), dtype = numpy.single)
    
    elif methodPrediction == "soft":
        finalPredictionsSoft = numpy.ascontiguousarray(numpy.array(finalPredictionsSoft).astype(numpy.single))
        finalPredictionsHard = numpy.zeros((1, 1), dtype = numpy.intc)
    
    
    completeSetFileNumbers = numpy.ascontiguousarray(completeSetFileNumbers.astype(numpy.intc))
    completeSetPositions = numpy.ascontiguousarray(completeSetPositions.astype(numpy.intc))
    
    
    cdef int[::1] setFileNumbersView = setFileNumbers
    cdef int[:, ::1] setPositionsView = setPositions
    
    cdef int[:, ::1] finalPredictionsHardView = finalPredictionsHard
    cdef float[:, :, ::1] finalPredictionsSoftView = finalPredictionsSoft
    
    cdef int[::1] completeSetFileNumbersView = completeSetFileNumbers
    cdef int[:, ::1] completeSetPositionsView = completeSetPositions
    
    
    cdef int widthSubwindowCVariable = widthSubwindow
    cdef int heightSubwindowCVariable = heightSubwindow
    
    cdef int errorOccured = 0
    
    cdef int methodPredictionCode = 0
    
    if methodPrediction == "hard":
        methodPredictionCode = 0
    
    elif methodPrediction == "soft":
        methodPredictionCode = 1
    
    else:
        return None
    
    
    numberClasses = 0
    
    if methodPrediction == "hard":
        numberClasses = finalPredictionsHard.max() + 1
    
    elif methodPrediction == "soft":
        numberClasses = finalPredictionsSoft.shape[2]
    
    
    cdef int numberClassesCVariable = numberClasses
    
    cdef int numberRowsSets = setFileNumbers.shape[0]
    cdef int numberRowsCompleteSets = completeSetFileNumbers.shape[0]
    
    
    outputsEachPixel = numpy.zeros((setFileNumbers.shape[0],), dtype = numpy.intc)
    
    cdef int[::1] outputsEachPixelView = outputsEachPixel
    
    
    outputGivenPixelHard = numpy.zeros((numberClasses,), dtype = numpy.intc)
    outputGivenPixelSoft = numpy.zeros((numberClasses,), dtype = numpy.single)
    
    cdef int[::1] outputGivenPixelHardView = outputGivenPixelHard
    cdef float[::1] outputGivenPixelSoftView = outputGivenPixelSoft
    
    
    cdef Py_ssize_t i = 0
    cdef int j = 0
    cdef int k = 0
    cdef Py_ssize_t n = 0
    
    positionCentralPixel = numpy.zeros((2,), dtype = numpy.intc)
    
    cdef int[::1] positionCentralPixelView = positionCentralPixel
    
    cdef int minimumRow = 0
    cdef int maximumRow = 0
    cdef int minimumColumn = 0
    cdef int maximumColumn = 0
    
    
    cdef Py_ssize_t beginIndexRow = 0
    cdef Py_ssize_t endIndexRow = 0
    cdef Py_ssize_t currentIndexRow = 0
    
    cdef int currentFileNumber = 0
    
    
    cdef int topYPosition = 0
    cdef int leftXPosition = 0
    
    cdef Py_ssize_t outputIndex = 0
    
    cdef int classPredicted = 0
    
    cdef int maximumFoundHard = 0
    cdef float maximumFoundSoft = 0.0
    cdef int finalClass = 0
    
    
    while (i < numberRowsSets) and (errorOccured == 0):
        #libc.stdio.printf("i: %zd\n", i)
        
        positionCentralPixelView = setPositionsView[i, :]
        
        if (heightSubwindowCVariable % 2) == 1:
            minimumRow = positionCentralPixelView[0] - (heightSubwindowCVariable // 2)
            maximumRow = positionCentralPixelView[0] + (heightSubwindowCVariable // 2)
        
        else:
            minimumRow = positionCentralPixelView[0] - (heightSubwindowCVariable // 2)
            maximumRow = positionCentralPixelView[0] + (heightSubwindowCVariable // 2) - 1
        
        
        if (widthSubwindowCVariable % 2) == 1:
            minimumColumn = positionCentralPixelView[1] - (widthSubwindowCVariable // 2)
            maximumColumn = positionCentralPixelView[1] + (widthSubwindowCVariable // 2)
        
        else:
            minimumColumn = positionCentralPixelView[1] - (widthSubwindowCVariable // 2)
            maximumColumn = positionCentralPixelView[1] + (widthSubwindowCVariable // 2) - 1
        
        
        currentFileNumber = setFileNumbersView[i]
        j = minimumRow
        while (j <= maximumRow) and (errorOccured == 0):
            
            k = minimumColumn
            while (k <= maximumColumn) and (errorOccured == 0):
                
                beginIndexRow = 0
                endIndexRow = numberRowsCompleteSets
                while beginIndexRow != endIndexRow:
                    
                    currentIndexRow = (endIndexRow + beginIndexRow) // 2
                    
                    if completeSetFileNumbersView[currentIndexRow] == currentFileNumber:
                        if completeSetPositionsView[currentIndexRow, 0] == j:
                            if completeSetPositionsView[currentIndexRow, 1] == k:
                                
                                if (heightSubwindowCVariable % 2) == 1:
                                    topYPosition = j - (heightSubwindowCVariable // 2)
                                
                                else:
                                    topYPosition = j - (heightSubwindowCVariable // 2) + 1
                                
                                
                                if (widthSubwindowCVariable % 2) == 1:
                                    leftXPosition = k - (widthSubwindowCVariable // 2)
                                
                                else:
                                    leftXPosition = k - (widthSubwindowCVariable // 2) + 1
                                
                                
                                outputIndex = (positionCentralPixelView[0] - topYPosition) * widthSubwindowCVariable + (positionCentralPixelView[1] - leftXPosition)
                                
                                if methodPredictionCode == 0:
                                    classPredicted = finalPredictionsHardView[currentIndexRow, outputIndex]
                                    
                                    #libc.stdio.printf("classPredicted: %d\n", classPredicted)
                                    #print("finalPredictionsHard shape: {}".format(finalPredictionsHard.shape))
                                    
                                    outputGivenPixelHardView[classPredicted] += 1
                                
                                elif methodPredictionCode == 1:
                                    n = 0
                                    while n < numberClassesCVariable:
                                        outputGivenPixelSoftView[n] += finalPredictionsSoftView[outputIndex, currentIndexRow, n]
                                        
                                        n += 1
                                
                                break
                                
                                
                                
                            
                            elif completeSetPositionsView[currentIndexRow, 1] < k:
                                beginIndexRow = currentIndexRow + 1
                            
                            else:
                                endIndexRow = currentIndexRow
                        
                        elif completeSetPositionsView[currentIndexRow, 0] < j:
                            beginIndexRow = currentIndexRow + 1
                        
                        else:
                            endIndexRow = currentIndexRow
                    
                    elif completeSetFileNumbersView[currentIndexRow] < currentFileNumber:
                        beginIndexRow = currentIndexRow + 1
                    
                    else:
                        endIndexRow = currentIndexRow
                
                
                if beginIndexRow == endIndexRow:
                    errorOccured = 1
                
                
                k += 1
            
            j += 1
        
        
        maximumFoundHard = -1
        maximumFoundSoft = -1.0
        finalClass = -1
        n = 0
        if methodPredictionCode == 0:
            while n < numberClassesCVariable:
                if outputGivenPixelHardView[n] > maximumFoundHard:
                    maximumFoundHard = outputGivenPixelHardView[n]
                    finalClass = n
                
                n += 1
        
        elif methodPredictionCode == 1:
            while n < numberClassesCVariable:
                if outputGivenPixelSoftView[n] > maximumFoundSoft:
                    maximumFoundSoft = outputGivenPixelSoftView[n]
                    finalClass = n
                
                n += 1
        
        outputsEachPixelView[i] = finalClass
        
        i += 1
    
    
    if errorOccured == 1:
        return None
    
    
    return outputsEachPixel

