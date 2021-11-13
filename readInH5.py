## install libraries for import
try:
    from pip import main as pipmain
except ImportError:
    from pip._internal import main as pipmain

pipmain(["install", 'h5py'])
pipmain(["install", 'numpy'])

## import libraries
import h5py
import numpy as np

## set number of spectra and path to the data files
# os.chdir("f:/Data/EMSLIBS_CONTEST/")   # selecting the directory containing the data files
spectraCount = 200  # selecting the number of spectra for each sample (maximum of 500)

##########################################
# Train Data
##########################################

print('Read in datafile')
trainFile = h5py.File("train.h5", 'r')  # training data, unless the filename was changed

wavelengths = trainFile["Wavelengths"]
print(wavelengths[list(wavelengths.keys())[0]])
wavelengths = wavelengths[list(wavelengths.keys())[0]][()]
# creates a one-dimensional array (vector) containing the wavelengths

print('Iterate through keys, find trainData')
for sample in list(trainFile["Spectra"].keys()):
    tempData = trainFile["Spectra"][sample][()]
    tempData = tempData[:, 0:spectraCount]
    if "trainData" not in locals():
        trainData = tempData.transpose()
    else:
        trainData = np.append(trainData, tempData.transpose(), axis=0)
# creates a two-dimensional array (matrix) containing the training data
# each row represents a single spectrum

print('Get trainClass')
trainClass = trainFile["Class"]["1"][()]
for i in range(0, 50000, 500):
    if i == 0:
        tempClass = trainClass[0:spectraCount]
    else:
        tempClass = np.append(tempClass, trainClass[i:(i + spectraCount)])
trainClass = tempClass
# creates a one-dimensional array (vector) containing the classes corresponding to the spectra in the training dataset
# the order of the classes is the same as the order of the spectra
# if the dataset is reordered (e.g., randomly sampled), make sure that the classes are reordered accordingly
trainFile.close()
del tempClass, tempData, i, sample

##########################################
# Test Data
##########################################

print('Load testFile')
testFile = h5py.File("test.h5", 'r')  # testing data, unless the filename was changed

print('Iterate through keys and set testData')
for sample in list(testFile["UNKNOWN"].keys()):
    tempData = testFile["UNKNOWN"][sample][()]

    if "testData" not in locals():
        testData = tempData.transpose()
    else:
        testData = np.append(testData, tempData.transpose(), axis=0)

# creates a two-dimensional array (matrix) containing the testing data
# each row represents a single spectrum
testFile.close()
del tempData, sample, spectraCount

print('trainData', trainData.shape, trainData)
print('trainClass', trainClass.shape, trainClass)
print('testData', testData.shape, testData)
print('waveLengths', wavelengths.shape, wavelengths)

# TODO dump as CSV's

##########################################
# End of loading script
##########################################
# Returns 4 variables -> trainData, trainClass, testData, wavelengths
##########################################
