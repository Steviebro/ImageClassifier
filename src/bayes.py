import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB

def saveBayesModel(bayesModel, fileName):
    with open('../models/bayes/'+fileName+'.pkl', 'wb') as file:
        pickle.dump(bayesModel, file)

def loadBayesModel(fileName):
    with open('../models/bayes/'+fileName+'.pkl', 'rb') as file:
        return pickle.load(file)

def trainBayesModel(trainingSet, fileName):
    means = []
    variances = []

    for i in range(10):
        classData = np.array([data[0].numpy() for data in trainingSet if data[1]==i]) # Extract all data from class i

        classMeans = np.mean(classData, axis=0)
        classVariances = np.var(classData, axis=0, ddof=1)
        means.append(classMeans)
        variances.append(classVariances)

    bayesModel = {
        'means': means,
        'variances': variances
    }

    saveBayesModel(bayesModel, fileName)
    return bayesModel

# Computes the gaussian likelihood (P(D|h))
def gaussianLikelihood(x, mu, sigma):
    exp = np.exp(-((x - mu) ** 2) / (2 * sigma))
    likelihood = (1 / np.sqrt(2 * np.pi * sigma)) * exp
    return likelihood

# Makes a class prediction for a given x: the class with the highest value for the sum of logs of gaussian likelihood
def predictClass(x, means, variances):
    scores = []

    for i in range(10):
        logLikelihood = np.sum(np.log(gaussianLikelihood(x, means[i], variances[i])))
        scores.append(logLikelihood)

    return np.argmax(scores)

# Tests the model against a testing set
def testBayesModel(bayesModel, testingSet):
    means = bayesModel['means']
    variances = bayesModel['variances']

    dataTest = np.array([data[0].numpy() for data in testingSet])
    groundTruths = np.array([data[1] for data in testingSet])
    predictions = []

    for x in dataTest:
        predictions.append(predictClass(x, means, variances))
    return groundTruths, predictions


def trainScikitBayesModel(trainingSet):
    gnb = GaussianNB()
    data = np.array([data[0].numpy() for data in trainingSet])
    classes = np.array([data[1] for data in trainingSet])
    gnb.fit(data, classes)
    saveBayesModel(gnb, 'scikitBayesModel')
    return gnb

def testScikitBayesModel(gnb, testingSet):
    dataTest = np.array([data[0].numpy() for data in testingSet])
    groundTruths = np.array([data[1] for data in testingSet])
    predictions = gnb.predict(dataTest)
    return groundTruths, predictions

