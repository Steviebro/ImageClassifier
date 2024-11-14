import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


def trainGaussianModel(trainingSet):
    means = []
    variances = []

    for i in range(10):
        classData = np.array([data[0].numpy() for data in trainingSet if data[1]==i]) # Extract all data from class i

        classMeans = np.mean(classData, axis=0)
        classVariances = np.var(classData, axis=0, ddof=1)
        means.append(classMeans)
        variances.append(classVariances)

    return means, variances

def saveModel(means, variances):
    gaussianModel = {
        'means': means,
        'variances': variances
    }
    with open('./models/gaussianModel.pkl', 'wb') as file:
        pickle.dump(gaussianModel, file)

def loadModel():
    with open('./models/gaussianModel.pkl', 'rb') as file:
        gaussianModel = pickle.load(file)

    means = gaussianModel['means']
    variances = gaussianModel['variances']
    return means, variances

def gaussianLikelihood(x, mu, sigma):
    exp = np.exp(-((x - mu) ** 2) / (2 * sigma))
    likelihood = (1 / np.sqrt(2 * np.pi * sigma)) * exp
    return likelihood

def predict(x, means, variances):
    logProbabilities = []

    for i in range(10):
        logLikelihood = np.sum(np.log(gaussianLikelihood(x, means[i], variances[i])))
        logProbabilities.append(logLikelihood)

    return np.argmax(logProbabilities)


def test(means, variances, testingSet):
    correct = 0
    for x, label in testingSet:
        x = x.numpy()
        predictedLabel = predict(x, means, variances)
        if predictedLabel == label:
            correct += 1

    accuracy = correct / len(testingSet)
    print(f"Manual Gaussian Naive Bayes Implementation Accuracy: {accuracy * 100:.2f}%")


def scikitClassifier(trainingSet, testingSet):
    gnb = GaussianNB()

    data = np.array([data[0].numpy() for data in trainingSet])
    classes = np.array([data[1] for data in trainingSet])

    gnb.fit(data, classes)

    dataTest = np.array([data[0].numpy() for data in testingSet])
    classesTest = np.array([data[1] for data in testingSet])

    pred = gnb.predict(dataTest)

    accuracy = accuracy_score(classesTest, pred)
    print(f"Scilearn Gaussian Naive Bayes Implementation Accuracy: {accuracy * 100:.2f}%")