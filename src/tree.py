import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def computeGini(data):
    classes = set(i[1] for i in data)
    gini = 0
    for i in classes:
        subset_i = [j for j in data if j[1] == i] #subset of data belonging to class i
        p_mk = len(subset_i)/len(data)
        gini += p_mk * (1 - p_mk)
    return gini

def splitData(data, featureIndex, splitPoint):
    left = [i for i in data if i[0][featureIndex].numpy() < splitPoint]
    right = [i for i in data if i[0][featureIndex].numpy() >= splitPoint]
    return left, right

def bestSplit(data):
    lowestGini = float('inf')
    bestFeature = None
    bestSplitPoint = None
    numOfFeatures = len(data[0][0])
    dataGini = computeGini(data)

    for featureIndex in range(numOfFeatures):
        featureValueSet = set(j[0][featureIndex] for j in data)
        featureValueSetMaximum = max(featureValueSet)
        featureValueSetMinimum = min(featureValueSet)
        for splitPoint in range(int(featureValueSetMinimum.item()), int(featureValueSetMaximum.item())+1):
            leftSplit, rightSplit = splitData(data, featureIndex, splitPoint)
            giniLeft, giniRight = computeGini(leftSplit), computeGini(rightSplit)
            gini = giniLeft * (len(leftSplit) / len(data)) + giniRight * (len(rightSplit) / len(data))
            if gini < lowestGini:
                lowestGini = gini
                bestFeature = featureIndex
                bestSplitPoint = splitPoint
            # print("gini",gini,"gini gain",dataGini-gini, "testing feature index:", featureIndex, "splitPoint:", splitPoint)


    # Return the best split if it has significant gain
    print("gini gain from best split:",dataGini-lowestGini)
    if dataGini - lowestGini > 0.01:
        return bestFeature, bestSplitPoint
    else:
        return None, None

def majorityClass(data):
    classes = set(i[1] for i in data)
    numOfElements = len(data)
    highestP_i = 0.0
    majority = None
    for i in classes:
        subset_i = [j for j in data if j[1] == i]
        p_i = len(subset_i)/numOfElements
        if p_i > highestP_i:
            highestP_i = p_i
            majority = i

    return majority

def buildTree(data, depth, maxDepth):
    if depth >= maxDepth or computeGini(data) == 0 or len(data) < 20:
        return {
            'nodeType': 'leaf',
            'prediction': majorityClass(data)
        }

    featureIndex, splitPoint = bestSplit(data)

    if featureIndex is None:
        return {
            'nodeType': 'leaf',
            'prediction': majorityClass(data)
        }

    leftSplit, rightSplit = splitData(data, featureIndex, splitPoint)
    return {
        'nodeType': 'parent',
        'feature': featureIndex,
        'splitPoint': splitPoint,
        'leftChild': buildTree(leftSplit, depth + 1, maxDepth),
        'rightChild': buildTree(rightSplit, depth + 1, maxDepth),
    }

def saveTreeModel(decisionTree, fileName):
    with open('../models/tree/'+fileName+'.pkl', 'wb') as file:
        pickle.dump(decisionTree, file)

def trainTreeModel(trainingSet, maxDepth, fileName):
    tree = buildTree(trainingSet, 0, maxDepth)
    saveTreeModel(tree, fileName)
    return tree

def loadTreeModel(fileName):
    with open('../models/tree/'+fileName+'.pkl', 'rb') as file:
        return pickle.load(file)

def predictClass(x, decisionTree):
    if decisionTree['nodeType'] == 'leaf':
        return decisionTree['prediction']

    if x[decisionTree['feature']] < decisionTree['splitPoint']:
        return predictClass(x, decisionTree['leftChild'])
    else:
        return predictClass(x, decisionTree['rightChild'])

def testTreeModel(decisionTree, testingSet):
    dataTest = np.array([data[0].numpy() for data in testingSet])
    groundTruths = np.array([data[1] for data in testingSet])
    predictions = []

    for x in dataTest:
        predictions.append(predictClass(x, decisionTree))

    return groundTruths, predictions

### Scikit

def trainScikitTreeModel(trainingSet, maxDepth, fileName):
    scikitTreeModel = DecisionTreeClassifier(max_depth=maxDepth,criterion='entropy',min_samples_split=20,min_samples_leaf=20)
    data = np.array([data[0].numpy() for data in trainingSet])
    classes = np.array([data[1] for data in trainingSet])

    scikitTreeModel.fit(data, classes)
    saveTreeModel(scikitTreeModel, fileName)
    return scikitTreeModel

def testScikitTreeModel(scikitTreeModel, testingSet):
    dataTest = np.array([data[0].numpy() for data in testingSet])
    groundTruths = np.array([data[1] for data in testingSet])
    predictions = scikitTreeModel.predict(dataTest)
    return groundTruths, predictions