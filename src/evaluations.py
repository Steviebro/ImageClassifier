import numpy as np

def getConfusionMatrix(groundTruths, predictions):
    confusionMatrix = np.zeros((10, 10), dtype=int)
    for i, groundTruth in enumerate(groundTruths):
        confusionMatrix[groundTruth, predictions[i]] += 1
    return confusionMatrix

def getAccuracy(groundTruths, predictions):
    correct = 0
    for i, groundTruth in enumerate(groundTruths):
        if groundTruth == predictions[i]:
            correct += 1
    return correct / len(groundTruths)

def getPrecisionAndRecall(confusionMatrix):
    precision = []
    recall = []
    for i in range(10):
        truePositives = confusionMatrix[i, i]
        falsePositives = np.sum(confusionMatrix[:,i]) - truePositives
        falseNegatives = np.sum(confusionMatrix[i, :]) - truePositives
        if truePositives + falsePositives == 0:
            precision.append(0)
        else:
            precision.append(truePositives / (truePositives + falsePositives))
        if truePositives + falseNegatives == 0:
            recall.append(0)
        else:
            recall.append(truePositives / (truePositives + falseNegatives))
    return np.mean(precision), np.mean(recall)

def getF1(precision, recall):
    if precision + recall == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)

def evaluate(groundTruths, predictions):
    confusionMatrix = getConfusionMatrix(groundTruths, predictions)
    accuracy = getAccuracy(groundTruths, predictions)
    precision, recall = getPrecisionAndRecall(confusionMatrix)
    f1 = getF1(precision, recall)
    print("Evaluation:===================+++++++++++++")
    print("Confusion Matrix:")
    print(confusionMatrix)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print("++++++++++++===================+++++++++++++")
    return confusionMatrix, accuracy, precision, recall, f1
