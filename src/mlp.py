import pickle
from venv import create

import torch as tr
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset

from sklearn.metrics import accuracy_score, confusion_matrix

def saveMLPModel(mlpModel, fileName):
    with open('../models/mlp/'+fileName+'.pkl', 'wb') as file:
        pickle.dump(mlpModel, file)

def loadMLPModel(fileName):
    with open('../models/mlp/'+fileName+'.pkl', 'rb') as file:
        return pickle.load(file)

# Sets between 2 and 4 layers; Duplicating or removing the middle layer as needed
def setLayers(numOfLayers, hiddenLayerSize):
    if numOfLayers < 2 or numOfLayers > 4:
        raise Exception("Invalid number of layers")

    layers = []
    layers.append(nn.Sequential(
        nn.Linear(50, hiddenLayerSize),
        nn.ReLU()
    ))
    if numOfLayers > 2:
        layers.append(nn.Sequential(
            nn.Linear(hiddenLayerSize, hiddenLayerSize),
            nn.BatchNorm1d(hiddenLayerSize),
            nn.ReLU()
        ))
    if numOfLayers > 3:
        layers.append(nn.Sequential(
            nn.Linear(hiddenLayerSize, hiddenLayerSize),
            nn.BatchNorm1d(hiddenLayerSize),
            nn.ReLU()
        ))
    layers.append(nn.Linear(hiddenLayerSize, 10))

    return nn.Sequential(*layers)

def createDataLoader(dataSet, batchSize):
    featureVectors = tr.stack([i[0] for i in dataSet], dim=0).float()
    classLabels = tr.tensor([i[1] for i in dataSet], dtype=tr.long)
    return tr.utils.data.DataLoader(TensorDataset(featureVectors, classLabels), batch_size=batchSize, shuffle=True)

def performTraining(mlpModel, dataLoader, lossCriterion, optimizer, numOfEpochs, device):
    averageLoss = 0.0
    for epoch in range(numOfEpochs):
        mlpModel.train()
        totalLoss = 0.0
        batch = 0
        for features, labels in dataLoader:
            features, labels = features.to(device), labels.to(device)

            # Find Outputs & Compute Loss
            outputs = mlpModel(features)
            loss = lossCriterion(outputs, labels)

            # Find Gradients & Update Parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch:", epoch, "batch:", batch, "loss:", loss.item())
            batch += 1
            totalLoss += loss.item()
        lossGain = averageLoss - totalLoss / len(dataLoader)
        averageLoss = totalLoss / len(dataLoader)
        print("================EPOCH:", epoch, "TOTAL LOSS:", totalLoss, "AVERAGE LOSS:", totalLoss / len(dataLoader), "LOSS GAIN:", lossGain, "===============================")

    return mlpModel


def trainMLPModel(trainingSet, fileName, numOfLayers, hiddenLayerSize, numOfEpochs, batchSize, learningRate):
    # Attach required layers
    mlpModel = setLayers(numOfLayers, hiddenLayerSize)

    # Set the device
    device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
    mlpModel.to(device)

    # Create data loader
    dataLoader = createDataLoader(trainingSet, batchSize)
    print("batch size:",batchSize)

    # Loss and Optimizer Parameters
    lossCriterion = nn.CrossEntropyLoss()
    optimizer = tr.optim.SGD(mlpModel.parameters(), lr=learningRate, momentum=0.9)
    print("learning rate:",learningRate)

    # Train
    mlpModel = performTraining(mlpModel, dataLoader, lossCriterion, optimizer, numOfEpochs, device)

    saveMLPModel(mlpModel, fileName)
    return mlpModel

def testMLPModel(mlpModel, testingSet):
    device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
    mlpModel.to(device)
    mlpModel.eval()

    dataLoader = createDataLoader(testingSet, 8)

    groundTruths = []
    predictions = []

    with tr.no_grad():
        for featureVectors, labels in dataLoader:
            featureVectors, labels = featureVectors.to(device), labels.to(device)

            outputs = mlpModel(featureVectors)

            prediction = tr.max(outputs, dim=1)[1]

            groundTruths.extend(labels.cpu().numpy())
            predictions.extend(prediction.cpu().numpy())

    return groundTruths, predictions