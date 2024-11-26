import pickle
import torch as tr
import torch.nn as nn
import numpy as np
from numpy.ma.extras import average
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix

def saveCNNModel(cnnModel, fileName):
    with open('../models/cnn/'+fileName+'.pkl', 'wb') as file:
        pickle.dump(cnnModel, file)

def loadCNNModel(fileName):
    with open('../models/cnn/'+fileName+'.pkl', 'rb') as file:
        return pickle.load(file)

# Sets the layers:
# If 10 layers are desired, layer 7 is skipped
# If 12 layers are desired, layer 7 is duplicated
def setLayers(numOfLayers, kernelSize):
    if numOfLayers != 10 and numOfLayers != 11 and numOfLayers != 12:
        raise Exception('Invalid number of layers used')
    if kernelSize != 2 and kernelSize != 3 and kernelSize != 4:
        raise Exception('Invalid kernel size')

    if kernelSize == 2:
        offset = 8*8
    if kernelSize == 3:
        offset = 7*7
    if kernelSize == 4:
        offset = 5*5

    layers = []
    layers.append(nn.Sequential( # Layer 1
        nn.Conv2d(3,64,kernelSize,1,1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    ))
    layers.append(nn.Sequential( # Layer 2
        nn.Conv2d(64,128,kernelSize,1,1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    ))
    layers.append(nn.Sequential( # Layer 3
        nn.Conv2d(128,256,kernelSize,1,1),
        nn.BatchNorm2d(256),
        nn.ReLU()
    ))
    layers.append(nn.Sequential( # Layer 4
        nn.Conv2d(256,256,kernelSize,1,1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    ))
    layers.append(nn.Sequential( # Layer 5
        nn.Conv2d(256,512,kernelSize,1,1),
        nn.BatchNorm2d(512),
        nn.ReLU()
    ))
    layers.append(nn.Sequential( # Layer 6
        nn.Conv2d(512,512,kernelSize,1,1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    ))
    if numOfLayers != 10: # Skip this layer for the reduced layer model
        layers.append(nn.Sequential( # Layer 7
            nn.Conv2d(512,512,kernelSize,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ))
    if numOfLayers == 13: # Duplicate the layer for the increased layer model
        layers.append(nn.Sequential(  # Layer 7
            nn.Conv2d(512, 512, kernelSize, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ))

    layers.append(nn.Sequential( # Layer 8
        nn.Conv2d(512,512,kernelSize,1,1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    ))
    # layers.append(nn.Flatten())

    layers.append(nn.Sequential( # Layer 9
        nn.Flatten(),
        nn.Linear(512*offset,4096),
        nn.ReLU(),
        nn.Dropout(0.5)
    ))
    layers.append(nn.Sequential( # Layer 10
        nn.Linear(4096,4096),
        nn.ReLU(),
        nn.Dropout(0.5)
    ))
    layers.append(nn.Sequential( # Layer 11
        nn.Linear(4096,10)
    ))
    return nn.Sequential(*layers)

def createDataLoader(dataSet, batchSize):
    images = tr.stack([i[0] for i in dataSet], dim=0)
    classLabels = tr.tensor([i[1] for i in dataSet], dtype=tr.long)
    return tr.utils.data.DataLoader(TensorDataset(images, classLabels), batch_size=batchSize, shuffle=True)

def performTraining(cnnModel, dataLoader, lossCriterion, optimizer, numOfEpochs, device):
    averageLoss = 0.0
    for epoch in range(numOfEpochs):
        cnnModel.train()
        totalLoss = 0.0
        batch = 0
        for images, labels in dataLoader:
            images, labels = images.to(device), labels.to(device)

            # Forward: Find Outputs
            outputs = cnnModel(images)
            loss = lossCriterion(outputs, labels)

            # Backward: Find Gradients and Update Parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch:", epoch, "batch:", batch, "loss:", loss.item())
            batch += 1
            totalLoss += loss.item()
        lossGain = averageLoss - totalLoss / len(dataLoader)
        averageLoss = totalLoss/len(dataLoader)
        print("================EPOCH:", epoch, "TOTAL LOSS:", totalLoss,"AVERAGE LOSS:", totalLoss/len(dataLoader),"LOSS GAIN:", lossGain, "===============================")

    return cnnModel

def trainCNNModel(trainingSet, fileName, numOfLayers, numOfEpochs, batchSize, kernelSize, learningRate):
    # Attach required layers
    cnnModel = setLayers(numOfLayers, kernelSize)
    print("kernel size:", kernelSize)

    # Set Device
    device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
    cnnModel.to(device)
    print("device:", device)

    # Create data loader
    dataLoader = createDataLoader(trainingSet, batchSize)
    print("batch size:",dataLoader.batch_size)

    # Loss and Optimizer Parameters
    lossCriterion = nn.CrossEntropyLoss()
    optimizer = tr.optim.SGD(cnnModel.parameters(), lr=learningRate, momentum=0.9)
    print("learning rate:", optimizer.param_groups[0]['lr'])

    # Train
    cnnModel = performTraining(cnnModel, dataLoader, lossCriterion, optimizer, numOfEpochs, device)

    saveCNNModel(cnnModel, fileName)
    return cnnModel

def testCNNModel(cnnModel, testingSet, batchSize):
    device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
    cnnModel.to(device)
    cnnModel.eval()

    dataLoader = createDataLoader(testingSet, batchSize)

    groundTruths = []
    predictions = []

    with tr.no_grad():
        for images, labels in dataLoader:
            images, labels = images.to(device), labels.to(device)

            outputs = cnnModel(images)

            prediction = tr.max(outputs, dim=1)[1]

            groundTruths.extend(labels.cpu().numpy())
            predictions.extend(prediction.cpu().numpy())

    accuracy = accuracy_score(groundTruths, predictions)
    confusionMatrix = confusion_matrix(groundTruths, predictions)
    print(confusionMatrix)
    print(f"CNN Implementation Accuracy: {accuracy * 100:.2f}%")
    return groundTruths, predictions