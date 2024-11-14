"""
preprocess.py

This module handles all data preprocessing steps.
The final method performPreprocessing() encompasses all required steps:
    1. Load the data from CIFAR10, keeping the first 500 training and 100 test elements
    2. Transform the images to tensors in preparation for ResNet-18 usage
    3. Use ResNet-18 to get feature vectors for each data element
    4. Use PCA to reduce the vector sizes from 512 --> 50
    5. Recombine the data with the labels in a list (1 element per image) of tuples (features, label)
"""

import torch as tr
import torchvision as tv
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from sklearn.decomposition import PCA


# Returns the training (500 per class) and testing (100 per class) sets
def load():
    # Load both the datasets
    trainingSet = tv.datasets.CIFAR10(root='./data', train=True, download=True)
    testingSet = tv.datasets.CIFAR10(root='./data', train=False, download=True)

    # Filter trainingSet, keeping only 500 of each class
    numKept = [0] * 10
    trainingSubSet = []

    for i in trainingSet:
        image, label = i
        if (numKept[label] < 500): # Keep only if we don't already have 500 of this class
            numKept[label] += 1
            trainingSubSet.append(i)

    # Filter testingSet, keeping only 500 of each class
    numKept = [0] * 10
    testingSubSet = []

    for i in testingSet:
        image, label = i
        if (numKept[label] < 100):
            numKept[label] += 1
            testingSubSet.append(i)

    return trainingSubSet, testingSubSet

# Applies transformations to prepare the training and testing sets for ResNet-18
def transform(trainingSubSet, testingSubSet):
    transformedTrainingSubSet = []
    transformedTestingSubSet = []

    # Define the transformations to apply to the dataset
    transformations = tv.transforms.Compose([
        tv.transforms.Resize(224), # maybe RandomResizedCrop instead?
        tv.transforms.ToTensor(),
        # tv.transforms.RandomHorizontalFlip(),
        tv.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)) # Values for datasets trained on ImageNet
    ])

    # Apply transformations
    for i in trainingSubSet:
        image, label = i
        transformedTrainingSubSet.append((transformations(image),label))

    for i in testingSubSet:
        image, label = i
        transformedTestingSubSet.append((transformations(image),label))

    return transformedTrainingSubSet, transformedTestingSubSet

# Gets ResNet-18 feature vectors for training and testing sets
def getFeatureVectors(trainingSubSet, testingSubSet):
    trainingFeatureVectors = []
    testingFeatureVectors = []

    # Remove outer layer of ResNet-18
    resnet18 = tv.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet18.fc = tr.nn.Identity()
    resnet18.eval()

    # Load the data into batches
    trainingLoader = DataLoader(trainingSubSet, batch_size=64, shuffle=False)
    testingLoader = DataLoader(testingSubSet, batch_size=64, shuffle=False)

    # Enable GPU use if available
    if tr.cuda.is_available():
        resnet18 = resnet18.cuda()
        device = tr.device('cuda')
    else:
        resnet18 = resnet18.cpu()
        device = tr.device('cpu')
    resnet18 = resnet18.to(device)

    # Apply ResNet-18 to the data batches
    with tr.no_grad():
        for i in trainingLoader:
            images, labels = i
            if tr.cuda.is_available():
                images = images.cuda()
            trainingFeatureVectors.append(resnet18(images))

        for i in testingLoader:
            images, labels = i
            if tr.cuda.is_available():
                images = images.cuda()
            testingFeatureVectors.append(resnet18(images))

    # Concatenate the batches into a single list
    trainingFeatureVectors = tr.cat(trainingFeatureVectors)
    testingFeatureVectors = tr.cat(testingFeatureVectors)
    return trainingFeatureVectors, testingFeatureVectors

def reduceFeatureVectors(trainingFeatureVectors, testingFeatureVectors):
    # Move the vectors to cpu and convert them to numpy
    trainingFeatureVectors = trainingFeatureVectors.cpu().numpy()
    testingFeatureVectors = testingFeatureVectors.cpu().numpy()

    # Set the number of components
    pca = PCA(n_components=50)

    # Fit the vector to the required number of components
    reducedTrainingFeatureVectors = pca.fit_transform(trainingFeatureVectors)
    reducedTestingFeatureVectors = pca.fit_transform(testingFeatureVectors)

    # Convert them back to tensor
    reducedTrainingFeatureVectors = tr.tensor(reducedTrainingFeatureVectors)
    reducedTestingFeatureVectors = tr.tensor(reducedTestingFeatureVectors)
    return reducedTrainingFeatureVectors, reducedTestingFeatureVectors

# Performs all Preprocessing: load, transform, feature vectors
# Returns a list containing tuples for each data element (features, label)
def performPreprocessing():
    # Load Datasets : Keep first 500/class
    trainingSet, testingSet = load()

    # Prepare DataSets for ResNet-18
    trainingSet, testingSet = transform(trainingSet, testingSet)

    # ResNet-18 Get Feature Vectors [512,1]
    trainingFeatureVectors, testingFeatureVectors = getFeatureVectors(trainingSet, testingSet)

    # ResNet-18 Reduce Feature Vectors to [50,1]
    reducedTrainingFeatureVectors, reducedTestingFeatureVectors = reduceFeatureVectors(trainingFeatureVectors, testingFeatureVectors)

    # Combine feature vectors with labels
    finalTraining = []
    finalTesting = []
    for i, featureVector in enumerate(reducedTrainingFeatureVectors):
        finalTraining.append((featureVector, trainingSet[i][1]))
    for i, featureVector in enumerate(reducedTestingFeatureVectors):
        finalTesting.append((featureVector, testingSet[i][1]))

    return finalTraining, finalTesting