import torch
import torch as tr
import torchvision as tv
from torch.utils.data import DataLoader
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

    resnet18 = tv.models.resnet18()
    resnet18.fc = tr.nn.Identity()

    trainingLoader = DataLoader(trainingSubSet, batch_size=8, shuffle=False)
    testingLoader = DataLoader(testingSubSet, batch_size=8, shuffle=False)

    if tr.cuda.is_available():
        resnet18 = resnet18.cuda()
        device = tr.device('cuda')
    else:
        resnet18 = resnet18.cpu()
        device = tr.device('cpu')

    resnet18 = resnet18.to(device)

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

    trainingFeatureVectors = tr.cat(trainingFeatureVectors)
    testingFeatureVectors = tr.cat(testingFeatureVectors)

    print(len(trainingFeatureVectors), len(testingFeatureVectors))
    return trainingFeatureVectors, testingFeatureVectors

def reduceFeatureVectors(trainingFeatureVectors, testingFeatureVectors):
    # Move the vectors to cpu and convert them to numpy
    trainingFeatureVectors = trainingFeatureVectors.cpu().numpy()
    testingFeatureVectors = testingFeatureVectors.cpu().numpy()

    # Set the number of components
    pca = PCA(n_components=50)

    reducedTrainingFeatureVectors = pca.fit_transform(trainingFeatureVectors)
    reducedTestingFeatureVectors = pca.fit_transform(testingFeatureVectors)

    reducedTrainingFeatureVectors = tr.tensor(reducedTrainingFeatureVectors)
    reducedTestingFeatureVectors = tr.tensor(reducedTestingFeatureVectors)
    return reducedTrainingFeatureVectors, reducedTestingFeatureVectors





