import time

from preprocess import *


def main():
    startTime = time.time()
    # Load Datasets : Keep first 500/class
    trainingSet, testingSet = load()


    # Testing Material #################
    print(type(trainingSet), type(trainingSet[0]), type(trainingSet[0][0]))
    for i in range(32):
        print(trainingSet[0][0].getpixel((i,0)))
    #################

    # Prepare DataSets for ResNet-18
    trainingSet, testingSet = transform(trainingSet, testingSet)

    # Testing Material #################
    print(type(trainingSet), type(trainingSet[0]), type(trainingSet[0][0]))
    print(trainingSet[0][0].dtype)
    print(trainingSet[0][0].shape)

    average = 0
    for i in range(7):
        for j in range(7):
            value = ((trainingSet[0][0][0][i][j+14]*0.229)+0.485)*255
            average += value
            print(value)
    print("Average is:",average/49)
    #################

    # ResNet-18 Get Feature Vectors [512,1]
    trainingFeatureVectors, testingFeatureVectors = getFeatureVectors(trainingSet, testingSet)
    print(type(trainingFeatureVectors), type(trainingFeatureVectors[0]))
    for i in trainingFeatureVectors[0]:
        print(i)
    # ResNet-18 Reduce Feature Vectors to [50,1]
    reducedTrainingFeatureVectors, reducedTestingFeatureVectors = reduceFeatureVectors(trainingFeatureVectors, testingFeatureVectors)
    print(type(reducedTrainingFeatureVectors), type(reducedTestingFeatureVectors[0]))

    print("Training Vectors:\n",reducedTrainingFeatureVectors, "\nTesting Vectors:\n",reducedTestingFeatureVectors)
    print(reducedTrainingFeatureVectors.shape, reducedTestingFeatureVectors.shape)
    print(reducedTrainingFeatureVectors.dtype, reducedTestingFeatureVectors.dtype)

    for i in reducedTrainingFeatureVectors[0]:
        print(i)


    # Visualizer:
    # plot_pca_2d(reducedTrainingFeatureVectors, reducedTestingFeatureVectors)


    print("Runtime:",time.time() - startTime, "seconds")




def test():
    return "hi"

if __name__ == "__main__": main()
