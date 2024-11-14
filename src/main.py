import time

from preprocess import *
from bayes import *




def main():
    startTime = time.time() # Runtime Metric

    tr.set_printoptions(sci_mode=False)
    # Step 1: Perform All Preprocessing: Load Images, Transform to Tensors, Get/Reduce Feature Vectors
    trainingSet, testingSet = performPreprocessing()
    #List[(Tensor,label)]

    # Step 2: Train and evaluate a model using Naive Bayes
    means, variances = trainGaussianModel(trainingSet)
    saveModel(means, variances)
    means, variances = loadModel()
    test(means, variances, testingSet)
    scikitClassifier(trainingSet, testingSet)

    print("Runtime:",time.time() - startTime, "seconds") # Runtime Metric

if __name__ == "__main__": main()
