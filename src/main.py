import time

from preprocess import *


def main():
    startTime = time.time() # Runtime Metric

    # Step 1: Perform All Preprocessing: Load Images, Transform to Tensors, Get/Reduce Feature Vectors
    performPreprocessing()

    # Step 2: Train and evaluate a model using Naive Bayes
    



    print("Runtime:",time.time() - startTime, "seconds") # Runtime Metric

if __name__ == "__main__": main()
