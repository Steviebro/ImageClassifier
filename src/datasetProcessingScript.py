import time
import preprocess
"""
Performs all preprocessing and saves/overwrites saved datasets:
Loads, transforms and extracts feature vectors.
"""
def main():
    startTime = time.time() # Runtime Metric

    # Perform All Preprocessing: Load Images, Transform to Tensors, Get/Reduce Feature Vectors
    trainingSet, testingSet = preprocess.performPreprocessing()
    trainingSetImages, testingSetImages = preprocess.performPreprocessingImages()

    print("Runtime:",time.time() - startTime, "seconds") # Runtime Metric

if __name__ == "__main__": main()
