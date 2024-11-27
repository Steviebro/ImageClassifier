import time

import preprocess
import bayes
import tree
import mlp
import cnn

"""
Run only if necessary or for checking purposes since saved models are overwritten.

Loads saved datasets.
Re-trains and overwrites all saved models.
Note: Comment out lines for models to NOT re-train.
"""
def main():
    startTime = time.time() # Runtime Metric

    # LOAD SAVED PREPROCESSED INPUT DATA:
    trainingSet, testingSet = preprocess.loadDataSets('featureVectors')
    trainingSetImages, testingSetImages = preprocess.loadDataSets('images')
    print("loaded datasets")

    # Step 2: NAIVE BAYES =================================================
    # RE-TRAIN:
    bayesModel = bayes.trainBayesModel(trainingSet, "bayesModel")
    scikitBayesModel = bayes.trainScikitBayesModel(trainingSet)


    # Step 3: DECISION TREE ==============================================
    treeModel = tree.trainTreeModel(trainingSet, 50, 'treeModel50MaxDepth')
    scikitTreeModel = tree.trainScikitTreeModel(trainingSet, 50, 'scikitTreeModel50MaxDepth')
    reducedTreeModel = tree.trainTreeModel(trainingSet, 40, 'treeModel50MaxDepth')
    increasedTreeModel = tree.trainTreeModel(trainingSet, 60, 'treeModel60MaxDepth')

    # Step 4: MULTI-LAYER PERCEPTRON ========================================
    mlpModel = mlp.trainMLPModel(trainingSet, 'mlpModel3Layers512LayerSize', 3, 512, 9, 32, 0.001)
    reducedDepthMLPModel = mlp.trainMLPModel(trainingSet, 'mlpModel2Layers512LayerSize', 2, 512, 9, 32, 0.001)
    increasedDepthMLPModel = mlp.trainMLPModel(trainingSet, 'mlpModel4Layers512LayerSize', 4, 512, 9, 32, 0.001)
    reducedLayerSizeMLPModel = mlp.trainMLPModel(trainingSet, 'mlpModel3Layers256LayerSize', 3, 256, 9, 32, 0.001)
    increasedLayerSizeMLPModel = mlp.trainMLPModel(trainingSet, 'mlpModel3Layers1024LayerSize', 3, 1024, 9, 32, 0.001)


    # Step 5: CONVOLUTED NEURAL NETWORK ========================================================
    cnnModel = cnn.trainCNNModel(trainingSetImages, 'cnnModel11Layers3Kernel', 11, 20, 32, 3, 0.001)
    reducedDepthCNNModel = cnn.trainCNNModel(trainingSetImages, 'cnnModel10Layers3Kernel', 10, 20, 32, 3, 0.001)
    increasedDepthCNNModel = cnn.trainCNNModel(trainingSetImages, 'cnnModel12Layers3Kernel', 12, 20, 32, 3, 0.001)
    reducedKernelSizeCNNModel = cnn.trainCNNModel(trainingSetImages, 'cnnModel11Layers2Kernel', 11, 20, 32, 2, 0.001)
    increasedKernelSizeCNNModel = cnn.trainCNNModel(trainingSetImages, 'cnnModel11Layers4Kernel', 11, 20, 32, 4, 0.001)

    print("Runtime:",time.time() - startTime, "seconds") # Runtime Metric

if __name__ == "__main__": main()
