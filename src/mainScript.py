import time

import preprocess
import bayes
import tree
import mlp
import cnn
import evaluations

"""
MAIN FILE TO RUN

Loads saved datasets and models.
Tests and evaluates each model on both the testing and training sets.
"""
def main():
    startTime = time.time() # Runtime Metric

    # LOAD SAVED PREPROCESSED INPUT DATA:
    print("Loading Datasets...")
    trainingSet, testingSet = preprocess.loadDataSets('featureVectors')
    trainingSetImages, testingSetImages = preprocess.loadDataSets('images')
    print("Loaded datasets!")

    # Step 2: NAIVE BAYES =================================================
    # LOAD SAVED MODELS:
    bayesModel = bayes.loadBayesModel("bayesModel")
    scikitBayesModel = bayes.loadBayesModel('scikitBayesModel')

    # TEST & EVALUATE:
    print("\nTesting and Evaluating the NAIVE BAYES Models:============================================================")
    # 1 - Main Model:
    print("\nNAIVE BAYES MAIN MODEL ON UNSEEN/TESTING DATA:")
    groundTruths, predictions = bayes.testBayesModel(bayesModel, testingSet)
    evaluations.evaluate(groundTruths, predictions)
    print("\nNAIVE BAYES MAIN MODEL ON SEEN/TRAINING DATA:")
    groundTruths, predictions = bayes.testBayesModel(bayesModel, trainingSet)
    evaluations.evaluate(groundTruths, predictions)

    # 2 - Scikit Model:
    print("\nNAIVE BAYES SCIKIT MODEL ON UNSEEN/TESTING DATA:")
    groundTruths, predictions = bayes.testScikitBayesModel(scikitBayesModel, testingSet)
    evaluations.evaluate(groundTruths, predictions)
    print("\nNAIVE BAYES SCIKIT MODEL ON SEEN/TRAINING DATA:")
    groundTruths, predictions = bayes.testScikitBayesModel(scikitBayesModel, trainingSet)
    evaluations.evaluate(groundTruths, predictions)


    # Step 3: DECISION TREE ==============================================
    # LOAD SAVED MODELS:
    treeModel = tree.loadTreeModel('treeModel50MaxDepth')
    scikitTreeModel = tree.loadTreeModel('scikitTreeModel50MaxDepth')
    reducedTreeModel = tree.loadTreeModel('treeModel40MaxDepth')
    increasedTreeModel = tree.loadTreeModel('treeModel60MaxDepth')

    # TEST & EVALUATE:
    print("\nTesting and Evaluating the DECISION TREE Models:============================================================")
    # 1 - Main Model
    print("\nDECISION TREE MAIN MODEL ON UNSEEN/TESTING DATA:")
    groundTruths, predictions = tree.testTreeModel(treeModel, testingSet)
    evaluations.evaluate(groundTruths, predictions)
    print("\nDECISION TREE MAIN MODEL ON SEEN/TRAINING DATA:")
    groundTruths, predictions = tree.testTreeModel(treeModel, trainingSet)
    evaluations.evaluate(groundTruths, predictions)

    # 2 - Scikit Model
    print("\nDECISION TREE SCIKIT MODEL ON UNSEEN/TESTING DATA:")
    groundTruths, predictions = tree.testScikitTreeModel(scikitTreeModel, testingSet)
    evaluations.evaluate(groundTruths, predictions)
    print("\nDECISION TREE SCIKIT MODEL ON SEEN/TRAINING DATA:")
    groundTruths, predictions = tree.testScikitTreeModel(scikitTreeModel, trainingSet)
    evaluations.evaluate(groundTruths, predictions)

    # 3 - Reduced Depth Model:
    print("\nDECISION TREE REDUCED DEPTH MODEL ON UNSEEN/TESTING DATA:")
    groundTruths, predictions = tree.testTreeModel(reducedTreeModel, testingSet)
    evaluations.evaluate(groundTruths, predictions)
    print("\nDECISION TREE REDUCED DEPTH MODEL ON SEEN/TRAINING DATA:")
    groundTruths, predictions = tree.testTreeModel(reducedTreeModel, trainingSet)
    evaluations.evaluate(groundTruths, predictions)

    # 4 - Increased Depth Model:
    print("\nDECISION TREE INCREASED DEPTH MODEL ON UNSEEN/TESTING DATA:")
    groundTruths, predictions = tree.testTreeModel(increasedTreeModel, testingSet)
    evaluations.evaluate(groundTruths, predictions)
    print("\nDECISION TREE INCREASED DEPTH MODEL ON SEEN/TRAINING DATA:")
    groundTruths, predictions = tree.testTreeModel(increasedTreeModel, trainingSet)
    evaluations.evaluate(groundTruths, predictions)

    # Step 4: MULTI-LAYER PERCEPTRON ========================================
    # LOAD SAVED MODELS:
    mlpModel = mlp.loadMLPModel('mlpModel3Layers512LayerSize')
    reducedDepthMLPModel = mlp.loadMLPModel('mlpModel2Layers512LayerSize')
    increasedDepthMLPModel = mlp.loadMLPModel('mlpModel4Layers512LayerSize')
    reducedLayerSizeMLPModel = mlp.loadMLPModel('mlpModel3Layers256LayerSize')
    increasedLayerSizeMLPModel = mlp.loadMLPModel('mlpModel3Layers1024LayerSize')

    # TEST & EVALUATE:
    print("\nTesting and Evaluating the MULTI-LAYER PERCEPTRON Models:============================================================")
    # 1 - Main Model:
    print("\nMULTI-LAYER PERCEPTRON MAIN MODEL ON UNSEEN/TESTING DATA:")
    groundTruths, predictions = mlp.testMLPModel(mlpModel, testingSet)
    evaluations.evaluate(groundTruths, predictions)
    print("\nMULTI-LAYER PERCEPTRON MAIN MODEL ON SEEN/TRAINING DATA:")
    groundTruths, predictions = mlp.testMLPModel(mlpModel, trainingSet)
    evaluations.evaluate(groundTruths, predictions)

    # 2 - Reduced Depth Model:
    print("\nMULTI-LAYER PERCEPTRON REDUCED DEPTH MODEL ON UNSEEN/TESTING DATA:")
    groundTruths, predictions = mlp.testMLPModel(reducedDepthMLPModel, testingSet)
    evaluations.evaluate(groundTruths, predictions)
    print("\nMULTI-LAYER PERCEPTRON REDUCED DEPTH MODEL ON SEEN/TRAINING DATA:")
    groundTruths, predictions = mlp.testMLPModel(reducedDepthMLPModel, trainingSet)
    evaluations.evaluate(groundTruths, predictions)

    # 3 - Increased Depth Model:
    print("\nMULTI-LAYER PERCEPTRON INCREASED DEPTH MODEL ON UNSEEN/TESTING DATA:")
    groundTruths, predictions = mlp.testMLPModel(increasedDepthMLPModel, testingSet)
    evaluations.evaluate(groundTruths, predictions)
    print("\nMULTI-LAYER PERCEPTRON INCREASED DEPTH MODEL ON SEEN/TRAINING DATA:")
    groundTruths, predictions = mlp.testMLPModel(increasedDepthMLPModel, trainingSet)
    evaluations.evaluate(groundTruths, predictions)

    # 4 - Reduced Layer Size Model:
    print("\nMULTI-LAYER PERCEPTRON REDUCED LAYER SIZE MODEL ON UNSEEN/TESTING DATA:")
    groundTruths, predictions = mlp.testMLPModel(reducedLayerSizeMLPModel, testingSet)
    evaluations.evaluate(groundTruths, predictions)
    print("\nMULTI-LAYER PERCEPTRON REDUCED LAYER SIZE MODEL ON SEEN/TRAINING DATA:")
    groundTruths, predictions = mlp.testMLPModel(reducedLayerSizeMLPModel, trainingSet)
    evaluations.evaluate(groundTruths, predictions)

    # 5 - Increased Layer Size Model:
    print("\nMULTI-LAYER PERCEPTRON INCREASED LAYER SIZE MODEL ON UNSEEN/TESTING DATA:")
    groundTruths, predictions = mlp.testMLPModel(increasedLayerSizeMLPModel, testingSet)
    evaluations.evaluate(groundTruths, predictions)
    print("\nMULTI-LAYER PERCEPTRON INCREASED LAYER SIZE MODEL ON SEEN/TRAINING DATA:")
    groundTruths, predictions = mlp.testMLPModel(increasedLayerSizeMLPModel, trainingSet)
    evaluations.evaluate(groundTruths, predictions)

    # Step 5: CNN
    # LOAD SAVED MODELS:
    cnnModel = cnn.loadCNNModel('cnnModel11Layers3Kernel')
    reducedDepthCNNModel = cnn.loadCNNModel('cnnModel10Layers3Kernel')
    increasedDepthCNNModel = cnn.loadCNNModel('cnnModel12Layers3Kernel')
    reducedKernelSizeCNNModel = cnn.loadCNNModel('cnnModel11Layers2Kernel')
    increasedKernelSizeCNNModel = cnn.loadCNNModel('cnnModel11Layers4Kernel')

    # TEST & EVALUATE:
    print("\nTesting and Evaluating the CONVOLUTIONAL NEURAL NETWORK Models:============================================================")
    # 1 - Main Model
    print("\nCONVOLUTIONAL NEURAL NETWORK MAIN MODEL ON UNSEEN/TESTING DATA:")
    groundTruths, predictions = cnn.testCNNModel(cnnModel, testingSetImages, 8)
    evaluations.evaluate(groundTruths, predictions)
    print("\nCONVOLUTIONAL NEURAL NETWORK MAIN MODEL ON SEEN/TRAINING DATA:")
    groundTruths, predictions = cnn.testCNNModel(cnnModel, trainingSetImages, 8)
    evaluations.evaluate(groundTruths, predictions)

    # 2 - Reduced Depth Model:
    print("\nCONVOLUTIONAL NEURAL NETWORK REDUCED DEPTH MODEL ON UNSEEN/TESTING DATA:")
    groundTruths, predictions = cnn.testCNNModel(reducedDepthCNNModel, testingSetImages, 8)
    evaluations.evaluate(groundTruths, predictions)
    print("\nCONVOLUTIONAL NEURAL NETWORK REDUCED DEPTH MODEL ON SEEN/TRAINING DATA:")
    groundTruths, predictions = cnn.testCNNModel(reducedDepthCNNModel, trainingSetImages, 8)
    evaluations.evaluate(groundTruths, predictions)

    # 3 - Increased Depth Model:
    print("\nCONVOLUTIONAL NEURAL NETWORK INCREASED DEPTH MODEL ON UNSEEN/TESTING DATA:")
    groundTruths, predictions = cnn.testCNNModel(increasedDepthCNNModel, testingSetImages, 8)
    evaluations.evaluate(groundTruths, predictions)
    print("\nCONVOLUTIONAL NEURAL NETWORK INCREASED DEPTH MODEL ON SEEN/TRAINING DATA:")
    groundTruths, predictions = cnn.testCNNModel(increasedDepthCNNModel, trainingSetImages, 8)
    evaluations.evaluate(groundTruths, predictions)

    # 4 - Reduced Kernel Size Model:
    print("\nCONVOLUTIONAL NEURAL NETWORK REDUCED KERNEL SIZE MODEL ON UNSEEN/TESTING DATA:")
    groundTruths, predictions = cnn.testCNNModel(reducedKernelSizeCNNModel, testingSetImages, 8)
    evaluations.evaluate(groundTruths, predictions)
    print("\nCONVOLUTIONAL NEURAL NETWORK REDUCED KERNEL SIZE MODEL ON SEEN/TRAINING DATA:")
    groundTruths, predictions = cnn.testCNNModel(reducedKernelSizeCNNModel, trainingSetImages, 8)
    evaluations.evaluate(groundTruths, predictions)

    # 5 - Increased Kernel Size Model:
    print("\nCONVOLUTIONAL NEURAL NETWORK INCREASED KERNEL SIZE MODEL ON UNSEEN/TESTING DATA:")
    groundTruths, predictions = cnn.testCNNModel(increasedKernelSizeCNNModel, testingSetImages, 8)
    evaluations.evaluate(groundTruths, predictions)
    print("\nCONVOLUTIONAL NEURAL NETWORK INCREASED KERNEL SIZE MODEL ON SEEN/TRAINING DATA:")
    groundTruths, predictions = cnn.testCNNModel(increasedKernelSizeCNNModel, trainingSetImages, 8)
    evaluations.evaluate(groundTruths, predictions)




    print("Runtime:",time.time() - startTime, "seconds") # Runtime Metric

if __name__ == "__main__": main()
