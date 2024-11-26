import time

import preprocess
import bayes
import tree
import mlp
import cnn
import evaluations


def main():
    startTime = time.time() # Runtime Metric

    # Step 1: Perform All Preprocessing: Load Images, Transform to Tensors, Get/Reduce Feature Vectors
    # trainingSet, testingSet = preprocess.performPreprocessing()
    # trainingSetImages, testingSetImages = preprocess.performPreprocessingImages()

    # LOAD SAVED PREPROCESSED INPUT DATA:
    # trainingSet, testingSet = preprocess.loadDataSets('featureVectors')
    # trainingSetImages, testingSetImages = preprocess.loadDataSets('images')
    print("loaded datasets")

    # Step 2: NAIVE BAYES =================================================
    # RE-TRAIN:
    # bayesModel = bayes.trainBayesModel(trainingSet, "bayesModel")
    # scikitBayesModel = bayes.trainScikitBayesModel(trainingSet)

    # LOAD SAVED MODELS:
    # bayesModel = bayes.loadBayesModel("bayesModel")
    # scikitBayesModel = bayes.loadBayesModel('scikitBayesModel')

    # TEST & EVALUATE:
    # 1 - Main Model:
    # print("testing testing set:")
    # groundTruths, predictions = bayes.testBayesModel(bayesModel, testingSet)
    # evaluations.evaluate(groundTruths, predictions)
    # print("testing training set:")
    # groundTruths, predictions = bayes.testBayesModel(bayesModel, trainingSet)
    # evaluations.evaluate(groundTruths, predictions)

    # 2 - Scikit Model:
    # print("testing testing set:")
    # groundTruths, predictions = bayes.testScikitBayesModel(scikitBayesModel, testingSet)
    # evaluations.evaluate(groundTruths, predictions)
    # print("testing training set:")
    # groundTruths, predictions = bayes.testScikitBayesModel(scikitBayesModel, trainingSet)
    # evaluations.evaluate(groundTruths, predictions)


    # Step 3: DECISION TREE ==============================================
    # RE-TRAIN:
    # treeModel = tree.trainTreeModel(trainingSet, 50, 'treeModel50MaxDepth')
    # scikitTreeModel = tree.trainScikitTreeModel(trainingSet, 50, 'scikitTreeModel50MaxDepth')
    # reducedTreeModel = tree.trainTreeModel(trainingSet, 40, 'treeModel50MaxDepth')
    # increasedTreeModel = tree.trainTreeModel(trainingSet, 60, 'treeModel60MaxDepth')

    # LOAD SAVED MODELS:
    # treeModel = tree.loadTreeModel('treeModel50MaxDepth')
    # scikitTreeModel = tree.loadTreeModel('scikitTreeModel50MaxDepth')
    # reducedTreeModel = tree.loadTreeModel('treeModel40MaxDepth')
    # increasedTreeModel = tree.loadTreeModel('treeModel60MaxDepth')

    # TEST & EVALUATE:
    # 1 - Main Model
    # print("testing testing set:")
    # groundTruths, predictions = tree.testTreeModel(treeModel, testingSet)
    # evaluations.evaluate(groundTruths, predictions)
    # print("testing training set:")
    # groundTruths, predictions = tree.testTreeModel(treeModel, trainingSet)
    # evaluations.evaluate(groundTruths, predictions)

    # 2 - Scikit Model
    # print("testing testing set:")
    # groundTruths, predictions = tree.testScikitTreeModel(scikitTreeModel, testingSet)
    # evaluations.evaluate(groundTruths, predictions)
    # print("testing training set:")
    # groundTruths, predictions = tree.testScikitTreeModel(scikitTreeModel, trainingSet)
    # evaluations.evaluate(groundTruths, predictions)

    # 3 - Reduced Depth Model:
    # print("testing testing set:")
    # groundTruths, predictions = tree.testTreeModel(reducedTreeModel, testingSet)
    # evaluations.evaluate(groundTruths, predictions)
    # print("testing training set:")
    # groundTruths, predictions = tree.testTreeModel(reducedTreeModel, trainingSet)
    # evaluations.evaluate(groundTruths, predictions)

    # 4 - Increased Depth Model:
    # print("testing testing set:")
    # groundTruths, predictions = tree.testTreeModel(increasedTreeModel, testingSet)
    # evaluations.evaluate(groundTruths, predictions)
    # print("testing training set:")
    # groundTruths, predictions = tree.testTreeModel(increasedTreeModel, trainingSet)
    # evaluations.evaluate(groundTruths, predictions)

    # Step 4: MULTI-LAYER PERCEPTRON ========================================
    # RE-TRAIN:
    # mlpModel = mlp.trainMLPModel(trainingSet, 'mlpModel3Layers512LayerSize', 3, 512, 9, 32, 0.001)
    # reducedDepthMLPModel = mlp.trainMLPModel(trainingSet, 'mlpModel2Layers512LayerSize', 2, 512, 9, 32, 0.001)
    # increasedDepthMLPModel = mlp.trainMLPModel(trainingSet, 'mlpModel4Layers512LayerSize', 4, 512, 9, 32, 0.001)
    # reducedLayerSizeMLPModel = mlp.trainMLPModel(trainingSet, 'mlpModel3Layers256LayerSize', 3, 256, 9, 32, 0.001)
    # increasedLayerSizeMLPModel = mlp.trainMLPModel(trainingSet, 'mlpModel3Layers1024LayerSize', 3, 1024, 9, 32, 0.001)

    # LOAD SAVED MODELS:
    # mlpModel = mlp.loadMLPModel('mlpModel3Layers512LayerSize')
    # reducedDepthMLPModel = mlp.loadMLPModel('mlpModel2Layers512LayerSize')
    # increasedDepthMLPModel = mlp.loadMLPModel('mlpModel4Layers512LayerSize')
    # reducedLayerSizeMLPModel = mlp.loadMLPModel('mlpModel3Layers256LayerSize')
    # increasedLayerSizeMLPModel = mlp.loadMLPModel('mlpModel3Layers1024LayerSize')

    # TEST & EVALUATE:
    # 1 - Main Model:
    # print("testing testing set:")
    # groundTruths, predictions = mlp.testMLPModel(mlpModel, testingSet)
    # evaluations.evaluate(groundTruths, predictions)
    # print("testing training set:")
    # groundTruths, predictions = mlp.testMLPModel(mlpModel, trainingSet)
    # evaluations.evaluate(groundTruths, predictions)

    # 2 - Reduced Depth Model:
    # print("testing testing set:")
    # groundTruths, predictions = mlp.testMLPModel(reducedDepthMLPModel, testingSet)
    # evaluations.evaluate(groundTruths, predictions)
    # print("testing training set:")
    # groundTruths, predictions = mlp.testMLPModel(reducedDepthMLPModel, trainingSet)
    # evaluations.evaluate(groundTruths, predictions)

    # 3 - Increased Depth Model:
    # print("testing testing set:")
    # groundTruths, predictions = mlp.testMLPModel(increasedDepthMLPModel, testingSet)
    # evaluations.evaluate(groundTruths, predictions)
    # print("testing training set:")
    # groundTruths, predictions = mlp.testMLPModel(increasedDepthMLPModel, trainingSet)
    # evaluations.evaluate(groundTruths, predictions)

    # 4 - Reduced Layer Size Model:
    # print("testing testing set:")
    # groundTruths, predictions = mlp.testMLPModel(reducedLayerSizeMLPModel, testingSet)
    # evaluations.evaluate(groundTruths, predictions)
    # print("testing training set:")
    # groundTruths, predictions = mlp.testMLPModel(reducedLayerSizeMLPModel, trainingSet)
    # evaluations.evaluate(groundTruths, predictions)

    # 5 - Increased Layer Size Model:
    # print("testing testing set:")
    # groundTruths, predictions = mlp.testMLPModel(increasedLayerSizeMLPModel, testingSet)
    # evaluations.evaluate(groundTruths, predictions)
    # print("testing training set:")
    # groundTruths, predictions = mlp.testMLPModel(increasedLayerSizeMLPModel, trainingSet)
    # evaluations.evaluate(groundTruths, predictions)

    # Step 5: CNN
    # RE-TRAIN:
    # cnnModel = cnn.trainCNNModel(trainingSetImages, 'cnnModel11Layers4Kernel', 11, 20, 32, 4, 0.001)

    # LOAD SAVED MODELS:
    # cnnModel = cnn.loadCNNModel('cnnModel11Layers4Kernel')

    # TEST & EVALUATE:
    # 1 - Main Model
    # print("testing testing set:")
    # groundTruths, predictions = cnn.testCNNModel(cnnModel, testingSetImages, 8)
    # evaluations.evaluate(groundTruths, predictions)
    # print("testing training set:")
    # cnn.testCNNModel(cnnModel, trainingSetImages, 8)
    # evaluations.evaluate(groundTruths, predictions)


    print("Runtime:",time.time() - startTime, "seconds") # Runtime Metric

if __name__ == "__main__": main()
