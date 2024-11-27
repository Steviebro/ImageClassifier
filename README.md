# COMP 472 Project

#### Author: Steven Gingras (40098045)
#### Team: Theta

### Python Version: 3.9

### Dependencies
The dependencies can be found in the `/requirements.txt` file.
Simply run `pip install` to install all required dependencies

## Important First Step: Saved Datasets
Since the saved models were too large for GitHub, I would ask you to please use this link to obtain the saved models (Note: Concordia account required):\
https://liveconcordia-my.sharepoint.com/:f:/g/personal/s_ingra_live_concordia_ca/Emr3pbxlMo9IjOQegXXIO2EBhtZw75b5Ptln2FWspZ00mg?e=NBdKDQ \
Once downloaded, please extract the `/models/` folder and replace the empty `/models/` directory with the downloaded folder.

# Run Instructions
There are only **THREE FILES** that are meant to be run in this project. Most notably the first one, the other two only if necessary.
1. **MAIN SCRIPT**: Run the script `/src/mainScript.py` to load the datasets and saved models then test and evaluate them on the testing and training sets.
2. If dataset (re-)preprocessing is required: run the script `/src/datasetProcessingScript.py` to load, preprocess and save the datasets.
3. If model (re-)training is required: run the script `/src/retrainModelsScript.py` to train all models and their variants and save the datasets. All but desired models can be commented out, but they are all there for convenience.
- **IMPORTANT WARNING**: Running scripts 2 or 3 **WILL OVERWRITE** the saved models and variants which will inevitably lead to some discrepancies in the results obtained in the evaluation.


# File Descriptions

### Source Directory (`/src/`):
#### Scripts:
  1. `mainScript.py`:
     - Loads saved datasets (images & feature vectors) and all saved models
     - Tests and evaluates each model both on the testing set and the training set
  2. `datasetProcessingScript.py`:
     - Performs all preprocessing to load, transform and extract features from the CIFAR10 dataset.
     - Saves (**OVERWRITES**) preprocessed data to the `models/datasets/` directory
  3. `retrainModelsScript.py`:
     - Loads saved datasets and re-trains the models and their variants.
     - Saves (**OVERWRITES**) the models to the `models/` directory within its appropriate subdirectory.
#### Models:
4. `bayes.py` 5. `tree.py` 6. `mlp.py` 7. `cnn.py`

Each of the above files contain code for their respective model category. Code for the following is included in each file:
- Saving/Loading the model and its variants.
- Training the model and its variants.
- Testing the model and its variants.

#### Other:
  8. `preprocess.py`:
    - Contains code for all preprocessing of the datasets.
  9. `evaluations.py`:
    - Contains code for generating evaluation metrics based on ground truths and predictions.


### Models Directory (`/models/`):
Contains all saved models, including their variants. An exhaustive list is as follows:

Naive Bayes:
1. `models/bayes/bayesModel.pkl`: Main Bayes Model
2. `models/bayes/scikitBayesModel.pkl`: Scikit Bayes Model

Decision Tree:
1. `models/tree/treeModel50MaxDepth.pkl`: Main Decision Tree Model
2. `models/tree/scikitTreeModel50MaxDepth.pkl`: Scikit Decision Tree Model
3. `models/tree/treeModel40MaxDepth.pkl`: Reduced Depth Decision Tree Model
4. `models/tree/treeModel60MaxDepth.pkl`: Increased Depth Decision Tree Model

Multi-Layer Perceptron:
1. `models/mlp/mlpModel3Layers512LayerSize.pkl`: Main MLP Model
2. `models/mlp/mlpModel2Layers512LayerSize.pkl`: Reduced Depth MLP Model
3. `models/mlp/mlpModel4Layers512LayerSize.pkl`: Increased Depth MLP Model
4. `models/mlp/mlpModel3Layers256LayerSize.pkl`: Reduced Layer Size MLP Model
5. `models/mlp/mlpModel3Layers1024LayerSize.pkl`: Increased Layer Size MLP Model

Convolutional Neural Network:
1. `models/cnn/cnnModel11Layers3Kernel.pkl`: Main CNN Model
2. `models/cnn/cnnModel10Layers3Kernel.pkl`: Reduced Depth CNN Model
3. `models/cnn/cnnModel12Layers3Kernel.pkl`: Increased Depth CNN Model
4. `models/cnn/cnnModel11Layers2Kernel.pkl`: Reduced Kernel Size CNN Model
5. `models/cnn/cnnModel11Layers4Kernel.pkl`: Increased Kernel Size CNN Model

### Other Files in Root:
1. `.gitignore`: Specifies files to be ignored by git.
2. `README.md`: Lists instructions and information about the project and files.
3. `requirements.txt`: Lists required dependencies for the project.