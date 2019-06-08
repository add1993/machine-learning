# Decision Trees using Scikit-learn

This repository contains the python code for decision trees using scikit learn. 

    DecisionTreeClassifier // Part of python scikit learn library

# Dataset

Monks dataset has been provided to run the program. The dataset is divided into testing and training set. 
To use any other dataset please change the `load_file` function depending on the columns you have in your dataset.
The monks dataset can also be downloaded from the [UCI repository](https://archive.ics.uci.edu/ml/index.php).  Other datasets can also be used but make sure that the dataset does not have missing or continuous values. 

# Running the Code
Use python3 to run the code. The dataset should be in the `data` folder parallel to the code. The `load_file` will automatically pick the files and start running the algorithm. 

    python3 decision_tree.py
Currently, the code is set to run for trees of depth 3, 4, 5. Please, change the values in the depth array in the main to run for other depths. 
