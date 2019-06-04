# Decision Trees using **ID3** (Iterative Dichotomiser 3) Algorithm

This repository contains the python implementation of ID3(Iterative Dichotomiser 3) Algorithm. The splitting heuristic used is the information gain which is then used to find the best attribute. 

##### Pseudocode
    ID3 (Examples, Target_Attribute, Attributes)
        Create a root node for the tree
        If all examples are positive, Return the single-node tree Root, with label = +.
        If all examples are negative, Return the single-node tree Root, with label = -.
        If number of predicting attributes is empty, then Return the single node tree Root,
        with label = most common value of the target attribute in the examples.
        Otherwise Begin
            A ← The Attribute that best classifies examples.
            Decision Tree attribute for Root = A.
            For each possible value, _vi_, of A,
                Add a new tree branch below Root, corresponding to the test A = _vi_.
                Let Examples(_vi_) be the subset of examples that have the value _vi_ for A
                If Examples(_vi_) is empty
                    Then below this new branch add a leaf node with label = most common target value in the examples
                Else below this new branch add the subtree ID3 (Examples(_vi_), Target_Attribute, Attributes – {A})
        End
        Return Root

# Dataset

Monks dataset has been provided to run the program. The dataset is divided into testing and training set. 
To use any other dataset please change the `load_file` function depending on the columns you have in your dataset.
The monks dataset can also be downloaded from the [UCI repository](https://archive.ics.uci.edu/ml/index.php).  Other datasets can also be used but make sure that the dataset does not have missing or continuous values. 

# Running the Code
Use python3 to run the code. The dataset should be in the `data` folder parallel to the code. The `load_file` will automatically pick the files and start running the algorithm. 

    python3 decision_tree.py
Currently, the code is set to run for trees of depth 3, 4, 5. Please, change the values in the depth array in the main to run for other depths. 
