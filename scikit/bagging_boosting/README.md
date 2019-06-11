# Bagging and Boosting using Scikit-learn

This repository contains the python code for running bagging and boosting Algorithm using scikit-learn.
Decision trees are used as the classifier for creating bagging and boosting models.
Bagging and boosting come under ensemble methods. An ensemble method is a technique that combines the predictions from multiple machine learning algorithms together to make more accurate predictions than any individual model.

# Bagging 
Boostrap aggregation or Bagging helps in reducing variance by averaging different models on the sampled dataset. The bias remains unchanged in bagging. Bagging can be easily used with decision trees as it have high variance.

##### Algorithm
    1. Let's say the dataset has N samples. So, firstly we will randomly take N samples from the dataset with replacement. 
    2. Next, we will learn a model on the randomly sampled data.
    3. We will repeat the above method of random sampling and creating a model M times.
    4. For making a prediction we can take majority voting. Let's assume we have 10 models and 7 models return a true 
    value on a given example then we will return true by majority voting.

##### Python code for Bagging using Scikit
    num_trees = 10
    seed = 7
    depth = 3
    clf = tree.DecisionTreeClassifier(max_depth=depth, min_samples_split=2)
    bagmodel = BaggingClassifier(base_estimator=clf, n_estimators=num_trees, random_state=seed)


# Boosting (AdaBoost)
In boosting we use weak learners (accuracy > 0.5, high bias) to reduce both bias and variance. It is an iterative algorithm that increases weights on misclassified examples so more focus is on the misclassified examples. 

##### Algorithm (Pseudocode)
    Set uniform example weights
    for Each base learner do
      Train base learner with weighted sample.
      Test base learner on all data.
      Set learner weight with weighted error.
      Set example weights based on ensemble predictions.
    end for

##### Python code for Boosting using Scikit
    num_trees = 10
    depth = 3
    clf_stump = tree.DecisionTreeClassifier(max_depth=depth, min_samples_split=2)
    adaboost = AdaBoostClassifier(base_estimator=clf_stump, n_estimators=num_trees)
    
# Dataset
Mushroom dataset is provided in the data folder. Use the UCI repository to get other datasets and make the required changes in the `load_file` function.
