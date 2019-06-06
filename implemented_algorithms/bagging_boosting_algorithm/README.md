# Bagging and Boosting algorithm implementation using Decision Trees

This repository contains the python implementation of bagging and boosting Algorithm.  Bagging and boosting come under ensemble methods. An ensemble method is a technique that combines the predictions from multiple machine learning algorithms together to make more accurate predictions than any individual model.

# Bagging 
Boostrap aggregation or Bagging helps in reducing variance by averaging different models on the sampled dataset. The bias remains unchanged in bagging. Bagging can be easily used with decision trees as it have high variance.

##### Algorithm
1. Let's say the dataset has N samples. So, firstly we will randomly take N samples from the dataset with replacement. 
2. Next, we will learn a model on the randomly sampled data.
3. We will repeat the above method of random sampling and creating a model M times.
4. For making a prediction we can take majority voting. Let's assume we have 10 models and 7 models return a true value on a given example then we will return true by majority voting.
