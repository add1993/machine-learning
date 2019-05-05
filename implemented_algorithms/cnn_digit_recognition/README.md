# MNIST Data Digit Recognition using Pytorch

We will be using  train.csv from MNIST dataset. It contains gray-scale images of hand-drawn digits, from zero through nine. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive. The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.” We only need to download the train.csv file.
test.csv is not required. The train.csv file contains 42k samples of images. To reduce time of running the program, you will only work with 10,000 randomly selected samples out of these, although make sure you have equal number of samples belonging to each label (i.e.,1,000 samples of label ‘0’, 1,000 samples of label ‘1’ and so on).
Using MNIST dataset we will try to create a CNN model in PyTorch for digit recognition. A sample small dataset containing 1000 samples for each digit is added in the repository to use directly.

## Getting Started

Dataset can be downloaded from Kaggle.
Dataset : [MNIST Dataset](https://www.kaggle.com/c/digit-recognizer/data)

## Structure of CNN Model
Convolutional layer -> Max pooling layer -> Convolutional layer - > Max pooling layer -> Fully connected layer x2 -> Softmax layer

### Prerequisites

Python3 with PyTorch is required with project. Other dependencies are numpy, pandas. 

## Authors

**Ayush Dobhal** 
