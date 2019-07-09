# Handwritten Digit Recognition using Keras and Tensorflow

Data Set: We will use the USPS digits data set. This data set has already been pre-processed and partitioned into train (5266 digits), validation (1094 digits) and test (930 digits). <br/>
The train, validation and test data are each tensors of size (n × 16 × 16 × 1), where n is the respective number of digits (examples). The greyscale images are represented as a 16 × 16 × 1 tensor in order to maintain consistency for matrix multiplication. If the images were RGB, then the images would be represented as a 16 × 16 × 3 tensor.
