# Handwritten Digit Recognition using Keras and Tensorflow

Data Set: We will use the USPS digits data set. This data set has already been pre-processed and partitioned into train (5266 digits), validation (1094 digits) and test (930 digits). <br/>
The train, validation and test data are each tensors of size (n × 16 × 16 × 1), where n is the respective number of digits (examples). The greyscale images are represented as a 16 × 16 × 1 tensor in order to maintain consistency for matrix multiplication. If the images were RGB, then the images would be represented as a 16 × 16 × 3 tensor.

#### Architecture of the Model
The deep model constructed using Keras to be a Sequential() model.
     
     build model(n kernels=8, kernel size=3, stride=2, n dense=32)
    
The four arguments to build model are parameters that define the behavior of various layers in the network. The architecture we will use is shown in Figure 1. For the 2D convolutional layer, the parameter n kernels specifies the number of kernels (filters), and each kernel is a square of size kernel size. The parameter stride defines the stride of the max-pooling layer as a stride × stride pooling window. The parameter n dense defines the number of hidden nodes in a densely connected layer.
