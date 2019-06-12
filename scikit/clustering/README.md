# Spectral Clustering implementation using Scikit-learn
In this repo we implement spectral clustering using eigenvalue decomposition of the given data and applying k-means on it.

# Dataset
We are using the default dataset present in scikit-learn called `circles` dataset. This contains concentric circles and we will
use spectral clustering to properly classify both the circles into two separate clusters.

##### Scikit code for generating dataset
  `x = datasets.make_circles(n_samples =1500, factor=.5, noise=.05)`
