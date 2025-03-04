# Adaptive Mean shift with distance distribution based local cluster size estimation

Implementation of an adaptive mean shift algorithm in python, intended to be used on datasets with varying local scale.

For each point, a cluster cardinality is estimated by locating the minimum density of the distance distribution from this point to all others. During the mean shift execution, the cluster size estimation is used to adaptively change the bandwidth. 

### Typical usage
    #initiates the class and estimates local cluster size
    ms=fctMS.AdaptiveMeanShift(X,minClusterSize=10,maxClusterSize=0.7) 
    ms.meanShift() #performs mean shift
    labels=ms.labels

There are 2 parameters: minClusterSize and maxClusterSize. Both can be set approximately with a large margin without affecting the results as long as all clusters have a cardinality between the minimum cluster size and the maximum cluster size.

You can test it with example.ipynb .

Below I'll discuss what are distance distributions and how they are used to estimate locally the cluster size.

## Distance distribution
A distance distribution is the distribution of distances from one point to other points. They can be derived mathematically for a particular distribution. For example, the isotropic Gaussian distribution in $d$ dimensions has a chi distribution with $d$ degrees of freedom. The figure below shows graphically how the chi cdf can be derived by integrating on the gaussian pdf for a given radius.

![Chi distribution](/figure/chi_2D_visualization.PNG)

This is also a good example of how distance distributions typically behave: there's a very low chance of drawing a distance near 0, a mode some distance away and a very low chance of drawing very large distances.

Our example above was a distance distribution between the center of the distribution and points it generated. But the same is true for distance distributions between points: 

*Let* $\mathbf{Y}_d$ *and* $\mathbf{Q}_d$ *be two d-dimensional distributions. Then, the squared Euclidean distance distribution* $X^2$ *between the 2 random points* $\mathbf{Q}$ and $\mathbf{Y}$ *is:*

$$
    X^2=\parallel\mathbf{Y}-\mathbf{Q}\parallel^2=\sum_i(Y_i-Q_i)^2
$$

Assuming $\mathbf{Y}_d$ and $\mathbf{Q}_d$ follow the same distribution, then $\parallel\mathbf{Y}-\mathbf{Q}\parallel^2$ has twice the variance of $\parallel\mathbf{Y}\parallel^2$, among other properties. 

## Using the minimum of the distance distribution density to estimate bandwidth for a distribution

We assume here that distance distributions will have two modes, one for the local cluster we are interested in and another that represents distances to other distributions. The aim of our algorithm is to find the minimum density between those 2 modes, which will be our estimate for the local cluster cardinality. Below is an example:

![Chi distribution](/figure/Cluster_size_estimation.png)
Distance from each points to the star are shown on the left part of the figure. In this case the algorithm finds a bandwidth of around 2, where the min marker is on the left side.

The $\gamma$ function acts mostly as a kernel density estimator in this situation and detects the minimum density between both modes. Below is the definition of the $\gamma$ function

$$
\gamma (\mathbf X_{k})=\frac{Var(\mathbf X_{k})}{(\mathbf E[\mathbf X_{k}]-X_{(k)})^2},
$$

where $\mathbf X_{k}=[X_{(1)},\dots,X_{(k)}]$.
The $\gamma$ function has high values before the first mode which avoid finding a minimum before the one we want to find. To avoid finding a minimum after the second mode we use a parameter called maxClusterSize which limits minimum finding the $maxClusterSize$  closest neighbors. There is also another parameter called minClusterSize with default value 10 which is used to avoid the variance in density with the closest neighbors. They are shown on the previous figure as "Cluster size boundaries". Those parameters can be loosely set and their exact value won't affect the result.

Once the local cluster cardinality is estimated, any statistics can be calculated on the local distribution of distances. Those statistics are than used to adapt the mode seeking process of the mean shift locally.
