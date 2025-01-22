# Adaptive Mean shift with distance distribution based local cluster size estimation

Implementation of an adaptive mean shift algorithm in python, intended to be used on datasets with varying local scale.

For each point, cluster size is estimated by locating the minimum density of the distance distribution from this point to all others. During the mean shift execution, the cluster size estimation is used to adaptively change the bandwidth. 

### Typical usage
    ms=fctMS.AdaptiveMeanShift(X,minClusterSize=10,maxClusterSize=0.75,removeBadEstimates=True)
    ms.meanShift(printClustersDebug=False)
    ms.clusterUnlabeledPoints()
    labels=ms.labels

There are 2 parameters: minClusterSize and maxClusterSize. Both can be set approximatively without affecting the results as long as all clusters have a size between the minimum cluster size and the maximum cluster size.

You can test it with example.ipynb .

## Distance distribution
A distance distribution is the distribution of distances from one point to other points. They can be derived mathematically for a particular distribution. For example, the isotropic normal distribution in $d$ dimensions has a chi distribution with $d$ degrees of freedom. The figure below shows graphically how the chi cdf can be derived by integrating on the gaussian pdf for a given radius.

![Chi distribution](/figure/chi_2D_visualization.PNG)

This is also a good example of how distance distributions typically behave: there's a very low chance of drawing a distance near 0, a mode some distance away and a very low chance of drawing very large distances.

Our example above was a distance distribution between the center of the distribution and points it generated. But the same is true for distance distributions between points: 

*Let* $\mathbf{Y}_d$ *and* $\mathbf{Q}_d$ *be two d-dimensional distributions. Then, the squared Euclidean distance distribution* $X^2$ *between the 2 random points* $\mathbf{Q}$ and $\mathbf{Y}$ *is:*
$$
    X^2=\parallel\mathbf{Y}-\mathbf{Q}\parallel^2=\sum_i(Y_i-Q_i)^2
$$

Assuming $\mathbf{Y}_d$ and $\mathbf{Q}_d$ follow the same distribution, then $\parallel\mathbf{Y}-\mathbf{Q}\parallel^2$ has twice the variance of $\parallel\mathbf{Y}\parallel^2$, among other properties. 

## Using the minimum of the distance distribution density to estimate bandwidth for a distribution

We assume here that distance distributions will have two modes, one for the local cluster we are interested in and another that represents distances to other distributions. The aim of our algorithm is to find the minimum density between those 2 modes, which will be our estimate for the local bandwidth. Below is an example:

![Chi distribution](/figure/density.png)
Distance from each points to the star are shown on the left part of the figure. In this case the algorithm finds a bandwidth of around 2, where the min marker is on the left side.

The $\gamma$ function acts mostly as a kde in this situation and detects the minimum density between both modes.
$$
\gamma (\mathbf X_{k})=\frac{Var(\mathbf X_{k})}{(\mathbf E[\mathbf X_{k}]-X_{(k)})^2},
$$
where $\mathbf X_{k}=[X_{(1)},\dots,X_{(k)}]$.
The $\gamma$ function has high values before the first modes which avoid finding a minimum before the one we want to find. To avoid finding a minimum after the second mode we use a parameter called maxClusterSize with a default value of 0.75 which limits minimum finding to 75% of all closest neighbors. There is also another parameter called minClusterSize with default value 10 which is used to avoid the variance in density with the closest neighbors. Those parameters can be loosely set and their exact value won't affect the result.

### Example results of cluster size estimation
Below is an example where we esitmate the number of near neighbors (N) that are in the same cluster using the above method.
![Chi distribution](/figure/NestimateGood.png)

The estimation is good for points near the center and away from other clusters but worse for points that are between clusters where in some cases it takes value we set for maxClusterSize.

Here is another example were there is less separation between clusters and we can see that points between clusters get a N estimate above 700, which is determined by the parameter maxClusterSize in this case. Those estimates will be classified as bad and those points won't be clustered during the mean shift.

![Chi distribution](/figure/NestimateBad.png)

## Details on $\gamma$ function and the minimum detection

The aim of $\gamma$ is to estimate de density, but it could have been done with standard KDE methods. Below is a figure comparing the 2.
![Chi distribution](/figure/density_with_kde.png)

The "kde" plot is obtained by performing kernel density estimation on the distance distribution from the star in the dataset on the right to all other points. As expected the density is low and varying a lot for low distances and there is a minimum between 2 modes. The minimum between the modes is the location we want to find. If we simply look for the minimum value of all kde values there might be 2 problems:
1. The minimum might be found before the first mode.
2. The minimum might be found after the second mode. 

To address the first problem we use the function $\gamma$ instead of the kde. Experimentally, it effectively acts as a kde until the first minimum but has high values before that, which avoids detecting a minimum before the first mode. 

Looking at the formula you can see that if there's only 2 values, $\gamma(\mathbf X_{2})=1$. Also note that the denominator reacts a lot faster than the numerator and is inversly proportional to the density.

The second problem is solved by having the maximum cluster size parameters. Indeed, usually the density after the second mode is only smaller near the end. So removing the tail of the distance distribution is enough to avoid finding a minimum in density after the second mode.

The $\gamma$ function is not well tested as a kde, so there might be so unforseen situations. It comes from (https://www.stat.cmu.edu/technometrics/59-69/VOL-01-03/v0103217.pdf) and was used as a maximum likelihood estimator for truncated normal distribution.

## Computional complexity

The complexity of this algorithm is currently $O(n^2 \log n)$, which is due to calculating sorted distance distributions for each points. It could potentially be reduced for large datasets by using a smaller maximum cluster size and finding nearest neighbors and their distance instead of sorting the whole distance matrix. Assuming m=max cluster size, the complexity could go down to $O(nm \log n)$, or even lower with approximate nearest neighbor search.