import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy.stats as sStats
from scipy.spatial import distance_matrix
import scipy.ndimage as ndimage

from sklearn.neighbors import KDTree
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances

cmap=mpl.colormaps['tab10']
my_dpi=200

import warnings


class DistanceDistribution:
    """
    A class used to manipulate and store information linked to distance distributions.

    It calculates the gamma function (a kind of KDE) on distance distributions and finds the minimum density of the gamma function.
    """
    def __init__(self,distanceMatrix,minClusterSize,maxClusterSize):
        self.minClusterSize=minClusterSize
        self.maxClusterSize=maxClusterSize
        self.dMSorted=np.sort(distanceMatrix,axis=1)[:,1:] #removes distance of 0

        if self.dMSorted.ndim==2:
            _,n=self.dMSorted.shape
        else:
            n=self.dMSorted.size
            self.dMSorted=self.dMSorted[np.newaxis,...]

        self.gamma=np.ones(self.dMSorted.shape)

        u=np.cumsum(self.dMSorted,axis=1)/np.arange(1,n+1) #truncated means
        m2=np.cumsum(self.dMSorted**2,axis=1)/np.arange(1,n+1) #second moment
        V=m2-u**2 
        self.gamma[:,1:]=V[:,1:]/(1e-5+u[:,1:]-self.dMSorted[:,1:])**2

    def minDensityPositionEstimate(self,minIdxOffset):
        """        
        Finds the minimum gamma value for each distance distribution and return it's index, 
        which represents the number of points in the local distribution.
        """

        minClusterSizeRatio=5
        #gaussian filter avoids very local variations in density where we could find a minimum
        self.filteredGamma=ndimage.gaussian_filter1d(self.gamma,self.minClusterSize/minClusterSizeRatio,mode='nearest',axis=1) 

        self.minIdx=np.int64((np.argmin(self.filteredGamma[:,self.minClusterSize:self.maxClusterSize],axis=1)+self.minClusterSize))+minIdxOffset
    
    def printDistanceDistribution(self,i,axisNumberNeighbor=True):
        g=self.gamma[i,:]
        #x=np.arange(g.size)+1#dd[i,offset:]
        distance=self.dMSorted[i,:]
        if axisNumberNeighbor==True:
            x=np.arange(1,g.shape[0]+1)
        else:
            x=distance


        filteredG=self.filteredGamma[i,:]
        minIdx=self.minIdx[i] #ms._scaleLearner.minIdx[i]


        title='Density versus distance'
        minDistance=x[self.minClusterSize]
        maxDistance=x[self.maxClusterSize]
        plt.title(title)
        plt.plot(x,g,label=r"$\gamma$")
        kde = KernelDensity(kernel='gaussian', bandwidth='silverman').fit(distance[:,np.newaxis])
        kdeValue=np.exp(kde.score_samples(distance[:,np.newaxis]))
        plt.plot(x,kdeValue,label=r"KDE density")
        plt.scatter(x[minIdx],g[minIdx],c='y',s=100,label='min')
        plt.vlines([minDistance,maxDistance],np.min(g),np.max(g),color=cmap(9),linestyles='dashed',label='Cluster size \nboundaries')
        if axisNumberNeighbor==True:
            plt.xlabel('Nearest neighbors')
        else:
            plt.xlabel('Distance')
        plt.ylabel(r"Density")

        plt.legend()
        plt.show()

class DatasetWrapper():
    
    def __init__(self,X):
        """
        Wrapper around a KD tree for nearest neighbor search.
        X:dataset
        """

        self.kdt = KDTree(X, leaf_size=10, metric='euclidean')

        self.X=X
        self.d=X.shape[1]

    def kNN(self,x,k=5,removeClosestNeighbour=False):
        if removeClosestNeighbour==True:
            k+=1
        voters=self.kdt.query(x, k, return_distance=False)
        if removeClosestNeighbour==True:
            voters=voters[1:]
        return voters
    

class AdaptiveMeanShift():
    """Adaptive mean shift wrapper class
    """
    def __init__(self,X,minClusterSize=5,maxClusterSize=0.75,removeBadEstimates=True,printInfo=False,minIdxOffset=0):
        """Estimates the local distribution size for each point and initiates necessary values.

        Parameters
        ----------
        X : numpy array (n,d)
            Set of n points of d dimensions
        minClusterSize : int or float
            Minimum cluster size, the value is a reasonable extreme and won't affect much the result.
            It can be specified either as an integer or a float equal to (min cluster size)/n.
        maxClusterSize : float or int, should be smaller than 0.8 or it's equivalent.
            Maximum cluster size, the value is a reasonable extreme and won't affect much the result.
            It can be specified either as an integer or a float equal to (min cluster size)/n.
        removeBadEstimates : bool, optional
            Removes datapoints with bad cluster size estimates, by default False
        """
        def verifyAndConvertBoundaries(nbPoints,minClusterSize,maxClusterSize):
            maxRatioClusterSize=0.8
            if (maxClusterSize>maxRatioClusterSize and maxClusterSize<=1) or (maxClusterSize>1 and (X.shape[0]/maxClusterSize)>maxRatioClusterSize):
                print('Maximum value for maxClusterSize is '+ str(maxRatioClusterSize) +' and has been set too high. It is now '+ str(maxRatioClusterSize)+'.')

            minClusterSize=int(minClusterSize) if minClusterSize>=1 else int(nbPoints*minClusterSize)
            maxClusterSize=int(maxClusterSize) if maxClusterSize>=1 else int(nbPoints*maxClusterSize)

            if minClusterSize>=maxClusterSize:
                raise NameError('Invalid cluster sizes: min cluster size is bigger then max cluster size.')
            return [minClusterSize,maxClusterSize]
        
        self._dM=pairwise_distances(X)
        self.N=X.shape[0]
        [self._minClusterSize,self._maxClusterSize]=verifyAndConvertBoundaries(self.N,minClusterSize,maxClusterSize)

        self._distanceDistributions=DistanceDistribution(self._dM,self._minClusterSize,self._maxClusterSize)
        self._distanceDistributions.minDensityPositionEstimate(minIdxOffset)

        self._closePointsThreshold=np.percentile(self._distanceDistributions.dMSorted[:,0],1) #sets the treshold to combine points during meanshift
        if self._closePointsThreshold==0:
            self._closePointsThreshold=1e-2

        self.X=X

        #Setting estimated distance parameters
        self._estimatedClusterSize=self._distanceDistributions.minIdx #estimate of the cluster size for each point
        self._estimatedDistanceSigma=np.zeros(self.N)
        self._estimatedDistanceMean=np.zeros(self.N)
        for i in range(self.N):
            self._estimatedDistanceSigma[i]=np.std(self._distanceDistributions.dMSorted[i,:self._estimatedClusterSize[i]])
            self._estimatedDistanceMean[i]=np.mean(self._distanceDistributions.dMSorted[i,:self._estimatedClusterSize[i]])
        self._estimatedDistanceThreshold=self._distanceDistributions.dMSorted[np.arange(self.N),self._estimatedClusterSize]

        #deal with points at the same location TO IMPROVE
        self._estimatedDistanceSigma[self._estimatedDistanceSigma==0]=np.percentile(self._estimatedDistanceSigma,10)
        self._estimatedDistanceMean[self._estimatedDistanceMean==0]=np.percentile(self._estimatedDistanceMean,10)

        X=self._manageBadEstimates(X,printInfo,minIdxOffset,removeBadEstimates)
        self._XW=DatasetWrapper(X)

    def _manageBadEstimates(self,X,printInfo,minIdxOffset,removeBadEstimates=True):
        """Manages points that have a bad cluster size estimation
        """
        def detectBadEstimates(filteredGamma,maxClusterSize,minClusterSize,estimatedClusterSize):
            """Identify values that were set by the max or min cluster size.
            """
            #1.1 factors checks a bit after the maxclusterSize, if the minimum changes then the minimum is caused by the maxclusterSize
            newMinClusterSize=int(0.66*minClusterSize)
            min1=np.argmin(filteredGamma[:,newMinClusterSize:int(maxClusterSize*1.1)],axis=1)+newMinClusterSize+minIdxOffset
            goodEstimateMask=np.invert(min1>estimatedClusterSize)
            return goodEstimateMask
            
        self.goodEstimateMask=detectBadEstimates(self._distanceDistributions.filteredGamma,self._maxClusterSize,self._minClusterSize,self._estimatedClusterSize)
        if printInfo==True:
            print('There are '+str(np.sum(np.invert(self.goodEstimateMask)))+' bad minimum density estimates.')
        if removeBadEstimates==True:
            mask=self.goodEstimateMask
            X=X[mask,:]
            self._estimatedClusterSize=self._estimatedClusterSize[mask]
            self._estimatedDistanceSigma=self._estimatedDistanceSigma[mask]
            self._estimatedDistanceMean=self._estimatedDistanceMean[mask]
            self._estimatedDistanceThreshold=self._estimatedDistanceThreshold[mask]
            if printInfo==True:
                print('They have been removed from the data used during the mean shift and assigned a label \'-1\'.')
            idx=np.where(np.invert(self.goodEstimateMask))
            dM=np.delete(self._dM,idx,axis=0)
            self._dM=np.delete(dM,idx,axis=1)
            self._badEstimatesRemoved=True
        return X

    def _getLocalDistanceParameters(self,centroids,k=5):
        """Return a sigma estimate for the centroid.

        Parameters
        ----------
        centroids : numpy array, (n,d)
            list of n centroid in d dimensions
        k : int, optional
            median parameter


        Returns
        -------
        Mean, Sigma and distance threshold parameter for each point
        """
        
        nIdx=self._XW.kNN(centroids,k=k)
        means=np.median(self._estimatedDistanceMean[nIdx],axis=1)[:,np.newaxis]
        sigmas=np.median(self._estimatedDistanceSigma[nIdx],axis=1)[:,np.newaxis]
        XcThresholds=np.median(self._estimatedDistanceThreshold[nIdx],axis=1)[:,np.newaxis]
        clusterSize=np.median(self._estimatedClusterSize[nIdx],axis=1)[:,np.newaxis]
        return [means,sigmas,XcThresholds,clusterSize]

    def _clusterClosePoints(self,points,centroidWeights,labelsFirstSize,threshold):
        """Combine centroids close to each other

        TODO: this function is pretty slow.

        Parameters
        ----------
        points : numpy array, (n,d)
            list of n centroid in d dimensions
        centroidWeights : int (n)
            Weight of each point
        labelsFirstSize : int (d)
            Label of each point.
        threshold : float
            Threshold distance to combine points
        """
        labelPoints=np.arange(points.shape[0])
        newLabelsFirstSize=np.zeros(labelsFirstSize.shape,dtype=np.int64)
        newLabels=np.ones(points.shape[0],dtype=np.int64)*-1
        clusters=np.zeros((0,points.shape[1]))
        newCentroidWeights=np.zeros(0)
        labelValue=0
        for i in range(points.shape[0]):
            if newLabels[i]==-1:
                
                soloPoints=points[newLabels==-1]
                sameMask=np.linalg.norm(soloPoints-points[i,:],axis=1)<threshold
                cluster=np.mean(soloPoints[sameMask,:],axis=0,keepdims=True)
                clusters=np.concatenate((clusters,cluster))
                newCentroidWeights=np.hstack((newCentroidWeights,np.sum((centroidWeights[newLabels==-1])[sameMask])))
                
                labelsInCluster=(labelPoints[newLabels==-1])[sameMask]
                
                y=np.where(newLabels==-1)[0]
                z=np.where(sameMask)[0]
                newLabels[y[z]]=labelValue
                newLabelsFirstSize[np.ma.in1d(labelsFirstSize,labelsInCluster).data]=labelValue
                labelValue+=1
                
        return [clusters,newCentroidWeights,newLabelsFirstSize]
        
    def _meanShift(self,printClustersDebug=False,useHighDimensionGaussianKernel=False):
        """Performs adaptive mean shift.

        Parameters
        ----------
        printClustersDebug : bool, optional
            Print clusters at each iteration after the first, only works in 2D.
        """
        def normalizeLabels(labels):
            uniques=np.unique(labels) 
            uniques=uniques[uniques>=0]
            newLabels=np.copy(labels)
            for i in range(uniques.size):
                newLabels[labels==uniques[i]]=i
            return newLabels
        
        d=self._XW.X.shape[1]
        centroids=np.copy(self._XW.X)
        labels=np.arange(self._XW.X.shape[0],dtype=np.int64)
        centroidWeights=np.ones(centroids.shape[0])
       
        #each j loop all centroids move towards the local centroid average
        for j in range(200):
            newCentroids=np.copy(centroids) 

            # get local parameters, first loop we use parameters at points directly
            if j==0:
                localSigmas=self._estimatedDistanceSigma[:,np.newaxis]
                localMeans=self._estimatedDistanceMean[:,np.newaxis]
                XcTresholds=self._estimatedDistanceThreshold[:,np.newaxis]
                clusterSizes=self._estimatedClusterSize[:,np.newaxis]
                XcAll=self._dM
            else:
                XcAll=pairwise_distances(centroids,centroids)
                [localMeans,localSigmas,XcTresholds,clusterSizes]=self._getLocalDistanceParameters(centroids)

            distanceMask=XcAll<=XcTresholds #NxN-1 matrix

            #clusterSize variant
            #distanceMask=np.tile(np.arange(XcAll.shape[1]),(XcAll.shape[0],1))<=clusterSizes


            #paragraph below calculate gaussian kernel values from one point to all other points on each line of the k matrix
            centroidWeightsSquare=np.tile(centroidWeights,(centroids.shape[0],1))
            if useHighDimensionGaussianKernel==False:
                k=(centroidWeightsSquare*distanceMask)*np.exp(-(XcAll*distanceMask)**2/(2*localSigmas**2))
            else:
                #variant of the Gaussian kernel designed for high dimensions
                val1=(XcAll*distanceMask)-(localMeans-4*localSigmas)
                num=val1.clip(min=0)**2
                k=(centroidWeightsSquare*distanceMask)*np.exp(-num/(2*(localSigmas)**2))
   

            for i in range(centroids.shape[0]):
                newCentroids[i,:] = np.sum(centroids[distanceMask[i,:]]*k[i,distanceMask[i,:]][:,np.newaxis],axis=0)/np.sum(k[i,distanceMask[i,:]])

            [newCentroids,centroidWeights,labels] = self._clusterClosePoints(newCentroids,centroidWeights,labels,threshold=self._closePointsThreshold)

            #debug tools
            if printClustersDebug:
                if j>0:
                    print2DClusters(self._XW.X,labels,newCentroids)

                if newCentroids.shape[0]==centroids.shape[0]:
                    a=np.sum(np.linalg.norm(centroids-newCentroids,axis=1))
                    print('{:.6f}'.format(a))
            #######################

            #end condition
            if newCentroids.shape[0]==centroids.shape[0] and j>1 and np.sum(np.linalg.norm(centroids-newCentroids,axis=1))<1e-5:
                centroids=newCentroids
                break
            centroids=newCentroids
        if j==99:
            print("_meanShift performed 100 loops without detecting end conditions, there might be a problem.")

        #remove clusters smaller than minimum size
        belowMinClusterSize=np.where(centroidWeights<self._minClusterSize)[0]
        aboveMinClusterSize=np.where(centroidWeights>self._minClusterSize)[0]
        for i in range(belowMinClusterSize.shape[0]):
            labels[labels==belowMinClusterSize[i]]=-1
        self.centroidWeights=centroidWeights[aboveMinClusterSize]
        self.clusterCentroids=centroids[aboveMinClusterSize]

        #estimates sigma for each cluster
        self.centroidSigma=np.zeros(self.centroidWeights.shape)
        for i in range(self.clusterCentroids.shape[0]):
            k=int(np.max([self.centroidWeights[i]/5,1]))
            [_,localSigmas,_,_]=self._getLocalDistanceParameters(centroids[[i],:],k=k)
            self.centroidSigma[i]=localSigmas[0][0]
            
        if self._badEstimatesRemoved==True:
            self.labels=-np.ones(self.N)
            self.labels[self.goodEstimateMask]=labels
        else:
            self.labels=labels
        
        #make sure that labels are from 0 to the number of classes, excluding -1
        self.labels=normalizeLabels(self.labels)

    def _classifyUnlabeledPoints(self):
        """Assigns unlabelled points to the closest cluster centroid accounting for the scale of each cluster.
        """
        sum1=np.sum(self.labels==-1)
        if sum1>0 and sum1!=self.X.shape[0]:
            points=self.X[self.labels==-1]
            distance=pairwise_distances(points,self.clusterCentroids)
            normalizedDistance=(distance**2)/(2*self.centroidSigma.T**2)
            newLabels=np.argmin(normalizedDistance,axis=1)
            self.labels[self.labels==-1]=newLabels

    def meanShift(self,classifyUnlabeledPoints=True,printClustersDebug=False,useHighDimensionGaussianKernel=False):
        """Performs adaptive mean shift and then clusters unlabeled points  to the closest cluster centroid 
        accounting for the scale of each cluster.
        """
        self._meanShift(printClustersDebug,useHighDimensionGaussianKernel)
        if classifyUnlabeledPoints==True and np.sum(self.labels==-1)>0:
            self._classifyUnlabeledPoints()

def print2DNestimates(X,estimatedClusterSize):
    """Prints the cluster size estimate at each point, only works in 2D.
    """
    plt.scatter(X[:,0],X[:,1],c=estimatedClusterSize,cmap='jet')
    plt.colorbar()
    plt.title('N estimate for each point')
        

        
def print2DClusters(X,labels,clusterCentroids):
    """Prints clusters in 2D

    Parameters
    ----------
    X : (n,d) matrix
        n points in d dimensions
    labels : int
        point labels, start at zero, -1 for unclustered points
    clusterCentroids : (labels max value,d) matrix
    """
    cmap=mpl.colormaps['tab10']
    if np.any(labels==-1):
        i=-1
        plt.scatter(X[labels==i,0], X[labels==i,1], color='y',s=15, marker='v',label='not part of a cluster' )
    i=0
    plt.scatter(X[labels==i,0], X[labels==i,1], color=cmap(i),s=15,label='member of cluster x' )
    for i in range(1,np.unique(labels).shape[0]):
        plt.scatter(X[labels==i,0], X[labels==i,1], color=cmap(i),s=15)

    i=0
    plt.scatter(clusterCentroids[i,0], clusterCentroids[i,1], marker='*',color=cmap(i),s=150,label='cluster x center',zorder=2,edgecolors='black')
    for i in range(1,clusterCentroids.shape[0]):
        plt.scatter(clusterCentroids[i,0], clusterCentroids[i,1], marker='*',color=cmap(i),s=150,zorder=2,edgecolors='black')
        plt.annotate(str(i),(clusterCentroids[i,0],clusterCentroids[i,1]))
    plt.title('meanshift clusters')
    plt.legend()
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    a=plt.xticks([])
    a=plt.yticks([])
    plt.show()

def generate2DClusters(ns,means,scales):
    d=means.shape[1]
    X=np.zeros((0,d))
    labels=np.zeros(0)
    for i in range(means.shape[0]):
        cov=np.identity(d)*scales[i]**2
        X2=sStats.multivariate_normal.rvs(mean=means[i,:],cov=cov,size=ns[i])
        X=np.concatenate((X,X2))
        labels=np.append(labels,np.ones(ns[i])*i)
    return [X,labels]