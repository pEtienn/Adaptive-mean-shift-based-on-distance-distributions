from sklearn.neighbors import KDTree

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy.stats as sStats

from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances
import scipy.ndimage as ndimage

import warnings

class ScaleLearner:
    """
    A class used to manipulate and store information linked to distance distributions.

    It calculates the gamma function (a kind of KDE) on distance distributions and finds the minimum density of the gamma function.
    """
    def __init__(self,distanceMatrix):
        self.dMSorted=np.sort(distanceMatrix,axis=1)[:,1:]

        if self.dMSorted.ndim==2:
            v,n=self.dMSorted.shape
        else:
            n=self.dMSorted.size
            self.dMSorted=self.dMSorted[np.newaxis,...]
        self.n=n

        self.gamma=np.ones(self.dMSorted.shape)

        u=np.cumsum(self.dMSorted,axis=1)/np.arange(1,n+1) #truncated means
        m2=np.cumsum(self.dMSorted**2,axis=1)/np.arange(1,n+1) #second moment
        V=m2-u**2 
        self.gamma[:,1:]=V[:,1:]/(1e-5+u[:,1:]-self.dMSorted[:,1:])**2

    def minDensityPositionEstimate(self,minClusterSize,maxClusterSize):
        """        
        Finds the minimum gamma value for each distance distribution and return it's index, 
        which represents the number of points in the local distribution.
        """

        minClusterSizeRatio=5
        self.filteredGamma=ndimage.gaussian_filter1d(self.gamma,minClusterSize/minClusterSizeRatio,mode='nearest',axis=1) 

        self.minIdx=np.int64((np.argmin(self.filteredGamma[:,minClusterSize:maxClusterSize],axis=1)+minClusterSize))

class DatasetWrapper():
    
    def __init__(self,X):
        """
        X:dataset
        label: label for each element of X,starts from 0 and increments by 1
        """
        
        """
        if annoyFound:
            f= X.shape[1]
            self.tree = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed
            for i in range(X.shape[0]):
                v = X[i,:]
                self.tree.add_item(i, v)
            self.tree.build(10) # 10 trees
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
    def __init__(self,X,minClusterSize=10,maxClusterSize=0.75,removeBadEstimates=True,printInfo=False,sigmaFactor=5):
        """Estimates the local distribution size for each point.

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
        self._dM=pairwise_distances(X)

        if (maxClusterSize>0.8 and maxClusterSize<=1) or (maxClusterSize>1 and (X.shape[0]/maxClusterSize)>0.8):
            print('Maximum value for maxClusterSize is 0.8. It is now 0.8.')

        minClusterSize=int(minClusterSize) if minClusterSize>=1 else int(X.shape[0]*minClusterSize)
        maxClusterSize=int(maxClusterSize) if maxClusterSize>=1 else int(X.shape[0]*maxClusterSize)

        if minClusterSize>=maxClusterSize:
            raise NameError('Invalid cluster sizes: min cluster size is bigger then max cluster size.')
        
        self._maxClusterSize=maxClusterSize
        self._minClusterSize=minClusterSize

        self._scaleLearner=ScaleLearner(self._dM)
        self._scaleLearner.minDensityPositionEstimate(minClusterSize,maxClusterSize)
        self._closePointsThreshold=np.percentile(self._scaleLearner.dMSorted[:,0],1) #sets the treshold to combine points during meanshift
        if self._closePointsThreshold==0:
            self._closePointsThreshold=1e-2
        self._sigmaFactor=sigmaFactor # good?
        self.X=X
        self.N=X.shape[0]
        self._estimatedClusterSize=self._scaleLearner.minIdx #estimate of the cluster size for each point
        self._estimatedSigma=self._scaleLearner.dMSorted[np.arange(self._scaleLearner.dMSorted.shape[0]),self._scaleLearner.minIdx]/self._sigmaFactor
        self._detectBadEstimates()
        if printInfo==True:
            print('There are '+str(np.sum(np.invert(self.goodEstimateMask)))+' bad minimum density estimates.')
        if removeBadEstimates==True:
            X=self._removeBadEstimates(X)
            if printInfo==True:
                print('They have been removed from the data used during the mean shift and assigned a label \'-1\'.')
            idx=np.where(np.invert(self.goodEstimateMask))
            dM=np.delete(self._dM,idx,axis=0)
            self._dM=np.delete(dM,idx,axis=1)
            self._badEstimatesRemoved=True
        else:
            self._changeBadNiEstimates
            if printInfo==True:
                print('To remove them from affecting the result set the parameter removeBadEstiamtes to True.')
            self._badEstimatesRemoved=False
        self._XW=DatasetWrapper(X)

    def _detectBadEstimates(self):
        """Identify values that were set by the max or min cluster size.
        """
        #1.1 factors checks a bit after the maxclusterSize, if the minimum changes then the minimum is caused by the maxclusterSize
        newMinClusterSize=int(0.66*self._minClusterSize)
        min1=np.argmin(self._scaleLearner.filteredGamma[:,newMinClusterSize:int(self._maxClusterSize*1.1)],axis=1)+newMinClusterSize
        self.goodEstimateMask=np.invert(min1>self._estimatedClusterSize)

    def _changeBadNiEstimates(self):
        """Change bad cluster size estimates to the nearest valid cluster size estimate.
        """

        idx=np.argwhere(np.invert(self.goodEstimateMask))
        proximityMatrix=np.argsort(self._dM,axis=1)
        NOrdered=np.zeros(proximityMatrix.shape)
        
        for i in range(proximityMatrix.shape[1]):
            NOrdered[proximityMatrix==i]=self._estimatedClusterSize[i]
            
        for i in range(idx.shape[0]):
            ns=NOrdered[idx[i],:]
            #print(ns)
            self._estimatedClusterSize[idx[i]]=ns[ns>0][0] #first n above 0

        self._estimatedClusterSize=np.int64(self._estimatedClusterSize)
        self._estimatedSigma=self._scaleLearner.dMSorted[np.arange(self._scaleLearner.dMSorted.shape[0]),self._estimatedClusterSize]/self._sigmaFactor

    def _removeBadEstimates(self,X):
        """Remove points with bad cluster size estimates.
        """

        mask=self.goodEstimateMask
        X=X[mask,:]
        self._estimatedClusterSize=self._estimatedClusterSize[mask]
        self._estimatedSigma=self._estimatedSigma[mask]
        return X

    def _getLocalSigma(self,centroids,k=5):
        """Return a sigma estimate for the centroid.

        Parameters
        ----------
        centroids : numpy array, (n,d)
            list of n centroid in d dimensions
        k : int, optional
            median parameter


        Returns
        -------
        Sigma parameter for each point
        """
        
        nIdx=self._XW.kNN(centroids,k=k)

        sigmas=self._estimatedSigma[nIdx]
        return np.median(sigmas,axis=1)

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
        
    def _meanShift(self,printClustersDebug=False):
        """Performs adaptive mean shift.

        Parameters
        ----------
        printClustersDebug : bool, optional
            Print clusters at each iteration after the first, only works in 2D.
        """
        d=self._XW.X.shape[1]
        centroids=np.copy(self._XW.X)
        labels=np.arange(self._XW.X.shape[0],dtype=np.int64)
        centroidWeights=np.ones(centroids.shape[0])
       
        #each j loop all centroids move towards the local centroid average
        for j in range(100):
            newCentroids=np.copy(centroids) 
            oldCentroids=np.copy(centroids) 
            
            if j==0:
                localSigmas=self._estimatedSigma[:,np.newaxis]
                XcAll=self._dM
            else:
                XcAll=pairwise_distances(centroids,centroids)
                localSigmas=self._getLocalSigma(centroids)[:,np.newaxis]
            XcTresholds=localSigmas*self._sigmaFactor
            distanceMask=XcAll<XcTresholds

            centroidWeightsSquare=np.tile(centroidWeights,(centroids.shape[0],1))

            k=(centroidWeightsSquare*distanceMask)*np.exp(-(XcAll*distanceMask)**2/(2*localSigmas**2))
            for i in range(centroids.shape[0]):
                newCentroids[i,:] = np.sum(centroids[distanceMask[i,:]]*k[i,distanceMask[i,:]][:,np.newaxis],axis=0)/np.sum(k[i,distanceMask[i,:]])

            [centroids,centroidWeights,labels] = self._clusterClosePoints(newCentroids,centroidWeights,labels,threshold=self._closePointsThreshold)

            #debug tools
            if printClustersDebug:
                if j>1:
                    print2DClusters(self.X,labels,centroids)

            if oldCentroids.shape[0]==centroids.shape[0] and printClustersDebug:
                a=np.sum(np.linalg.norm(centroids-oldCentroids,axis=1))
                print('{:.6f}'.format(a))
            #######################

            #end condition
            if oldCentroids.shape[0]==centroids.shape[0] and j>1 and np.sum(np.linalg.norm(centroids-oldCentroids,axis=1))<1e-5:
                break
        
        #remove clusters smaller than minimum size
        belowMinClusterSize=np.where(centroidWeights<self._minClusterSize)[0]
        aboveMinClusterSize=np.where(centroidWeights>self._minClusterSize)[0]
        for i in range(belowMinClusterSize.shape[0]):
            labels[labels==belowMinClusterSize[i]]=-1

        self.centroidSigma=localSigmas[aboveMinClusterSize]
        self.clusterCentroids=centroids[aboveMinClusterSize]
        self.centroidWeights=centroidWeights[aboveMinClusterSize]
        if self._badEstimatesRemoved==True:
            self.labels=-np.ones(self.N)
            self.labels[self.goodEstimateMask]=labels
        else:
            self.labels=labels
        
        #make sure that labels are from 0 to the number of classes, excluding -1
        self.labels=self._normalizeLabels(self.labels)

    def _normalizeLabels(self,labels):
        uniques=np.unique(labels) 
        uniques=uniques[uniques>=0]
        newLabels=np.copy(labels)
        for i in range(uniques.size):
            newLabels[labels==uniques[i]]=i
        return newLabels


    def _clusterUnlabeledPoints(self):
        """Assigns unlabelled points to the closest cluster centroid accounting for the scale of each cluster.
        """
        points=self.X[self.labels==-1]
        distance=pairwise_distances(points,self.clusterCentroids)
        normalizedDistance=distance**2/(2*self.centroidSigma.T**2)
        newLabels=np.argmin(normalizedDistance,axis=1)
        self.labels[self.labels==-1]=newLabels

    def meanShift(self,clusterUnlabeledPoints=True):

        """Performs adaptive mean shift and then clusters unlabeled points  to the closest cluster centroid 
        accounting for the scale of each cluster.
        """
        self._meanShift()
        if clusterUnlabeledPoints==True:
            self._clusterUnlabeledPoints()

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