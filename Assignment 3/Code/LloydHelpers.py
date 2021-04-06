# Modules.
import sys, random, csv
from scipy.spatial import distance
import pandas as pd
import numpy as np

def uniformInitialization(data, kValue):

    # Initialize to K random centers.
    temp = np.random.choice(len(data), kValue,replace=False)
    randomCenters = data[temp, :]

    return randomCenters

def kMeansPlusPlus(data, kValue):

    # This K-Means++ was modified from the following source:
    # Source: https://www.geeksforgeeks.org/ml-k-means-algorithm/

    # Initializing a list of centroids, add a ranomly selected data point to the list.
    centroids = []
    centroids.append(data[np.random.randint(data.shape[0]), :])

    # Compute the remaining k-1 centroids.
    for i in range(kValue - 1):

        # Storing the distance of data points from nearest centroid.
        distanceList = []
        for row in range(data.shape[0]):

            # Grabbing all the data in row J of the matrix.
            point = data[row, :]

            # Grabbing the max value of a variable python can handle.
            pythonMax = sys.maxsize

            # Calculate distance of 'point' from each of the previously selectec centroid
            # and store the minimum.
            for j in range(len(centroids)):
                #tempDistance = distance.euclidean(point, centroids[j])
                tempDistance = np.sum((point - centroids[j])**2)
                minimumTemp = min(pythonMax, tempDistance)

            distanceList.append(minimumTemp)

        # Select the data point with the maximum distance as our next center.
        distanceList = np.array(distanceList)
        nextCenter = data[np.argmax(distanceList), :]
        centroids.append(nextCenter)
    
    return centroids

def clusterPoints(rawData, centersData):

    # Creating a dictionary of clusters.
    # Where the value is a list of points belong to that cluster.
    cluster = {}

    # Looping through each entry of rawData.
    for x in rawData:

        # Finding the best key.
        bestCenterKey = min([(i[0], np.linalg.norm(x-centersData[i[0]])) for i in enumerate(centersData)], key=lambda t:t[1])[0]

        # If key is in the cluster, append the value to the list.
        if bestCenterKey in cluster:

            cluster[bestCenterKey].append(x)

        else:
            # Other wise create a new list at the bestCenterKey index.
            cluster[bestCenterKey] = [x]

    return cluster

def pickNewCenters(rawData, cluster):

    # List of new centers.
    newCenter = []
    
    # Sorting the cluster keys.
    sortedKeys = sorted(cluster.keys())

    # Loop through each key.
    for key in sortedKeys:

        # Calculating the mean of the list of the key.
        newCenter.append(np.mean(cluster[key], axis=0))

    return newCenter

def checkingConvergence(newCenter, oldCenter):

    # Checking if the data in the centers are equal.
    return (set([tuple(newVal) for newVal in newCenter]) == set([tuple(oldVal) for oldVal in oldCenter]))

def kMeansClustering(rawData, centersData, loopMax, kVal):

    # This K-Means was modified from the following source:
    # Source: https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
    
    # Flag for checking for divergence.
    dataDiverged = True

    oldCenters = centersData
    # Loop until convergence.
    for i in range(loopMax):

        # Cluster the data.
        cluster = clusterPoints(rawData, oldCenters)

        # Find new center.
        newCenters = pickNewCenters(oldCenters, cluster)

        # Checking if the centers are equal, meaning no change has occured since last iteration
        if checkingConvergence(newCenters, oldCenters):
            dataDiverged = False
            break

        oldCenters = newCenters

    if dataDiverged:
        print(f'Convergence did not happen in {loopMax} iterations')

    return newCenters, cluster
    