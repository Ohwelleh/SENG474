# Modules
import sys, random, csv
import matplotlib.pyplot as plt
from scipy.spatial import distance
import LloydHelpers as LH
import pandas as pd
import numpy as np

def uniformKmeans(clusteringData, fileName, rangeMax):

    # Initializing a list for storing the cost.
    cost = []

    # Initializing a range of k cluster.
    K = range(2,rangeMax)

    # Looping through each k cluster.
    for k in K:

        # Uniform Random Initialization.
        uniformRandomCenters = LH.uniformInitialization(clusteringData, k)

        # Run Lloyds Algorithm using Uniform Initialization.
        uniformResults, cluster = LH.kMeansClustering(clusteringData, uniformRandomCenters, 100, k)

        # Calculating the cost.
        # Reference source: https://gist.github.com/adityachandupatla/622d50d1dce80777dce4b2839760777f
        costValue = 0
        for key in cluster.keys():
            for listEle in cluster[key]:
                costValue = costValue + np.linalg.norm(listEle - uniformResults[key])

        cost.append(costValue)

    # Plot Elbow.
    # Reference source: https://pythonprogramminglanguage.com/kmeans-elbow-method/
    plt.figure()
    plt.xlabel('K Values')
    plt.ylabel('Cost')
    plt.title('Elbow Graph of Cost vs K')
    plt.plot(K, cost, 'bx-')
    saveFile = f'Elbow Graph Images/Uniform K Means Elbow {fileName}.png'
    plt.savefig(saveFile)

def kMeansPlus(clusteringData, fileName, rangeMax):

    # Initializing a list for storing the cost.
    cost = []

    # Initializing a range of k cluster.
    K = range(2,rangeMax)

    # Looping through each k cluster.
    for k in K:
        
        # K-Means++ Initialization
        kPlusCenters = LH.kMeansPlusPlus(clusteringData, k)

        # Run Lloyds Algorithm using Uniform Initialization.
        kPlusResults, cluster = LH.kMeansClustering(clusteringData, kPlusCenters, 100, k)
    
        # Calculating the cost.
        # Reference source: https://gist.github.com/adityachandupatla/622d50d1dce80777dce4b2839760777f
        costValue = 0
        for key in cluster.keys():
            for listEle in cluster[key]:
                costValue = costValue + np.linalg.norm(listEle - kPlusResults[key])

        cost.append(costValue)

    # Plot Elbow.
    # Reference source: https://pythonprogramminglanguage.com/kmeans-elbow-method/
    plt.figure()
    plt.xlabel('K Values')
    plt.ylabel('Cost')
    plt.title('Elbow Graph of Cost vs K')
    plt.plot(K, cost, 'bx-')
    saveFile = f'Elbow Graph Images/K-Means++ Elbow {fileName}.png'
    plt.savefig(saveFile)

def main():
    
    # Retrieve the commmand line arguments.
    dataFile = sys.argv[1]
    rangeMax = int(sys.argv[2])

    # Set rangeMax to 1 if a number 0 or less is passed.
    if rangeMax <= 0:
        rangeMax = 1

    # Storing the file name for the output graphs (excluding the file extension).
    length = len(dataFile) - 4
    fileName = dataFile[:length]

    # Reading in the data.
    rawData = pd.read_csv(dataFile)

    # Convert data to numpy array.
    clusteringData = rawData.values

    # Uniform Random Initialization Elbow Graph.
    uniformKmeans(clusteringData, fileName, rangeMax)

    # K-Means++ Initialization Elbow Graph.
    kMeansPlus(clusteringData, fileName, rangeMax)


if __name__ == '__main__':
    main()