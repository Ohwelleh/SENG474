'''
                Machine Learning Lloyd's Algorithm (K-means)
            SENG 474 Data Mining, Spring 2021, University of Victoria.
                    Professor: Nishant Mehta
                        Austin Bassett
'''

# Modules.
import sys
import matplotlib.pyplot as plt
import LloydHelpers as LH
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Checking if the proper number of arguments were pased at the command line.
if len(sys.argv) != 5:
    print('Error: This program requires 5 arguments.')
    print(f'{len(sys.argv)} arguments were passed')
    exit()

def uniformInitializationResults(clusteringData, kValue, fileName, dimension, loopMax):

    # Uniform Random Initialization.
    uniformRandomCenters = LH.uniformInitialization(clusteringData, kValue)

    # Run Lloyds Algorithm using Uniform Initialization.
    uniformResults, cluster = LH.kMeansClustering(clusteringData, uniformRandomCenters, loopMax, kValue)

    # The following plot data was modified from the following source.
    # Source: https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
    # Plot 2D scatter plot.
    plt.figure()
    title = f'Uniform K Means 2D Scatter Plot: {fileName} using {kValue} Clusters'
    plt.title(title)

    if dimension != 2:
        
        # Extracting the cluster points.
        for key in cluster.keys():
            xAxis = [ele[0] for ele in cluster[key]]
            yAxis = [ele[1] for ele in cluster[key]]
            zAxis = [ele[2] for ele in cluster[key]]
            plt.scatter(xAxis, yAxis, zAxis, marker='.')

    else:

        # Extracting the cluster points.
        for key in cluster.keys():
            xAxis = [ele[0] for ele in cluster[key]]
            yAxis = [ele[1] for ele in cluster[key]]
            plt.scatter(xAxis, yAxis, marker='.')

    saveFile = f'Lloyd\'s Images/Uniform K Means 2D Scatter {fileName} using {kValue} Clusters.png'
    plt.savefig(saveFile)

    # The following 3D scatter plot code used the following source as reference.
    # Source: https://stackoverflow.com/questions/1985856/how-to-make-a-3d-scatter-plot-in-python
    # 3D Scatter Plot
    fig = plt.figure()
    ax = Axes3D(fig)
    title = f'Uniform K Means 3D Scatter Plot: {fileName} using {kValue} Clusters'
    plt.title(title)

    if dimension != 2:
        
        # Extracting the cluster points.
        for key in cluster.keys():
            xAxis = [ele[0] for ele in cluster[key]]
            yAxis = [ele[1] for ele in cluster[key]]
            zAxis = [ele[2] for ele in cluster[key]]
            ax.scatter(xAxis, yAxis, zAxis, marker='.')

    else:

        # Extracting the cluster points.
        for key in cluster.keys():
            xAxis = [ele[0] for ele in cluster[key]]
            yAxis = [ele[1] for ele in cluster[key]]
            ax.scatter(xAxis, yAxis, marker='.')

    saveFile = f'Lloyd\'s Images/Uniform K Means 3D Scatter {fileName} using {kValue} Clusters.png'
    plt.savefig(saveFile)

def kMeansPlusPlusResults(clusteringData, kValue, fileName, dimension, loopMax):

    # K-Means++ Initialization.
    kPlusPlusCenters = LH.kMeansPlusPlus(clusteringData, kValue)

    # Run Lloyd's Algorithm using K-Means++ Initialization.
    kPlusResults, cluster = LH.kMeansClustering(clusteringData, kPlusPlusCenters, loopMax, kValue)
    
    # The following plot data was modified from the following source.
    # Source: https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
    # Plot 2D scatter plot.
    plt.figure()
    title = f'K-Means++ 2D Scatter Plot: {fileName} using {kValue} Clusters'
    plt.title(title)

    if dimension != 2:
        
        # Extracting the cluster points.
        for key in cluster.keys():
            xAxis = [ele[0] for ele in cluster[key]]
            yAxis = [ele[1] for ele in cluster[key]]
            zAxis = [ele[2] for ele in cluster[key]]
            plt.scatter(xAxis, yAxis, zAxis, marker='.')

    else:

        # Extracting the cluster points.
        for key in cluster.keys():
            xAxis = [ele[0] for ele in cluster[key]]
            yAxis = [ele[1] for ele in cluster[key]]
            plt.scatter(xAxis, yAxis, marker='.')

    saveFile = f'Lloyd\'s Images/K-Means++ {fileName} using {kValue} Clusters.png'
    plt.savefig(saveFile)

    # The following 3D scatter plot code used the following source as reference.
    # Source: https://stackoverflow.com/questions/1985856/how-to-make-a-3d-scatter-plot-in-python
    # 3D Scatter Plot
    fig = plt.figure()
    ax = Axes3D(fig)
    title = f'K-Means++ 3D Scatter Plot: {fileName} using {kValue} Clusters'
    plt.title(title)

    if dimension != 2:
        
        # Extracting the cluster points.
        for key in cluster.keys():
            xAxis = [ele[0] for ele in cluster[key]]
            yAxis = [ele[1] for ele in cluster[key]]
            zAxis = [ele[2] for ele in cluster[key]]
            ax.scatter(xAxis, yAxis, zAxis, marker='.')

    else:

        # Extracting the cluster points.
        for key in cluster.keys():
            xAxis = [ele[0] for ele in cluster[key]]
            yAxis = [ele[1] for ele in cluster[key]]
            ax.scatter(xAxis, yAxis, marker='.')

    saveFile = f'Lloyd\'s Images/K-Means++ 3D Scatter {fileName} using {kValue} Clusters.png'
    plt.savefig(saveFile)


def main():

    # Retrieve the commmand line arguments.
    dataFile = sys.argv[1]
    uniformClusterVal = int(sys.argv[2])
    kPlusClusterVal = int(sys.argv[3])
    loopMax = int(sys.argv[4])

    # Storing the file name for the output graphs (excluding the file extension).
    length = len(dataFile) - 4
    fileName = dataFile[:length]

    # # Reading in the data.
    rawData = pd.read_csv(dataFile)
    dimension = len(rawData.columns)

    # Convert data to numpy array.
    clusteringData = rawData.values

    # Uniform Random Initialization.
    if uniformClusterVal >= 1:
        uniformInitializationResults(clusteringData, uniformClusterVal, fileName, dimension, loopMax)

    # K-Means++ Initialization.
    if kPlusClusterVal >= 1:
        kMeansPlusPlusResults(clusteringData, kPlusClusterVal, fileName, dimension, loopMax)


if __name__ == '__main__':
    main()
