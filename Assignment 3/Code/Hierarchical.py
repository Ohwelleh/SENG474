'''
            Machine Learning Hierarchical Agglomerative Clustering
           SENG 474 Data Mining, Spring 2021, University of Victoria.
                    Professor: Nishant Mehta
                        Austin Bassett
'''

# Modules.
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D

def HierarchicalClusteringSingle(data, numCluster, fileName, dimension):
    
    # Initialize the model.
    singleModel = AgglomerativeClustering(n_clusters=numCluster, affinity='euclidean', linkage='single')

    # Fit the model to the data.
    singleModel.fit(data)

    # The following plot data was modified from the following source.
    # Source: https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
    # Plot the data.
    plt.figure()
    title = f'Single Linkage 2D Scatter Plot: {fileName}'
    plt.title(title)

    if dimension != 2:
        plt.scatter(data.iloc[:,0], data.iloc[:,1], data.iloc[:,2], c=singleModel.labels_, marker='.', cmap='jet')
    else:
        plt.scatter(data.iloc[:,0], data.iloc[:,1], c=singleModel.labels_, marker='.', cmap='jet')

    saveFile = f'Single Linkage Images/Single 2D Scatter {fileName}.png'
    plt.savefig(saveFile)

    # The following 3D scatter plot code used the following source as reference.
    # Source: https://stackoverflow.com/questions/1985856/how-to-make-a-3d-scatter-plot-in-python
    # 3D Scatter Plot
    fig = plt.figure()
    ax = Axes3D(fig)
    title = f'Single Linkage 3D Scatter Plot: {fileName}'
    plt.title(title)

    if dimension != 2:
        ax.scatter(data.iloc[:,0], data.iloc[:,1], data.iloc[:,2], c=singleModel.labels_, marker='.', cmap='jet')
    else:
        ax.scatter(data.iloc[:,0], data.iloc[:,1], c=singleModel.labels_, marker='.', cmap='jet')

    saveFile = f'Single Linkage Images/Single 3D Scatter {fileName}.png'
    plt.savefig(saveFile)


def HierarchicalClusteringAverage(data, numCluster, fileName, dimension):
    
    # Initialize the model.
    averageModel = AgglomerativeClustering(n_clusters=numCluster, affinity='euclidean', linkage='average')

    # Fit the model to the data.
    averageModel.fit(data)

    # The following plot data was modified from the following source.
    # Source: https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
    # Plot the data.
    plt.figure()
    title = f'Average Linkage 2D Scatter Plot: {fileName}'
    plt.title(title)

    if dimension != 2:
        plt.scatter(data.iloc[:,0], data.iloc[:,1], data.iloc[:,2], c=averageModel.labels_, marker='.', cmap='jet')
    else:
        plt.scatter(data.iloc[:,0], data.iloc[:, 1], c=averageModel.labels_, marker='.', cmap='jet')

    saveFile = f'Average Linkage Images/Average 2D Scatter {fileName}.png'
    plt.savefig(saveFile)

    # The following 3D scatter plot code used the following source as reference.
    # Source: https://stackoverflow.com/questions/1985856/how-to-make-a-3d-scatter-plot-in-python
    # 3D Scatter Plot
    fig = plt.figure()
    ax = Axes3D(fig)
    title = f'Average Linkage 3D Scatter Plot: {fileName}'
    plt.title(title)

    if dimension != 2:
        ax.scatter(data.iloc[:,0], data.iloc[:,1], data.iloc[:,2], c=averageModel.labels_, marker='.', cmap='jet')
    else:
        ax.scatter(data.iloc[:,0], data.iloc[:, 1], c=averageModel.labels_, marker='.', cmap='jet')

    saveFile = f'Average Linkage Images/Average 3D Scatter {fileName}.png'
    plt.savefig(saveFile)

def main():
    
    # Getting the command line input variables.
    dataFile = sys.argv[1]
    singleCluster = int(sys.argv[2])
    averageCluster = int(sys.argv[3])

    # Read the data.
    data = pd.read_csv(dataFile)

    # Getting the dimension of the data.
    dimension = len(data.columns)
    if dimension < 2 or dimension > 3:
        print('Error: This program was designed to run on 2 or 3 demension data')
        print(f'This file is of dimension: {dimension}')
        exit()

    # Getting the name of the file (excluding the file extension).
    length = len(dataFile) - 4
    fileName = dataFile[:length]

    # Single linkage cluster.
    HierarchicalClusteringSingle(data, singleCluster, fileName, dimension)

    # Average linkage cluster.
    HierarchicalClusteringAverage(data, averageCluster, fileName, dimension)




if __name__ == '__main__':
    main()