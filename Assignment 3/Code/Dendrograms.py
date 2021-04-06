# Modules.
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import pandas as pd
import sys

def generateDenodrogram():
    
    # Get the file name from command line.
    try:
        fileName = sys.argv[1]
    except:
        print("Error: No file was passed.")
        exit()

    # Reading in the data.
    data = pd.read_csv(fileName)

    # Extracting the file name (no extension).
    length = len(fileName) - 4

    # The following Dendrogram code was modified from the following source.
    # Source: https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
    # Creating the single linkage diagram.
    plt.figure(figsize=(10, 7))
    title = "Single Dendogram: " + fileName[:length]
    plt.title(title)
    dendoGram = shc.dendrogram(shc.linkage(data, method='single'), truncate_mode='lastp')
    saveTitle = f'Single Linkage Images/Single{fileName[:length]}.png'
    plt.savefig(saveTitle)

    # Creating the average linkage diagram.
    plt.figure(figsize=(10, 7))
    title = "Average Dendogram: " + fileName[:length]
    plt.title(title)
    dendoGram = shc.dendrogram(shc.linkage(data, method='average'), truncate_mode='lastp')
    saveTitle = f'Average Linkage Images/Average{fileName[:length]}.png'
    plt.savefig(saveTitle)



if __name__ == '__main__':
    generateDenodrogram()