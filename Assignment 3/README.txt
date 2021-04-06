Theses programs were developed to run on Python 3. Executing these programs on a different version may result in unexpected behaviour.

ElbowGraphs:
------------
Use the following command to generate elbow graphs for both the Uniform Random Initialization and K-Means++ Initialization.

		python3 ElbowGraphs.py <CSV File> <Cluster Range Max>

Cluster Range Max: Integer for the K value range going from 1 to Cluster Range Max.

The output graphs will be saved in the Elbow Graph Images folder.

Lloyds:
--------
Use the following command to run Lloyds algorithm.

		python3 Lloyds.py <CSV File> <Uniform Cluster Val> <K-Means++ Cluster Val> <Loop Max>

Uniform Cluster Val: Integer for number of clusters to use. A value of 0 or less will skip this method.
K-Means++ Cluster Val: Integer for number of clusters to use. A value of 0 or less will skip this method.
Loop Max: Integer for prevent Lloyd's algorithm from entering an infinite loop in the case of divergence.

The output will be two graphs, one 2D and one 3D.
The graphs will be saved in the Lloyd's Images folder.


Hierarchical:
-------------
Use the following command to run the hierarchical agglomerative clustering models.

		python3 Hierarchical.py <CSV File> <Single Clusters> <Average Clusters>

Single Clusters: Integer for the number of clusters to use with single linkage.
Average Clusters: Integer for the number of clusters to use with average linkage.

The output are two clustering graphs, one 2D and one 3D.
The graphs will be found in their respective image folders.


Dendrograms:
-------------
Use the following command to generate two dendrograms, one using single linkage and one using average linkage.

		python3 Dendrograms.py <Data file>

Note: Dendrograms.py uses Pandas read_csv function for extracting the data. Check the Pandas documentation if you are using a different file type.

The dendrograms will be saved in Single Linkage Images & Average Linkage Images.


Attributions:
-------------
Parts of the Hierarchical Agglomerative Clustering functions and the dendrograms were from the following website:
https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/

Sklearn Agglomerative Clustering function:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

3D scatter code used the following stack overflow as reference:
https://stackoverflow.com/questions/1985856/how-to-make-a-3d-scatter-plot-in-python

The following resources were used as references for Lloyd's Algorithm Code (Found in LloydHelpers.py):
Source 1: https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
Source 2: https://gdcoder.com/implementation-of-k-means-from-scratch-in-python-9-lines/
Source 3: https://www.geeksforgeeks.org/ml-k-means-algorithm/

Elbow graph code used the following source as reference:
https://pythonprogramminglanguage.com/kmeans-elbow-method/

Cost function for the Elbow graph used the following source as reference:
https://gist.github.com/adityachandupatla/622d50d1dce80777dce4b2839760777f
