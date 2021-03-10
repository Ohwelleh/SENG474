Austin Bassett
V00905244
SENG 474 Assignment 1
Nishant Mehta
==================

My code was developed to run on Python 3 and the folowing modules will need to be installed for proper execution:
1. Numpy (pip3 install numpy)
2. Pandas (pip3 install pands)
3. Sklearn (pip3 install Sklearn)
4. Matplotlib (pip3 install matplotlib)

Requirements For Input Data
----------------------------
This applies to the Decision Tree, Random Forest, and Neural Network.

1. The dataset must be in CSV format. (NOTE:These programs were tested on file extension .data and .csv, both were formatted CSV. Unexpected behaviour may occur with other types.)

2. The header row must be deleted; the programs add their own indices.

Code Execution.
================

Each program creates an image to visualize the information extracted from the machine learning method. The section after this contains information on what each program outputs.

(All programs were used with the .csv/.data files in the same directory as the program executing them)

Argument Definitions
---------------------
CSVFile (String) = The CSV that has been prepped.
LabelIndex (Integer) = Column number that the label resides in. (The programs start indexing at 0)
Split (Float) = The Percent to which the CSV data will be split for training. (Number between 0.0 - 1.0)
Output (String) = The output name for saving the images.
NumberOfHidden (Integer) = The max number of nodes in the hidden layer of the neural network. This number must be greater than or equal to 3.
MaxTree (Integer) = The maximum number of trees the data set. (Range from 1 to MaxTree of step size of 1)

See below for example execution

To execute the Decision Tree Code on the command line:

	python3 DecisionTree.py <CSVFile> <LabelIndex> <Split> <Output>

To execute the Random Forest Code on the command line:

	python3 RandomForest.py <CSVFile> <LabelIndex> <Split> <MaxTrees> <Output>

To execute the NeuralNet Code on the command line:

	python3 NeuralNet.py <CSVFile> <LabelIndex> <NumberOfHidden> <Split> <Output>

Example: Example.csv
0,    1,   2,   3,     4,     5,   6,   7,     8,   9,   10,  11,  12,  13(This row will be added by the programs.)
63.0, 1.0, 1.0, 145.0, 233.0, 1.0, 2.0, 150.0, 0.0, 2.3, 3.0, 0.0, 6.0, 0

	python3 DecisionTree.py Example.csv 13 0.25 exampleOutput


Outputs
========

All image outputs are sent to the Images folder.

DecisionTree.py
----------------
(Outputs 2 of each, one using Gini criterion and one using Entropy criterion)
(Alpha was the pruning variable, as Alpha increases, more nodes are pruned)
- 3 Graphs:
	- Number of Nodes vs Alpha
	- Depth of Tree vs Alpha
	- Accuracy vs Alpha

- Image of the best decision tree (Highest test accuracy)
- Output to the command line is the Depth and Accuracy values of this best decision tree.

RandomForest.py
---------------
- Graph of Accuracy vs Number of Trees in the Forest
- Output to the command line is the Number of Trees and Accuracy values of the best (Highest test accuracy) random forest.

NeuralNetwork.py
----------------
- Graph of Accuracy vs Size of Hidden Layer.
- Output to the command line is the Number of Nodes in the Hidden Layer and Accuracy values of the best (Highest test accuracy) neural network.


Attributions
================
- Refering to Lab 1 and 2 .ipynb files for examples and logic on Sklearn(Decision Tree), Numpy, Pandas.
- Nishant for providing the clean dataset.

- For the code of implementing a decision tree: https://benalexkeen.com/decision-tree-classifier-in-python-using-scikit-learn/

- For the pruning tree code: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

- Parts of the Random Forest code was from the Sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

- Parts of the Neural Network code was from the Sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html