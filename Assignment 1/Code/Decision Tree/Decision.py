'''
        Machine Learning Decsion Tree with pruning.
        SENG 474 Data Mining, Spring 2021, University of Victoria.
                Professor: Nishant Mehta
                    Austin Bassett
'''
# Modules
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from subprocess import check_call
import matplotlib.pyplot as plt
import pandas as pd
import sys

def treePrunning(dataSet: str, labelIndex: int, splitPercent: float, outputName: str, passedCritererion: str):

        # Reading in the data.
        data = pd.read_csv(dataSet, header=None)

        # Breaking the data in Features (X) and Label (y).
        X = data.drop(labelIndex, axis=1)
        y = data[labelIndex]

        # Split the X and y into training and testing segments.
        XTrain, XTest, yTrain, yTest = train_test_split(X, y, random_state=0, train_size=splitPercent)

        # Initalizing the Decision Tree.
        model = DecisionTreeClassifier(criterion=passedCritererion,random_state=0)

        # Minimal cost complexity pruning recursively finds the node with the “weakest link”. 
        # The weakest link is characterized by an effective alpha, where the nodes with the smallest effective alpha are pruned first.
        # Taken from Sklearn Decision Tree Site (Source: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)
        path = model.cost_complexity_pruning_path(XTrain, yTrain)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities

        # List of different trees.
        potentialTrees = []

        # Training trees using effective alphas.
        for ccp_alpha in ccp_alphas:
                newTree = DecisionTreeClassifier(criterion=passedCritererion,random_state=0, ccp_alpha=ccp_alpha)
                newTree.fit(XTrain, yTrain)
                potentialTrees.append(newTree)

        # Remove the las element in potentialTrees and ccp_alphas as they are trivial tree wiht only one node.
        potentialTrees = potentialTrees[:-1]
        ccp_alphas = ccp_alphas[:-1]

        # Graphing the relation between number of nodes and depth of tree and alpha
        # Source: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
        node_counts = [clf.tree_.node_count for clf in potentialTrees]
        depth = [clf.tree_.max_depth for clf in potentialTrees]
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
        ax[0].set_xlabel("Alpha")
        ax[0].set_ylabel("Number of Nodes")
        ax[0].set_title("Number of Nodes vs Alpha")
        ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
        ax[1].set_xlabel("Alpha")
        ax[1].set_ylabel("Depth of Tree")
        ax[1].set_title("Depth vs Alpha")
        fig.tight_layout()

        nodeDepthGraphTitle = outputName + passedCritererion.capitalize() + " Depth Node Vs Alpha.png"
        saveImage = "Images/" + nodeDepthGraphTitle
        fig.savefig(saveImage)

        # Getting the test scores
        trainScores = [tree.score(XTrain, yTrain) for tree in potentialTrees]
        testScores = [tree.score(XTest, yTest) for tree in potentialTrees]

        # Graphing the relation between accuracy and alpha.
        # Source: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
        fig, ax = plt.subplots()
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Alpha for Training and Testing sets")
        ax.plot(ccp_alphas, trainScores, marker='o', label="train",
                drawstyle="steps-post")
        ax.plot(ccp_alphas, testScores, marker='o', label="test",
                drawstyle="steps-post")
        ax.legend()

        accuracyGraphTitle = outputName + passedCritererion.capitalize() + "Accuracy Vs Alpha.png"
        saveImage = "Images/" + accuracyGraphTitle
        fig.savefig(saveImage)

        # Getting the index location of the best tree.
        bestTree = testScores.index(max(testScores))

        # Plotting the best tree.
        figBest = plt.figure(figsize=(25, 20))
        plottedBest = plot_tree(potentialTrees[bestTree], feature_names=X.columns)

        # Saving the plotted best tree.
        bestTreeTitle = outputName + "Best " + passedCritererion.capitalize() +" Tree.png"
        saveImage = "Images/" + bestTreeTitle
        figBest.savefig(saveImage)

        # Output the best test accuracy and the depth of the decision tree.
        bestAccuracy = testScores[bestTree]
        bestDepth = depth[bestTree]

        print(f"The best Decision Tree ({passedCritererion}):")
        print(f"Depth: {bestDepth}")
        print(f"Accuracy: {bestAccuracy}\n")

def main():
        
        # Getting the passed in arguments.
        csvFile = sys.argv[1]
        labelIndex = int(sys.argv[2])
        splitPercent = float(sys.argv[3])
        outputName = sys.argv[4]
        
        # Changing -0.0 to 0.0
        if splitPercent == -0.0:
                splitPercent = splitPercent * -1
        
        # Check if split is between 0.0 - 1.0
        if not (0.0 <= splitPercent <= 1.0):
                print("Error: Split percent is out of range")
                exit(0)

        # Getting the best tree using both gini and entropy criterion.
        treePrunning(csvFile, labelIndex, splitPercent, outputName, "gini")
        treePrunning(csvFile, labelIndex, splitPercent, outputName, "entropy")
        

if __name__ == '__main__':
        main()