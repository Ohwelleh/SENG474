'''
           Machine Learning Random Forest (No Pruning).
        SENG 474 Data Mining, Spring 2021, University of Victoria.
                Professor: Nishant Mehta
                    Austin Bassett
'''

# Modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sys

def randoForest(dataSet: str, labelIndex: int, splitPercent: float, maxTrees: int, outputName: str, passedCritererion: str):

        # Reading in the data.
        data = pd.read_csv(dataSet, header=None)

        # Breaking the data in Features (X) and Label (y).
        X = data.drop(labelIndex, axis=1)
        y = data[labelIndex]

        # Split the X and y into training and testing segments.
        XTrain, XTest, yTrain, yTest = train_test_split(X, y, random_state=0, train_size=splitPercent)

        # Creating a list of different number of trees per forest.
        numberOfTrees = list(range(1, maxTrees, 1))

        # List to contain all the random forest.
        potentialRandomForest = []

        # Training a list of different sized random forests.
        for trees in numberOfTrees:
                rf = RandomForestClassifier(n_estimators=trees, criterion=passedCritererion, max_features="sqrt",random_state=0)
                rf.fit(XTrain, yTrain)
                potentialRandomForest.append(rf)

        # Graphing the nodes in relation to accuracy.
        trainScores = [tree.score(XTrain, yTrain) for tree in potentialRandomForest]
        testScores = [tree.score(XTest, yTest) for tree in potentialRandomForest]

        # This code was modified from my DecisionTree.py.
        # Original source: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
        # Graphing the accuracy of the training and testing sets with respect the the number of trees in the forest.
        fig, ax = plt.subplots()
        ax.set_xlabel("Number of Trees in the Forest.")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Number of Trees for Training and Testing sets")
        ax.plot(numberOfTrees, trainScores, marker='o', label="train")
        ax.plot(numberOfTrees, testScores, marker='o', label="test")
        ax.legend()

        # Saving the image.
        accuracyVsTreesTitle = outputName + passedCritererion.capitalize() + " Accuracy vs Number of Trees.png"
        saveImage = "Images/" + accuracyVsTreesTitle
        fig.savefig(saveImage)

        # Output the best test accuracy and the number of trees in the forest.
        bestIndex = testScores.index(max(testScores))
        bestAccuracy = testScores[bestIndex]
        bestNumberOfTrees = numberOfTrees[bestIndex]

        print(f"The best Random Forest ({passedCritererion}):")
        print(f"Number of Trees: {bestNumberOfTrees}")
        print(f"Accuracy: {bestAccuracy}\n")


def main():
        
        # Getting the passed in arguments.
        csvFile = sys.argv[1]
        labelIndex = int(sys.argv[2])
        splitPercent = float(sys.argv[3])
        maxTreePerForest = int(sys.argv[4])
        outputName = sys.argv[5]
        
        # Changing -0.0 to 0.0
        if splitPercent == -0.0:
                splitPercent = splitPercent * -1
        
        # Check if split is between 0.0 - 1.0
        if not (0.0 <= splitPercent <= 1.0):
                print("Error: Split percent is out of range")
                exit(0)

        # Getting the best tree using both gini and entropy criterion.
        randoForest(csvFile, labelIndex, splitPercent, maxTreePerForest, outputName, "gini")
        randoForest(csvFile, labelIndex, splitPercent, maxTreePerForest, outputName, "entropy")
        

if __name__ == '__main__':
        main()