'''
            Machine Learning Neural Network.
    SENG 474 Data Mining, Spring 2021, University of Victoria.
                Professor: Nishant Mehta
                    Austin Bassett
'''

# Modules
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sys
import pandas as pd

def neuralNetwork(dataSet: str, labelIndex: int, numHidden: int, splitPercent: float, outputName: str, passedSolver: str):

    # Reading in the data.
    data = pd.read_csv(dataSet, header=None)

    # Breaking the data in Features (X) and Label (y).
    X = data.drop(labelIndex, axis=1)
    y = data[labelIndex]

    # Split the X and y into training and testing segments.
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, random_state=0, train_size=splitPercent)

    # Creating a range of different hidden layers
    diffentHidden = list(range(3, numHidden, 2))

    # List to contain all the nerual networks.
    potentialNeuralNetworks = []

    # Training all the neural networks.
    for hidden in diffentHidden:
        network = MLPClassifier(hidden_layer_sizes=hidden, solver=passedSolver, max_iter=300 ,random_state=0)
        network.fit(XTrain, yTrain)
        potentialNeuralNetworks.append(network)

    # Graphing the nodes in relation to accuracy.
    trainScores = [net.score(XTrain, yTrain) for net in potentialNeuralNetworks]
    testScores = [net.score(XTest, yTest) for net in potentialNeuralNetworks]

    # This code was modified from my DecisionTree.py.
    # Original source: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
    fig, ax = plt.subplots()
    ax.set_xlabel("Number of Nodes in Hidden Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Size of Hidden Layer for Training and Testing sets")
    ax.plot(diffentHidden, trainScores, marker='o', label="train")
    ax.plot(diffentHidden, testScores, marker='o', label="test")
    ax.legend()

    # Saving the image.
    accuracyVsNodeTitle = outputName + passedSolver.capitalize() + " Hidden Node Vs Accuracy.png"
    saveImage = "Images/" + accuracyVsNodeTitle
    fig.savefig(saveImage)

    # Output the best test accuracy and the number of nodes in the hidden layer.
    bestIndex = testScores.index(max(testScores))
    bestAccuracy = testScores[bestIndex]
    bestNumberOfHiddenNodes = diffentHidden[bestIndex]

    print(f"The best Neural Network ({passedSolver}):")
    print(f"Number of Nodes in Hidden Layer: {bestNumberOfHiddenNodes}")
    print(f"Accuracy: {bestAccuracy}\n")


def main():
        
    # Getting the passed in arguments.
    csvFile = sys.argv[1]
    labelIndex = int(sys.argv[2])
    numberHidden = int(sys.argv[3])
    splitPercent = float(sys.argv[4])
    outputName = sys.argv[5]

    # Check if numberHidden is greater than or equal to 3
    if not numberHidden >= 3:
        print("Error: Number of hidden layers needs to be at least 3.")
        exit(0)

    # Changing -0.0 to 0.0
    if splitPercent == -0.0:
        splitPercent = splitPercent * -1
        
    # Check if split is between 0.0 - 1.0
        if not (0.0 <= splitPercent <= 1.0):
            print("Error: Split percent is out of range")
            exit(0)

    # Getting the best tree using both gini and entropy criterion.
    neuralNetwork(csvFile, labelIndex, numberHidden, splitPercent, outputName, "sgd")
    neuralNetwork(csvFile, labelIndex, numberHidden, splitPercent, outputName, "adam")
        

if __name__ == '__main__':
        main()