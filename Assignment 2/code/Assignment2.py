'''
            Machine Learning Logistic Regression, Support Vector Machine, K-Fold Cross-Validation.
        SENG 474 Data Mining, Spring 2021, University of Victoria.
                Professor: Nishant Mehta
                    Austin Bassett
'''

# Modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import hinge_loss, mean_squared_error
import matplotlib.pyplot as plt
from mnist_reader import load_mnist
import numpy as np
import time

def supportVectorMachines(XTrain, XTest, YTrain, YTest, c0, alpha):
    
    # List of Linear Support Vector Machines.
    potentialModels = []
    regularizationNumbers = []

    # Training potential SVM using a Linear kernal.
    for exponent in range(10):
        
        # Creating a logarithmically spaced grid for the regularization number.
        regularizationNum = c0 * (alpha ** exponent)

        linearModel = SVC(kernel='linear', C=regularizationNum)
        linearModel.fit(XTrain, YTrain)
        potentialModels.append(linearModel)
        regularizationNumbers.append(regularizationNum)

    # Calculating the training and testing error for each model.
    trainingScore = []
    testingScore = []

    for model in potentialModels:
        trainPredict = model.predict(XTrain)
        trainError = hinge_loss(YTrain, trainPredict)
        testPredict = model.predict(XTest)
        testError = hinge_loss(YTest, testPredict)
        trainingScore.append(trainError)
        testingScore.append(testError)

    # Graphing the testing error and training error against the change in regularization number.
    # This code was a modification from the following source locaiton.
    # Source: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
    fig, ax = plt.subplots()
    ax.set_xlabel("Regularization Number")
    ax.set_ylabel("Error")
    ax.set_title("Error vs Regularization Number for Training and Testing sets")
    ax.plot(regularizationNumbers, trainingScore, marker='o', label="train", drawstyle="steps-post")
    ax.plot(regularizationNumbers, testingScore, marker='o', label="test", drawstyle="steps-post")
    ax.legend()

    # Save image.
    errorGraphTitle = "Unoptimized SVM Linear Kernal.png"
    saveImage = "Unoptimized Results/" + errorGraphTitle
    fig.savefig(saveImage)

    # Print best Regularization numbers
    bestTrainingIndex = trainingScore.index(min(trainingScore))
    bestTraining = trainingScore[bestTrainingIndex]
    trainingRegular = regularizationNumbers[bestTrainingIndex]

    bestTestIndex = testingScore.index(min(testingScore))
    bestTest = testingScore[bestTestIndex]
    testingRegular = regularizationNumbers[bestTestIndex]

    print('Unoptimized Linear SVM Results')
    print(f'Lowest Training Error: {bestTraining}')
    print(f'With Regularization: {trainingRegular}')

    print(f'Lowest Testing Error: {bestTest}')
    print(f'With Regularization: {testingRegular}')


def logistic(XTrain, XTest, YTrain, YTest, c0, alpha):

    # List of potential models and regularization numbers.
    potentialModels = []
    regularizationNumbers = []

    # Training models on different regularization numbers.
    for exponent in range(10):

        # Creating a logarithmically spaced grid for the regularization number.
        regularizationNum = c0 * (alpha ** exponent)

        linearModel = LogisticRegression(penalty='l2', C=regularizationNum)
        linearModel.fit(XTrain, YTrain)
        potentialModels.append(linearModel)
        regularizationNumbers.append(regularizationNum)

    # Calculating the training and testing error for each model.
    trainingScore = []
    testingScore = []

    for model in potentialModels:
        trainPredict = model.predict(XTrain)
        trainError = mean_squared_error(YTrain, trainPredict)
        testPredict = model.predict(XTest)
        testError = mean_squared_error(YTest, testPredict)
        trainingScore.append(trainError)
        testingScore.append(testError)
        
    # Graphing the testing error and training error against the change in regularization number.
    # This code was a modification from the following source locaiton.
    # Source: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
    fig, ax = plt.subplots()
    ax.set_xlabel("Regularization Number")
    ax.set_ylabel("Error")
    ax.set_title("Error vs Regularization Number for Training and Testing sets")
    ax.plot(regularizationNumbers, trainingScore, marker='o', label="train", drawstyle="steps-post")
    ax.plot(regularizationNumbers, testingScore, marker='o', label="test", drawstyle="steps-post")
    ax.legend()

    # Save image.
    errorGraphTitle = "Unoptimized Logistic Regression.png"
    saveImage = "Unoptimized Results/" + errorGraphTitle
    fig.savefig(saveImage)

    # Print best Regularization numbers
    bestTrainingIndex = trainingScore.index(min(trainingScore))
    bestTraining = trainingScore[bestTrainingIndex]
    trainingRegular = regularizationNumbers[bestTrainingIndex]

    bestTestIndex = testingScore.index(min(testingScore))
    bestTest = testingScore[bestTestIndex]
    testingRegular = regularizationNumbers[bestTestIndex]

    print('Unoptimized Logistic Regression Results')
    print(f'Lowest Training Error: {bestTraining}')
    print(f'With Regularization: {trainingRegular}')

    print(f'Lowest Testing Error: {bestTest}')
    print(f'With Regularization: {testingRegular}')

def gaussianSVM(XTrainData, XTestData, YTrainData, YTestData, c0, alpha, kFoldVal):

    # Lists for storing gamma and the best regularization exponent.
    gammaValue = []
    bestExponent = []

    betaTracker = 1

    # Finding the best regularization number for each gamma.
    for beta in np.logspace(2.3, 3.4, num=10):

        print(f'Beta iteration: {betaTracker}')
        # Storing beta value.
        betaValue = 1/beta
        gammaValue.append(betaValue)

        # List for storing the potiential regularization exponents.
        potentialExponent = []

        # Find the best regularization exponents.
        for iteration in range(10):

            # Calculating a regularization number.
            regularizationNumber = c0 * (alpha ** iteration)
            result = kFold(XTrainData, YTrainData, kFoldVal, 'gaussian', regularizationNumber, gamma=betaValue)
            potentialExponent.append(result)

        # Extract the best exponent
        best = potentialExponent.index(max(potentialExponent))
        bestExponent.append(best)

        betaTracker = betaTracker + 1

    # Lists for storing training and test scores.
    trainingScore = []
    testingScore = []

    # Train and test all the models using the optimized parameters.
    for modelNumber in range(10):

        # Getting the best values.
        exponent = bestExponent[modelNumber]
        gamma = gammaValue[modelNumber]

        # Calculate regulatization number.
        regularizationNumber = c0 * (alpha ** exponent)

        # Train the model.
        model = SVC(kernel='rbf', C=regularizationNumber, gamma=gamma)
        model.fit(XTrainData, YTrainData)

        # Store the scores.
        trainPredict = model.predict(XTrainData)
        trainError = hinge_loss(YTrainData, trainPredict)
        testPredict = model.predict(XTestData)
        testError = hinge_loss(YTestData, testPredict)
        trainingScore.append(trainError)
        testingScore.append(testError)

    # Plot the data and store the image.
    # This code was a modification from the following source locaiton.
    # Source: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
    fig, ax = plt.subplots()
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Error")
    ax.set_title("Error vs Gamma for Training and Testing sets")
    ax.plot(gammaValue, trainingScore, marker='o', label="train", drawstyle="steps-post")
    ax.plot(gammaValue, testingScore, marker='o', label="test", drawstyle="steps-post")
    ax.legend()

    # Save image.
    errorGraphTitle = "Gaussian.png"
    saveImage = "Optimized Results/" + errorGraphTitle
    fig.savefig(saveImage)

    # Print best Regularization numbers
    bestTrainingIndex = trainingScore.index(min(trainingScore))
    bestTraining = trainingScore[bestTrainingIndex]
    trainingRegular = c0 * (alpha ** bestExponent[bestTrainingIndex])
    bestTrainGamma = gammaValue[bestTrainingIndex]

    bestTestIndex = testingScore.index(min(testingScore))
    bestTest = testingScore[bestTestIndex]
    testingRegular = c0 * (alpha ** bestExponent[bestTestIndex])
    bestTestGamma = gammaValue[bestTestIndex]

    print('Optimized Gaussian SVM Results')
    print(f'Lowest Training Error: {bestTraining}')
    print(f'With Regularization: {trainingRegular}, Gamma: {bestTrainGamma}')

    print(f'Lowest Testing Error: {bestTest}')
    print(f'With Regularization: {testingRegular}, Gamma: {bestTestGamma}')


def optimizedModels(XTrainData, XTestData, YTrainData, YTestData, modelType, c0, alpha, kFoldVal):
    
    # List of potential exponents for the regularization number.
    potentialExponents = []

    # Find the optimal exponent for the regularization.
    for exponent in range(10):
        regularizationNumber = c0 * (alpha ** exponent)
        potentialExponents.append(kFold(XTrainData, YTrainData, kFoldVal, modelType, regularizationNumber))
    
    # Extract the best exponent for regularization number.
    bestExponent = potentialExponents.index(max(potentialExponents))
    optimizedRegularizationNumber = c0 * (alpha ** bestExponent)

    # Train and test the respectecive model.
    if modelType == 'SVM':

        model = SVC(kernel='linear', C=optimizedRegularizationNumber, random_state=0)
        model.fit(XTrainData, YTrainData)

        # Calculating the training and testing error
        trainPredict = model.predict(XTrainData)
        trainingError = hinge_loss(YTrainData, trainPredict)
        testPredict = model.predict(XTestData)
        testingError = hinge_loss(YTestData, testPredict)

    elif modelType == 'logistic':

        model = LogisticRegression(penalty='l2', C=optimizedRegularizationNumber, random_state=0)
        model.fit(XTrainData, YTrainData)
        trainPredict = model.predict(XTrainData)
        trainingError = mean_squared_error(YTrainData, trainPredict)
        testPredict = model.predict(XTestData)
        testingError = mean_squared_error(YTestData, testPredict)

    # Output results to the command line
    print(f'The Optimized error for {modelType}')
    print(f'Training: {trainingError}')
    print(f'Testing: {testingError}')
    print(f'With Regularization: {optimizedRegularizationNumber}')

def kFold(xTrainData, yTrainData, kFold, modelType, regularizationNumber, gamma=None):
    
    # Split the training & testing list into subarrays
    xData = np.split(np.array(xTrainData), kFold)
    yData = np.split(np.array(yTrainData), kFold)

    # List for storing the scores.
    scores = []

    # Iterate over the grid of data.
    for rowNumber in range(kFold):
        # List to be reset with each iteration.
        XTrain = []
        YTrain = []

        for columnNumber in range(kFold):
            # If columnNumber == rowNumber, go to next iteration.
            if columnNumber == rowNumber:
                # This index number will be used for the testing set.
                continue

            # Append data to the training lists.
            XTrain.append(xData[columnNumber])
            YTrain.append(yData[columnNumber])

        XTrain = list([item for items in XTrain for item in items])
        YTrain = list([item for items in YTrain for item in items])

        XTest = list(xData[rowNumber])
        YTest = list(yData[rowNumber])

        # Train and test the respectecive model.
        if modelType == 'SVM':

            model = SVC(kernel='linear', C=regularizationNumber)
            model.fit(XTrain, YTrain)

        elif modelType == 'logistic':
    
            model = LogisticRegression(penalty='l2', C=regularizationNumber)
            model.fit(XTrain, YTrain)
        
        else:

            # Support Vector Machine using the Gaussian Kernal.
            model = SVC(kernel='rbf', C=regularizationNumber, gamma=gamma)
            model.fit(XTrain, YTrain)


        # Record the scores (Accuracy).
        results = model.score(XTest, YTest)
        scores.append(results)

    # Return the average of the scores.
    average = sum(scores) / len(scores)

    return average

def scaleData(XData, YData):

    # Variables
    size = len(XData)
    
    # Lists for scaled data
    XScaled = []
    YScaled = []

    # Loop through all the items scale them.
    for iteration in range(size):

        # Check if the label is in class 5 or 7
        if YData[iteration] in [5, 7]:
            value = XData[iteration] / 255
            XScaled.append(value)

            # Assigning 0 to class 5 and 1 for class 7
            if YData[iteration] == 5:
                YScaled.append(0)

            else:
                YScaled.append(1)

    return XScaled, YScaled


def main():

    # Get start time.
    startTime = time.time()
    
    # Read in the data.
    XTrain, YTrain = load_mnist('data/fashion', kind='train')
    XTest, YTest = load_mnist('data/fashion', kind='t10k')

    # Convert the set data to lists.
    XTrain = list(XTrain)
    XTest = list(XTest)

    YTrain = list(YTrain)
    YTest = list(YTest)

    # Scale the data, as we only want the sneaker and sandals
    XTrain, YTrain = scaleData(XTrain, YTrain)
    XTest, YTest = scaleData(XTest, YTest)

    # Reducing the data set for the SVM for reducing training time.
    reducedDataPoint = int(len(XTrain) / 2)
    XTrain = XTrain[:reducedDataPoint]
    YTrain = YTrain[:reducedDataPoint]

    # Unotimized Models.
    logistic(XTrain, XTest, YTrain, YTest, 0.05, 1.73)
    supportVectorMachines(XTrain, XTest, YTrain, YTest, 0.002, 3.24)

    # Optimized Models.
    optimizedModels(XTrain, XTest, YTrain, YTest, 'logistic', 0.05, 1.75, 8)
    optimizedModels(XTrain, XTest, YTrain, YTest, 'SVM', 0.002, 3.24, 8)

    # Training and Testing the SVM Gaussian Model.
    gaussianSVM(XTrain, XTest, YTrain, YTest, 0.002, 3.24, 8)

    # Total Execution Time
    totalTime = time.time() - startTime
    print(f'Execution Time: {totalTime} seconds')

if __name__ == '__main__':
    main()