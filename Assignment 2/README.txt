This program will run a logistic regression, linear support vector machine, and a gaussian support vector machine on the Fashion-MNIST dataset, but only using the Sandal and Sneakers sections (Class 5 and 7 respectively). Additionally, the program finds optimized regularization values for each machine learning model and runs the data against these new models.

***********
Note: Due to the maximum file submission size on connext being 50MB I could not keep the entire dataset in the folder, instead only the folder and file I needed are there. Here is a link to the entire dataset Github repo.

Dataset Source: https://github.com/zalandoresearch/fashion-mnist

This link was provided by Nishant in the assignment description for the purpose of this assignment.

**********


This code was developed to run on Python 3 with the following libraries needed:

	-Sklearn
	-Numpy
	-Matplotlib


Use the following command in the command line to execute my program:

	python3 Assignment2.py

Use the following command to calculate a 95% confidence interval:

	python3 Confidence.py <Error Value><Size of testing data>

Error Value: Is a number in decimal form. (E.G. 20% Error would be entered as 0.20)
Size of testing data: Integer (E.G. testing set has 20 entries, so 20 would be entered here).

Output of Assignment2.py
--------------------------
Unoptimized Logistic Regression:
A graph of Errors Vs. Regularization Numbers (Unoptimized Results Folder)
The best training and testing error with their regularization value output to the command line.

Unoptimized Linear Support Vector Machine:
A graph of Errors Vs. Regularization Numbers (Unoptimized Results Folder)
The best training and testing error with their regularization value output to the command line.

Optimized Logistic Regression:
The best training and testing error with their regularization value output to the command line.

Optimized Linear Support Vector Machine:
The best training and testing error with their regularization value output to the command line.

Gaussian Support Vector Machine:
A graph of Errors Vs. Gamma Values. (Optimized Results Folder)
The best training and testing error with their regularization value and gamma values output to the command line.
-------------------------------

This calculations uses the following formula

	Z * sqrt( X / N )

	Z = 1.96 for 95% confidence interval
	X = Error Value * ( 1 - Error Value )
	N = Size of testing data

The resulting interval is output.

Example Execution:

Input:
python3 Confidence.py 0.038 2000

Output:
Confidence Interval: 0.008379544426757339


Attribution
--------------
My confidence calculation code was taken from this site:
https://machinelearningmastery.com/confidence-intervals-for-machine-learning/

Nishant for providing the dataset from:
Dataset Source: https://github.com/zalandoresearch/fashion-mnist

Sklearn Logistic Regression Model:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

Sklearn SVM Model:
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

Graphing Code was modified from this source:
https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
