import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text

def make_dataset(n, d = 4, p = 0):
    """
    Create a dataset with boolean features and a binary class label.
    The label is assigned as x1 ^ x2 V x3 ^ x4.
    
    Arguments:
      n - The number of instances to generate
      m - The number of features per instance.  Any features beyond the first four
          are irrelevant to determining the class label.
      p - The probability that the true class label as computed by the expression
          above is flipped.  Said differently, this is the probability of class noise.
    """
    
    assert d >= 4, 'The dataset must have at least 4 features'
    X = [np.random.randint(2, size = d) for _ in range(n)]
    y = [(x[0] and x[1]) or (x[2] and x[3]) for x in X]
    y = [v if random.random() >= p else (v + 1) % 2 for v in y]
    return X, y

"""
When evaluating the accuracy of a classifier, the right way to do it is to have a test set of instances that were not used 
to train the classifier and measure on those instances.  

The train_test_split() function in scikit makes it easy to create training and testing sets.  
"""

# Exploring impacts on overfitting

# Size of Dataset
test_acc = []
train_acc = []
diffSizes = [10,100,500,1000,5000]

# Each iteration we plot a point for differing sizes and their respective accuracies
for f in diffSizes:
    X, y = make_dataset(f, d = 10, p = 0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, clf.predict(X_test)))
    
plt.plot(diffSizes, train_acc, label = 'train')
plt.plot(diffSizes, test_acc, label = 'test')
plt.legend()


# Number of Irrelevant Features
test_acc = []
train_acc = []

numIrrelevant = [1,100,500,1000,5000] # must be greater than 0

# Each iteration we plot a point for differing sizes and their respective accuracies
for f in diffSizes:
    X, y = make_dataset(1000, d = f, p = 0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, clf.predict(X_test)))
    
plt.plot(numIrrelevant, train_acc, label = 'train')
plt.plot(numIrrelevant, test_acc, label = 'test')
plt.legend()

# Probability of Class Noise
test_acc = []
train_acc = []

noiseProb = [0,0.3,0.5,0.80,1]

# Each iteration we plot a point for differing sizes and their respective accuracies
for f in noiseProb:
    X, y = make_dataset(1000, d = 10, p = f)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, clf.predict(X_test)))
    
plt.plot(noiseProb, train_acc, label = 'train')
plt.plot(noiseProb, test_acc, label = 'test')
plt.legend()

# Min number of samples required for a node to be split
test_acc = []
train_acc = []

minSampleSplit = [2,4,10,100,500,1000] #value must be >=2

# Each iteration we plot a point for differing sizes and their respective accuracies
for f in minSampleSplit:
    X, y = make_dataset(1000, d = 10, p = 0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    clf = tree.DecisionTreeClassifier(min_samples_split=f)
    clf.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, clf.predict(X_test)))
    
plt.plot(minSampleSplit, train_acc, label = 'train')
plt.plot(minSampleSplit, test_acc, label = 'test')
plt.legend()