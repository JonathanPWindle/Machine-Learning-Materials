#!/usr/bin/python

import sys
import numpy as np
from time import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from emailPreProcess import preprocess

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# Splice down the training data to 1% of all data
# features_train = features_train[:round(len(features_train)/100)]
# labels_train = labels_train[:round(len(labels_train)/100)]

clf = SVC(C=10000.0 ,kernel="rbf")

clf.fit(features_train, labels_train)

predictions = clf.predict(features_test)

print(accuracy_score(labels_test,predictions))
unique, counts = np.unique(predictions, return_counts=True)

print(dict(zip(unique, counts)))