#!/usr/bin/python

import sys
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from emailPreProcess import preprocess

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# Create gaussian naive bayes classifier
clf = GaussianNB()
# Fit the classifier using training data
t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")
# Predict using the test data
t0 = time()
predictions = clf.predict(features_test)
print("prediction time:", round(time()-t0, 3), "s")
# Determine quality of the classifier
print(accuracy_score(labels_test,predictions))