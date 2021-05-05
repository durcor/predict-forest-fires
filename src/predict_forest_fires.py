#!/usr/bin/python3
from pandas import read_csv
from sklearn.neural_network import MLPClassifier

# Goal:
# - predict values in the 1st or 2nd column (fire severity classes) from
# other columns that contain remote sensing parameters for the different
# forest patches (every row - 1 forest patch).

# Vocabulary:
# severity: Forest Fire Severity
# BA: Basal Area Loss
# Range from 1 - 9
data = read_csv("subset_1percent.csv")
severity_classes = data['Severity_classes']
ba_classes = data['BA_classes']
print(severity_classes)
print(ba_classes)

# Predict Severity classes
clf = MLPClassifier()
severity_neural_net = clf.fit(data, severity_classes)

# Predict BA classes
ba_neural_net = clf.fit(data, ba_classes)
