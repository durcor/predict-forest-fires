#!/usr/bin/env python
from pandas import read_csv
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Goal:
# - predict values in the 1st or 2nd column (fire severity classes) from
# other columns that contain remote sensing parameters for the different
# forest patches (every row - 1 forest patch).

# Vocabulary:
# severity: Forest Fire Severity
# BA: Basal Area Loss
# Range from 1 - 9
small_data = read_csv("../data/subset_1percent.csv")
medium_data = read_csv("../data/subset_5percent.csv")
big_data = read_csv("../data/subset_10percent.csv")

X = small_data.iloc[:, 2:]

def predict(y, class_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    nn = MLPClassifier(hidden_layer_sizes=(5,), max_iter=10000, learning_rate='adaptive')
    nn.fit(X_train, y_train)

    y_pred = nn.predict(X_test)
    print(class_type)
    print("Tested sample:   ", y_test.tolist()[:40])
    print("Predicted sample:", y_pred.tolist()[:40])
    print("Accuracy of predictions:", accuracy_score(y_test, y_pred))
    print()

predict(small_data["Severity_classes"], "Severity Classes")
predict(small_data["BA_classes"], "BA Classes")
