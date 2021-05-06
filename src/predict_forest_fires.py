#!/usr/bin/env python
from pandas import read_csv
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def broad_accuracy_score(y_true, y_pred):
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()
    sum = 0
    for i in range(len(y_true)):
        if y_pred[i] in [y_true[i], y_true[i] + 1, y_true[i] - 1]:
            sum += 1
    return sum / len(y_true)


# Vocabulary:
# Severity: Forest Fire Severity
# BA: Basal Area Loss
# Range from 1 - 9

data = read_csv("../data/subset_1percent.csv")
# data = read_csv("../data/subset_5percent.csv")
# data = read_csv("../data/subset_10percent.csv")

# Read all recorded parameters into the variable to be trained/tested.
X = data.iloc[:, 2:]


# Predict BA and Severity classes.
def predict(y, class_type):
    # Partition samples to train and test.
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # NOTE: Tweak the parameters of this to improve prediction.
    nn = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10000)
    # Train the neural network.
    nn.fit(X_train, y_train)

    # Predict the class of the tested sample.
    y_pred = nn.predict(X_test)

    print(class_type)
    # Print out samples for user to check for differences and overall accuracy
    # of prediction.
    print("Tested sample:   ", y_test.tolist()[:40])
    print("Predicted sample:", y_pred.tolist()[:40])
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    broad_accuracy = round(broad_accuracy_score(y_test, y_pred) * 100, 2)
    print("Exact accuracy of predictions: " + str(accuracy) + '%')
    print("Broad accuracy of predictions: " + str(broad_accuracy) + '%')
    # TODO: Find a more broad accuracy measurement by allowing for a margin of
    # error of +/-1 for each predicted class.
    print()


predict(data["Severity_classes"], "Severity Classes")
predict(data["BA_classes"], "BA Classes")
