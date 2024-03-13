# Made by:
# Roei Yanku - 207014440
# Please Plug in your directory in line 12!!
# This code uses the KNN algorithm to classify patients as
# either having heart disease or not

import pandas as pd
import numpy as np
from collections import Counter

# Load the data from Excel into a pandas DataFrame
df = pd.read_csv(r'Please-Provide-Directory-Here\heart_statlog_cleveland_hungary_final.csv')

# Split the data into features (X) and target (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the data into training and test sets (you can use other methods for this as well)
train_size = int(0.8 * len(df))  # Use 80% of the data for training
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the k-NN algorithm
def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for test_row in X_test.values:
        distances = np.sqrt(np.sum((X_train.values - test_row)**2, axis=1))
        nearest_neighbors = y_train.iloc[np.argsort(distances)[:k]]
        most_common = Counter(nearest_neighbors).most_common(1)
        predictions.append(most_common[0][0])
    return predictions

# Make predictions using k=1 (you can choose a different k)
k = 1

y_pred = knn_predict(X_train, y_train, X_test, k)

y_train_pred = knn_predict(X_train, y_train, X_train, k)
y_test_pred = knn_predict(X_train, y_train, X_test, k)

# Evaluate the model on the training set
accuracy_train = sum(y_train_pred == y_train) / len(y_train)
print(f'Accuracy on train set: {accuracy_train}')

# Evaluate the model on the test set
accuracy_test = sum(y_test_pred == y_test) / len(y_test)
print(f'Accuracy on test set: {accuracy_test}')

# Define a function to calculate precision, recall, and F-measure
def calculate_metrics(y_true, y_pred):
    TP = sum((y_true == 1) & (y_pred == 1))
    TN = sum((y_true == 0) & (y_pred == 0))
    FP = sum((y_true == 0) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))

    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f_measure = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return precision, recall, f_measure

# Calculate precision, recall, and F-measure on the test set
precision, recall, f_measure = calculate_metrics(y_test.values, np.array(y_test_pred))

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F-measure: {f_measure}')
