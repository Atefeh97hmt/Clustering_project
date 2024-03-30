import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_distance(first_data, second_data):
    n_first_data = first_data.shape[0]
    n_second_data = second_data.shape[0]
    distances = np.zeros([n_first_data, n_second_data], dtype=float)
    for i in range(n_first_data):
        for j in range(n_second_data):
            distances[i, j] = np.sqrt(np.sum((first_data[i, :] - second_data[j, :]) ** 2))
    return distances


file = pd.read_excel('LatLong.xlsx')
data = file.values[:, 2:].astype(float)
u, label = np.unique(file.values[:, 1], return_inverse=True)
n_samples, n_features = data.shape
classes = np.unique(label)
n_classes = len(classes)

# Feature selection using FDR
FDR = np.zeros(n_features)
for feature in range(n_features):
    for i in range(n_classes):
        ui = np.mean(data[label == classes[i], feature])
        vi = np.var(data[label == classes[i], feature])
        for j in range(i + 1, n_classes):
            uj = np.mean(data[label == classes[j], feature])
            vj = np.var(data[label == classes[j], feature])
            FDR[feature] += (ui - uj) ** 2 / (vi ** 2 + vj ** 2)

selected_features = FDR.argsort()[-5:]
data = data[:, selected_features]

# K-means clustering
K = 2
max_iterations = 100
initial_centers = data[np.random.choice(n_samples, K, replace=False)]
centers = np.copy(initial_centers)
for iteration in range(max_iterations):
    distances = calculate_distance(data, centers)
    cluster_assignments = np.argmin(distances, axis=1)
    for j in range(K):
        centers[j] = np.mean(data[cluster_assignments == j], axis=0)

# Calculate accuracy
TP = 0
FP = 0
FN = 0
TN = 0
for i in range(n_samples):
    for j in range(i + 1, n_samples):
        if cluster_assignments[i] == cluster_assignments[j] and label[i] == label[j]:
            TP += 1
        elif cluster_assignments[i] != cluster_assignments[j] and label[i] == label[j]:
            FP += 1
        elif cluster_assignments[i] == cluster_assignments[j] and label[i] != label[j]:
            FN += 1
        elif cluster_assignments[i] != cluster_assignments[j] and label[i] != label[j]:
            TN += 1

accuracy = (TP + TN) / (TP + TN + FP + FN)
print('Accuracy of K-means is ', accuracy)

# Plotting
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
title = plt.title('Elbow for K-Means clustering')
plt.show()
