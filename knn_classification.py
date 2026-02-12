import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset: [Study Hours, Attendance]

X = np.array([[2, 60], [4,65], [6,70], [8,80], [10,90], [3,55], [1,50], [7,85]])

y = np.array([0, 0, 1, 1, 1, 0, 0, 1])

# Train-Test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# KNN Model

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions

y_prediction = knn.predict(X_test)

# Model Evaluation

print("Accuracy:",accuracy_score(y_test, y_prediction))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_prediction))