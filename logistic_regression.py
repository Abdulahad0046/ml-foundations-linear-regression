import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# features: [Study Hours, Attendance]

X = np.array([[2, 60], [4,70], [6,80], [8,90], [1,50], [3,65], [5,75], [7,85]])

# Labels: Pass(1) or Fail(0)
y = np.array([0, 0, 1, 1, 0, 0, 1, 1])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# Create Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_prediction = model.predict(X_test)

# Predict probabilities
y_prob = model.predict_proba(X_test)

print("Predicted probabilities:")
print(y_prob)


# Model Evaluation
accuracy = accuracy_score(y_test, y_prediction)
cm = confusion_matrix(y_test, y_prediction)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
