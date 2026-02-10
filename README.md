# Linear Regression: Study Hours vs Marks

## Problem
The goal of this project is to understand and implement Linear Regression to predict student marks based on the number of study hours.

## Approach
- Created a simple dataset representing study hours and corresponding marks
- Visualized the data using scatter plots
- Trained a Linear Regression model using scikit-learn
- Predicted marks and plotted the regression line
- Printed model parameters (slope and intercept)

## Output
- A regression line showing the relationship between study hours and marks
- Model parameters:
  - Slope indicates how much marks increase per additional hour of study
  - Intercept represents baseline marks when study hours are zero

## What I Learned
- How Linear Regression works conceptually and mathematically
- The meaning of slope and intercept in real-world terms
- How a model learns patterns from data
- How to visualize model predictions
- How to structure and document a machine learning project

## Model Evaluation

To evaluate the performance of the Linear Regression model, I used the following metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted values.
- **MSE (Mean Squared Error):** Penalizes larger errors by squaring them.
- **RMSE (Root Mean Squared Error):** Square root of MSE, in the same unit as the target variable.

### Results
The MAE and RMSE values are nearly zero, indicating that the model fits the dataset extremely well.

### Observation
This happens because the dataset follows a perfect linear relationship.  
In real-world scenarios, data usually contains noise and outliers, so error values are typically higher.

### Key Learning
- Always evaluate models on unseen test data
- Low error does not always mean a realistic model
- Understanding data is as important as choosing the model
