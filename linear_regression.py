import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# ------------------------------
# Step 1: create Dataset
# -------------------------------

# study hours (input features)
hours = np.array([1,2,3,4,5,6,7,8]).reshape(-1, 1)

# marks (target variable)
marks = np.array([35,40,45,50,55,60,65,70])

# ------------------------------
# Step 2: Visualize data
# -------------------------------

plt.scatter(hours,marks, color='blue')
plt.xlabel("Study Hours")  
plt.ylabel("Marks")
plt.title("Study Hours vs marks")
plt.show()

# -------------------------------
# Step 2.5: Train-Test split
# -------------------------------

x_train, x_test, y_train, y_test = train_test_split(hours, marks, test_size=0.2, random_state=42)

# ------------------------------
# Step 3: Train Linear Regression Model
# -------------------------------

model = LinearRegression()
model.fit(x_train, y_train)

# ------------------------------
# Step 4: Make Predictions
# -------------------------------

y_prediction = model.predict(x_test)

# ------------------------------    
# Step 5: Plot regression line
# ------------------------------

plt.scatter(hours,marks, color='blue', label='Actual Data')
predicted_full = model.predict(hours)  # only for visualization
plt.plot(hours, predicted_full, color='red', label='Regression Line')
plt.xlabel("study hours")
plt.ylabel("Marks")
plt.title("Linear Regression result")
plt.legend()
plt.show()

# ------------------------------
# Step 6: print model parameters
# ------------------------------

print("Slope(m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# ------------------------------
# Step 7: Model Evaluation
# ------------------------------

# MAE: Average absolute error
mae = mean_absolute_error(y_test, y_prediction)

# MSE: Mean squared error
mse = mean_squared_error(y_test, y_prediction)

# RMSE: Root mean squared error
rmse = mse ** 0.5

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Overall Conclusion:

# The model learns the relationship between study hours and marks
# Slope tells how much marks increase per additional hour
# Intercept is baseline marks when study hours are zero
