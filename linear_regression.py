import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

# ------------------------------
# Step 3: Train Linear Regression Model
# -------------------------------

model = LinearRegression()
model.fit(hours, marks)

# ------------------------------
# Step 4: Make Predictions
# -------------------------------

predicted_marks = model.predict(hours)

# ------------------------------    
# Step 5: Plot regression line
# ------------------------------

plt.scatter(hours,marks, color='blue', label='Actual Data')
plt.plot(hours, predicted_marks, color='red', label='regression Line')
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

# Overall Conclusion:

# The model learns the relationship between study hours and marks
# Slope tells how much marks increase per additional hour
# Intercept is baseline marks when study hours are zero
