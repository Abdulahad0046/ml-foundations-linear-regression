import numpy as np
from sklearn.linear_model import LinearRegression

# Input features: hours studied , sleep hours
X = np.array([[2,6],[4,7],[6,8],[8,7],[10,9]])

# Output: marks
y = np.array([40,55,65,70,85])

model = LinearRegression()
model.fit(X, y)

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
