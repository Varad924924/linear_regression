import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Input values
x_input = input("Enter the value of X (comma-separated): ")
y_input = input("Enter the value of Y (comma-separated): ")

# Convert input strings to lists of floats
x = np.array(list(map(float, x_input.split(',')))).reshape(-1, 1)  # Reshape for sklearn
y = np.array(list(map(float, y_input.split(','))))

# Initialize the linear regression model
model = LinearRegression()

# Fit the model
model.fit(x, y)

# Get model parameters
b0 = model.intercept_
b1 = model.coef_[0]

print(f"b0: {b0}")
print(f"b1: {b1}")

# Make predictions
ypred = model.predict(x)

# Calculate residuals and total sum of squares (TSS)
residuals = mean_squared_error(y, ypred) * len(y)  # Mean Squared Error * number of samples
tss = np.sum((y - np.mean(y)) ** 2)

# Calculate R-squared (coefficient of determination)
r2 = r2_score(y, ypred)

print(f"Residuals (Sum of Squares of Residuals): {residuals}")
print(f"Total Sum of Squares (TSS): {tss}")
print(f"R-squared (Coefficient of Determination): {r2}")

if r2 == 1:
    print("Model is fit")
else:
    print("Model is not fit")
