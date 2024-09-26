import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("D:/neuai/Linear Regression/Salary_Data.csv")
print(df)


# Input values
x_input = df.YearsExperience
y_input = df.Salary

print(x_input)
print(y_input)


# Convert input strings to lists of floats
#x = np.array(list(map(float, x_input.split(',')))).reshape(-1, 1)  # Reshape for sklearn
#y = np.array(list(map(float, y_input.split(','))))

x = df.drop('YearsExperience',axis=1)
y = df.drop('Salary',axis=1)
#x = np.array(x_input.reshape(-1, 1))
#y = np.array(y_input)
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
