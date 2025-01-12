import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load data from Excel file
file_path = r'datafiles/ig_raw_data.xlsx'  # Path to the uploaded Excel file
sheet_name = 'regression'  # Replace with the sheet name if necessary

variable_x = "t"
variable_y = "k"


# Read the Excel file
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Assuming the first column is 'x' and the second column is 'y'
x = data[variable_x].values.reshape(-1, 1)  # Reshape for sklearn
y = data[variable_y].values

# Function to perform polynomial regression
def polynomial_regression(x, y, degree):
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x)

    # Fit the model
    model = LinearRegression()
    model.fit(x_poly, y)

    # Make predictions
    y_pred = model.predict(x_poly)

    # Calculate R² score and MSE
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    # Get coefficients
    coefficients = model.coef_
    intercept = model.intercept_

    return r2, mse, coefficients, intercept

# Perform polynomial regression for degrees 1 to 3
for degree in range(1, 4):
    r2, mse, coefficients, intercept = polynomial_regression(x, y, degree)
    print(f"Degree: {degree}")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.2e}")
    print(f"Coefficients: {coefficients}\n")
    print(f"Intercept: {intercept}\n")
