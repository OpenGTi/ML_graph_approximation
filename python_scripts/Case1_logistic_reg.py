# Import necessary libraries
import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator

# Assign variables
variable_0 = "beta"
variable_dependent = "ig_value"
midpoint = 10 # Define the midpoint of x in the trained data for plotting

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Step 1: Load the Excel file
file_path = r'datafiles/ig_raw_data.xlsx'  # Path to the uploaded Excel file
sheet_to_read = 'hd30'

try:
    data = pd.read_excel(file_path, sheet_name=sheet_to_read)
    print(f"Successfully loaded {file_path}")
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
    sys.exit(1)

# Step 2: Display the first few rows to understand the structure of the data
print("\nFirst few rows of the dataset:")
print(data.head())

# Step 3: Data Validation
required_columns = [variable_0, variable_dependent]
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    print(f"Error: The dataset must contain the following columns: {required_columns}")
    print(f"Missing columns: {missing_columns}")
    sys.exit(1)

# Check for missing values
print("\nChecking for missing values:")
print(data[required_columns].isnull().sum())

# Drop rows with missing values
initial_count = len(data)
data = data.dropna(subset=required_columns)
final_count = len(data)
print(f"\nDropped {initial_count - final_count} rows due to missing values.")

# Check for infinite values
print("\nChecking for infinite values:")
x_value_inf = np.isinf(data[variable_0]).sum()
y_value_inf = np.isinf(data[variable_dependent]).sum()
print(f"{variable_0}: {x_value_inf}, {variable_dependent}: {y_value_inf}")

# Remove infinite values
data = data[np.isfinite(data[variable_0]) & np.isfinite(data[variable_dependent])]
removed_inf = (x_value_inf + y_value_inf)
if removed_inf > 0:
    print(f"Dropped {removed_inf} rows due to infinite values.")
else:
    print("No infinite values found.")

# Verify that Beta has no zero or negative values for log scale
if np.any(data[variable_0] <= 0):
    print("Warning: Independent variable_0 or x contains zero or negative values, which are undefined on a logarithmic scale.")
    # sys.exit(1)

# Step 4: Extract the independent (X) and dependent (y) variables
X = data[variable_0].values.reshape(-1, 1)  # Independent variable (x values)
y = data[variable_dependent].values  # Dependent variable (y values)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"\nTraining data points: {len(X_train)}, Testing data points: {len(X_test)}")


# Step 11: Define and fit the nonlinear model on training data
def nonlinear_model(x, constant_1, constant_2, constant_3):
    """
    Nonlinear model function: y = 1 / (constant_1 + constant_2 * x^(constant_3))

    Parameters:
    - x: Independent variable
    - constant_1: Parameter 1
    - constant_2: Parameter 2
    - constant_3: Parameter 3 (exponent)

    Returns:
    - Predicted y values
    """
    return 1 / (constant_1 + constant_2 * x ** constant_3)

# Initial guesses for constants
initial_guesses = [1.0, 1.0, -0.8]  # Starting with constant_3 as -0.8 based on previous value

# Perform curve fitting with bounds
# It's reasonable to bound constant_3 to avoid extreme exponents; adjust as needed
bounds = ([0, 0, -10], [np.inf, np.inf, 10])  # constant_3 is negative

try:
    # Increase max function evaluations
    popt, pcov = curve_fit(nonlinear_model, X_train.flatten(), y_train, p0=initial_guesses, bounds=bounds, maxfev=10000)
    constant_1, constant_2, constant_3 = popt
    print(f"\nEstimated constants:")
    print(f"constant_1 = {constant_1:.4f}")
    print(f"constant_2 = {constant_2:.4f}")
    print(f"constant_3 = {constant_3:.4f}")
except RuntimeError as e:
    print("Error - curve_fit failed:", e)
    constant_1, constant_2, constant_3 = [np.nan, np.nan, np.nan]

# Generate predictions from the nonlinear model on test data
if not np.isnan(constant_1):
    y_pred_nl_test = nonlinear_model(X_test.flatten(), constant_1, constant_2, constant_3)
else:
    y_pred_nl_test = None


#######

# Step 12: Generate predictions across a range of Beta values for visualization with adjusted density
# Define the new minimum for beta_range
x_range_min = X.min()
y_range_max = X.max()

# Define separate ranges with different densities
# Allocate more points between two zones
x_density_low = np.linspace(x_range_min, midpoint, 100, endpoint=False)  # 300 points between 0.01 and 1
x_density_high = np.linspace(midpoint, y_range_max, 100)                # 100 points from 1 to max

# Concatenate the ranges
x_range = np.concatenate((x_density_low, x_density_high)).reshape(-1, 1)

# Corrected print statement without indexing
print(f"\nTotal points in beta_range: {len(x_range)}")
print(f"x_range starts at {x_range.min()} and ends at {x_range.max()}")

# Generate predictions from the nonlinear model across the x range
if not np.isnan(constant_1):
    y_predictions_nl = nonlinear_model(x_range.flatten(), constant_1, constant_2, constant_3)
else:
    y_predictions_nl = np.full_like(x_range.flatten(), np.nan)

# Step 13: Plot the original data and model predictions with adjusted scales
plt.figure(figsize=(12, 6))

# Plot Training Data
plt.scatter(X_train, y_train, color='blue', label='Training Data', alpha=0.6, edgecolor='k')

# Plot Testing Data
plt.scatter(X_test, y_test, color='red', label='Testing Data', alpha=0.6, edgecolor='k')


# Plot Nonlinear Model predictions
if not np.isnan(constant_1):
    plt.plot(x_range, y_predictions_nl, color='purple', label='Nonlinear Model Fit', linestyle='-.')

# Set plot title and labels
plt.title(f"Model Fits for {variable_dependent} vs. {variable_0}")
plt.xlabel(variable_0)
plt.ylabel(variable_dependent)


plt.legend(loc='best')

# Add grid for better readability
plt.grid(True, which='both', linestyle='--', alpha=0.7)  # Enable grid for both major and minor ticks

# Display the plot
plt.show()

# Step 14: Display the nonlinear model equation
if not np.isnan(constant_1):
    nonlinear_equation = f"y = 1 / ({constant_1:.4f} + {constant_2:.4f} * x^({constant_3:.4f}))"
    print("\nEstimated Nonlinear Model Equation:")
    print(nonlinear_equation)

# Step 15: Compute and display performance metrics

# Nonlinear Model Performance on Test Data
if not np.isnan(constant_1):
    mse_nl_test = mean_squared_error(y_test, y_pred_nl_test)
    r2_nl_test = r2_score(y_test, y_pred_nl_test)
else:
    mse_nl_test = np.nan
    r2_nl_test = np.nan

# Display the results
print("\nModel Performance Metrics on Test Data:")

if not np.isnan(mse_nl_test):
    print(f"Nonlinear Model - MSE: {mse_nl_test:.2e}, R²: {r2_nl_test:.4f}")

# Optional: Compute additional metrics
if not np.isnan(mse_nl_test):
    mae_nl_test = mean_absolute_error(y_test, y_pred_nl_test)
    rmse_nl_test = np.sqrt(mse_nl_test)
    print(f"Nonlinear Model - MAE: {mae_nl_test:.4f}, RMSE: {rmse_nl_test:.4f}")

# Step 16: Optional - Plot residuals for both models
plt.figure(figsize=(12, 6))


# Nonlinear Model Residuals
if y_pred_nl_test is not None:
    plt.subplot(2, 1, 2)
    plt.scatter(X_test, y_pred_nl_test - y_test, color='purple', alpha=0.5)
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Nonlinear Model Residuals (Test Data)')
    plt.xlabel(variable_0)
    plt.ylabel('Residuals')

plt.tight_layout()
plt.show()
