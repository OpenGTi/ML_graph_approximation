import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Step 1: Load and concatenate data from all worksheets
file_path = r'datafiles/ig_raw_data.xlsx'  # Path to the Excel file

# Load the Excel file
try:
    xls = pd.ExcelFile(file_path)
except FileNotFoundError:
    raise FileNotFoundError(f"The file '{file_path}' was not found. Please check the path and try again.")

variable_0 = "beta"
variable_dependent = "ig_value"
required_columns = {variable_0, variable_dependent}
sheet_to_read = 'hd02'

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

# Step 4: Extract the independent (X) and dependent (y) variables
X = data[variable_0].values.reshape(-1, 1)  # Independent variable (x values) = beta
y = data[variable_dependent].values  # Dependent variable (y values) = IG


def nonlinear_model(x, c1, c2, c3):
    """
    y = 1 / (c1 + c2 * x^(-c3))
    """
    return 1.0 / (c1 + c2 * x ** -c3)

m, n, k = 4.898, 0.405, 0.945

# Generate predictions using elementwise arrays
y_pred = nonlinear_model(X, m, n, k)

# Compute R² and MSE
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"\nR² (Test): {r2:.4f}")
print(f"MSE (Test): {mse:.2e}")
