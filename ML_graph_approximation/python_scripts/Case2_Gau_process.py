# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, RationalQuadratic, ConstantKernel, \
    DotProduct, WhiteKernel
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from scipy.optimize import fmin_l_bfgs_b
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from functools import partial
import warnings
import joblib


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Step 1: Load and concatenate data from all worksheets
file_path = r'datafiles/rb_raw_GPR.xlsx'  # Path to your Excel file
sheet_to_read = ''

# Load the Excel file
try:
    xls = pd.ExcelFile(file_path)
except FileNotFoundError:
    raise FileNotFoundError(f"The file '{file_path}' was not found. Please check the path and try again.")

variable_0 = "log_k"
variable_1 = "log_Eb_Es"
variable_2 = "log_L_d"
variable_dependent = "R_b"
required_columns = {variable_0, variable_1, variable_2, variable_dependent}
df_list = []  # To store dataframes from each sheet

# Loop through each sheet, load the data, and append to the list
if sheet_to_read:
    df = pd.read_excel(file_path, sheet_name=sheet_to_read)
    df_list.append(df)
else:
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if not required_columns.issubset(df.columns):
            print(f"Sheet '{sheet_name}' is missing required columns. Skipping this sheet.")
            continue
        df_list.append(df)

# Concatenate all dataframes
if not df_list:
    raise ValueError("No sheets contain the required columns. Please check your Excel file.")
data = pd.concat(df_list, ignore_index=True)

# Step 2: Data Cleaning - Drop rows with missing values
data = data.dropna(subset=required_columns)

# Step 3: Extract the independent and dependent variables
X = data[[variable_0, variable_1, variable_2]].values  # Independent variables
y = data[variable_dependent].values        # Dependent variable

# Step 4: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, 'scaler_gpr_A.joblib')
print("Scaler has been saved successfully.")

######################################
#Step 6: Initialize and fit the Gaussian Process model with GridSearchCV

# Custom optimizer function
def custom_optimizer(obj_func, initial_theta, bounds):
    result = fmin_l_bfgs_b(
        func=obj_func,
        x0=initial_theta,
        bounds=bounds,
        maxiter=30000  # Increased iteration number
    )
    return result[0], result[1]  # Return only the optimized parameters and function value

kernels = [
    RBF(length_scale=0.1),
    RBF(length_scale=1.0),
    RBF(length_scale=10.0),
    Matern(length_scale=0.1, nu=0.5),
    Matern(length_scale=1, nu=1.5),
    Matern(length_scale=10, nu=2.5),
    RationalQuadratic(length_scale=0.1, alpha=0.01),
    RationalQuadratic(length_scale=0.5, alpha=0.1),
    RationalQuadratic(length_scale=1.0, alpha=1.0),
    RationalQuadratic(length_scale=10.0, alpha=1.0),
]

param_grid = {
    'alpha': [1e-9, 1e-6, 1e-3],
    'kernel': kernels,
    'normalize_y': [False, True],
    'n_restarts_optimizer': [5, 10]
}

gpr_cv = GridSearchCV(
    GaussianProcessRegressor(n_restarts_optimizer=5, random_state=42, optimizer=custom_optimizer),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1
)
print("Searching for best GPR parameters...")

gpr_cv.fit(X_train_scaled, y_train)
best_gpr = gpr_cv.best_estimator_
print("Best GPR parameters found:", gpr_cv.best_params_)




# Save the trained model
joblib.dump(best_gpr, 'gaussian_process_model_A.joblib')
print("\nTrained model has been saved successfully.")


# Step 7: Predict y_values for the training and test data using the best model
y_train_pred_gpr = best_gpr.predict(X_train_scaled)
y_test_pred_gpr = best_gpr.predict(X_test_scaled)

# Step 8: Calculate R-squared, MSE, and MAE for Gaussian Process
r2_train_gpr = r2_score(y_train, y_train_pred_gpr)
mse_train_gpr = mean_squared_error(y_train, y_train_pred_gpr)
mae_train_gpr = mean_absolute_error(y_train, y_train_pred_gpr)

r2_test_gpr = r2_score(y_test, y_test_pred_gpr)
mse_test_gpr = mean_squared_error(y_test, y_test_pred_gpr)
mae_test_gpr = mean_absolute_error(y_test, y_test_pred_gpr)

print(f"Gaussian Process - Training Data - R-squared: {r2_train_gpr:.4f}, MSE: {mse_train_gpr:.2e}, MAE: {mae_train_gpr:.4f}")
print(f"Gaussian Process - Test Data - R-squared: {r2_test_gpr:.4f}, MSE: {mse_test_gpr:.2e}, MAE: {mae_test_gpr:.4f}\n")

# Step 9: Generate a fine grid for variable_0 and variable_1
variable0_min, variable0_max = X[:, 0].min(), X[:, 0].max()
variable1_min, variable1_max = X[:, 1].min(), X[:, 1].max()

variable0_range = np.linspace(variable0_min, variable0_max, 200)
variable1_range = np.linspace(variable1_min, variable1_max, 200)
variable0_grid, variable1_grid = np.meshgrid(variable0_range, variable1_range)
X_grid = np.column_stack([variable0_grid.ravel(), variable1_grid.ravel()])

# Step 10: Scale the grid data
X_grid_scaled = scaler.transform(X_grid)

# Step 11: Predict with Gaussian Process on the fine grid
rb_gpr_grid = best_gpr.predict(X_grid_scaled).reshape(variable1_grid.shape)

# Step 12: Apply Gaussian smoothing (sigma=0 means no smoothing, but code is left for consistency)
rb_gpr_grid_smooth = gaussian_filter(rb_gpr_grid, sigma=0)

# Step 13: Plotting Gaussian Process fits
print("Generating 3D plot for model predictions...")
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot Gaussian Process predictions
gpr_surface = ax.plot_surface(
    variable0_grid, variable1_grid, rb_gpr_grid_smooth,
    cmap='viridis', alpha=0.6, linewidth=0, antialiased=False
)

# Add data points
ax.scatter(
    X_train[:, 0], X_train[:, 1], y_train,
    color='blue', label='Training Data', s=10
)
ax.scatter(
    X_test[:, 0], X_test[:, 1], y_test,
    color='orange', label='Test Data', s=20
)

# Setting labels and title
ax.set_title("Gaussian Process Regression", fontsize=16)
ax.set_xlabel('log(k)', fontsize=12)
ax.set_ylabel('log(Eb/Es)', fontsize=12)
ax.set_zlabel('Rb value', fontsize=12)

# Create custom legend
legend_elements = [
    # Line2D([0], [0], color='green', lw=4, label='Gaussian Process Predictions'),
    Line2D([0], [0], marker='o', color='w', label='Training Data',
           markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Test Data',
           markerfacecolor='orange', markersize=10)
]
ax.legend(handles=legend_elements, loc='best')

plt.show()
print("3D plot generated.\n")

# Step 14: Actual vs. Predicted plots
print("Generating Actual vs. Predicted plots...")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred_gpr, color='blue', label='Training Data', alpha=0.6)
plt.scatter(y_test, y_test_pred_gpr, color='orange', label='Test Data', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel(f'Actual {variable_dependent} Value', fontsize=12)
plt.ylabel(f'Predicted {variable_dependent} Value', fontsize=12)
plt.title('Gaussian Process Predictions vs. Actual Values', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()
print("Actual vs. Predicted plots generated.\n")

# Step 15: Compute residuals
residuals_gpr_train = y_train - y_train_pred_gpr
residuals_gpr_test = y_test - y_test_pred_gpr

# Step 16: Plot the residuals
print("Generating Residuals plots...")
plt.figure(figsize=(12, 6))

# Gaussian Process Residuals vs Variable 0 (Training)
plt.subplot(2, 2, 1)
plt.scatter(X_train[:, 0], residuals_gpr_train, color='blue', alpha=0.6)
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel(variable_0, fontsize=12)
plt.ylabel(f'Residual (Actual {variable_dependent} - Prediction)', fontsize=12)
plt.title(f'Gaussian Process Residuals vs {variable_0} (Training Data)', fontsize=14)

# Gaussian Process Residuals vs Variable 1 (Training)
plt.subplot(2, 2, 2)
plt.scatter(X_train[:, 1], residuals_gpr_train, color='blue', alpha=0.6)
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel(variable_1, fontsize=12)
plt.ylabel(f'Residual (Actual {variable_dependent} - Prediction)', fontsize=12)
plt.title(f'Gaussian Process Residuals vs {variable_1} (Training Data)', fontsize=14)

plt.tight_layout()
plt.show()
print("Residuals plots generated.\n")

# Later in your code or in a different script
# Load the model and scaler
# loaded_model = joblib.load('file name of the model.joblib')  # Replace with actual filename
# scaler = joblib.load('file name of the scaler.joblib')  # Replace with actual filename
# print("Model and scaler have been loaded successfully.")

# Prepare new data for prediction (ensure it's in the correct format)
# new_data = np.array([[value1, value2, value3]])  # Replace with actual values
# new_data_scaled = scaler.transform(new_data)  # Scale the new data

# Make predictions
# predictions = loaded_model.predict(new_data_scaled)
#
# # Display the predictions
# print("Predictions for the new data:", predictions)
