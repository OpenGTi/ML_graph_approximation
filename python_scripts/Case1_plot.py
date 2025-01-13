# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

# Step 1: Load and concatenate data from all worksheets
file_path = r'datafiles/ig_raw_data.xlsx'  # Path to your Excel file

# Load the Excel file
try:
    xls = pd.ExcelFile(file_path)
except FileNotFoundError:
    raise FileNotFoundError(f"The file '{file_path}' was not found. Please check the path and try again.")

variable_0 = "h_d"
variable_1 = "beta"
variable_dependent = "ig_value"
required_columns = {variable_0, variable_1, variable_dependent}
df_list = []  # To store dataframes from each sheet


# Loop through each sheet, load the data, and append to the list
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

X_data = data[[variable_0, variable_1]].values  # Independent variables
y_data = data[variable_dependent].values

# Calculate IG based on the given formula
def ig_cal(h_d, beta):
    t = np.log(h_d)
    m = (-0.0059 * t ** 3 + 0.0411 * t ** 2 - 0.0978 * t + 1.0855) ** 5
    n = -0.0032 * t ** 3 + 0.0011 * t ** 2 + 0.0655 * t + 0.4927
    k = 0.0043 * t ** 2 - 0.0465 * t + 0.8603
    IG = 1 / (m + n * beta ** (-k))
    return IG

y_pred = ig_cal(X_data[:, 0], X_data[:, 1])

# Compute R² and MSE
r2 = r2_score(y_data, y_pred)
mse = mean_squared_error(y_data, y_pred)

print(f"\nR² (Test): {r2:.4f}")
print(f"MSE (Test): {mse:.2e}")


# Step 9: Generate a prediction grid
print("Generating a prediction grid for visualization...")

# Create a grid for h_d and beta
h_d_grid = np.linspace(0.2, 30, 200)
log_beta_grid = np.linspace(-2, 2, 200)
beta_grid = 10**log_beta_grid

h_d_grid, beta_grid = np.meshgrid(h_d_grid, beta_grid)

# Interpolate IG values over the grid
IG_grid = ig_cal(h_d_grid, beta_grid)

print("Prediction grid generated.\n")

# Step 12: Visualization - 3D Surface Plot
print("Generating 3D surface plot for model predictions...")
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the GPR prediction surface
surface = ax.plot_surface(
    h_d_grid, np.log10(beta_grid), IG_grid,
    cmap='inferno', alpha=0.7, linewidth=0, antialiased=False
)

# Add data points
ax.scatter(
    X_data[:, 0], np.log10(X_data[:, 1]), y_data,
    color='green', label='Input Data', s=10
)

# Customize the plot
ax.set_title("3D Illustration of IG Value", fontsize=16)
ax.set_xlabel('h/d', fontsize=12)
ax.set_ylabel('Normalized gibson modulus (beta) in log scale', fontsize=12)
ax.set_zlabel('Displacement influence factor, IG', fontsize=12)

# Create custom legend using Patch for the colormap
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Input Data',
           markerfacecolor='green', markersize=10)
]
ax.legend(handles=legend_elements, loc='best')

# Set axis limits for better visualization
ax.set_xlim([30, 0.2])
ax.set_ylim([np.log10(0.01), np.log10(100)])
ax.set_zlim([np.nanmin(IG_grid)*0.9, np.nanmax(IG_grid)*1.1])  # Dynamically set z-limits

plt.tight_layout()
plt.show()
print("3D surface plot generated.\n")