import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.optimize import fmin_l_bfgs_b


# Load the saved model and scaler

# Custom optimizer function
def custom_optimizer(obj_func, initial_theta, bounds):
    result = fmin_l_bfgs_b(
        func=obj_func,
        x0=initial_theta,
        bounds=bounds,
        maxiter=30000  # Increased iteration number
    )
    return result[0], result[1]  # Return only the optimized parameters and function value

################
# model = load_model("fcnn_regression_model.keras")
# scaler = load("scaler_fcnn.joblib")
###############
model = joblib.load('gaussian_process_model_final2.joblib')
scaler = load("scaler_gpr_final2.joblib")
##############


# ==========================
# 1. Preprocessing
# ==========================

file_path = r'datafiles/rb_raw_fcnn.xlsx'
xls = pd.ExcelFile(file_path)

# variable_0, variable_1, variable_2 = "log_k", "log_Eb_Es", "L_d" # for FCNN
variable_0, variable_1, variable_2 = "log_k", "log_Eb_Es", "log_L_d" # for GPR
variable_dependent = "R_b"
required_columns = {variable_0, variable_1, variable_2, variable_dependent}

df_list = []
for sheet_name in xls.sheet_names:
    df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
    if not required_columns.issubset(df_sheet.columns):
        print(f"Sheet {sheet_name} is missing required columns.")
        continue
    df_list.append(df_sheet)

df_raw = pd.concat(df_list, ignore_index=True)

df = df_raw
# When a specific L/d layer is targeted, obtain the relevant data by filtering
# df = df[df['L_d'] == 30]
X = df[[variable_0, variable_1, variable_2]].values
y = df[variable_dependent].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)


# ==========================
# Visualization
# ==========================

# Example: Two 3D surfaces with two different fixed values of variable_2
variable0_min, variable0_max = X[:, 0].min(), X[:, 0].max()
variable1_min, variable1_max = X[:, 1].min(), X[:, 1].max()

variable0_range = np.linspace(variable0_min, variable0_max, 100)
variable1_range = np.linspace(variable1_min, variable1_max, 100)
variable0_grid, variable1_grid = np.meshgrid(variable0_range, variable1_range)

# Function to generate predictions given a fixed variable_2
def predict_surface(var2_value):
    X_grid = np.column_stack([
        variable0_grid.ravel(),
        variable1_grid.ravel(),
        np.full_like(variable0_grid.ravel(), var2_value)
    ])
    X_grid_scaled = scaler.transform(X_grid)
    predictions = model.predict(X_grid_scaled).reshape(variable1_grid.shape)
    return predictions

# Generate predictions for different variable_2
# for FCNN input
# var2_fixed_a = 75
# var2_fixed_b = 60
# var2_fixed_c = 50
# var2_fixed_d = 35
# var2_fixed_e = 25
# var2_fixed_f = 20
# var2_fixed_g = 10
# var2_fixed_h = 8
# var2_fixed_i = 5

# for GPR input
var2_fixed_a = np.log10(75)
var2_fixed_b = np.log10(60)
var2_fixed_c = np.log10(50)
var2_fixed_d = np.log10(35)
var2_fixed_e = np.log10(25)
var2_fixed_f = np.log10(20)
var2_fixed_g = np.log10(10)
var2_fixed_h = np.log10(8)
var2_fixed_i = np.log10(5)

fcnn_grid_a = predict_surface(var2_fixed_a)
fcnn_grid_b = predict_surface(var2_fixed_b)
fcnn_grid_c = predict_surface(var2_fixed_c)
fcnn_grid_d = predict_surface(var2_fixed_d)
fcnn_grid_e = predict_surface(var2_fixed_e)
fcnn_grid_f = predict_surface(var2_fixed_f)
fcnn_grid_g = predict_surface(var2_fixed_g)
fcnn_grid_h = predict_surface(var2_fixed_h)
fcnn_grid_i = predict_surface(var2_fixed_i)

# Plotting both surfaces
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# # Surface for var2 = 75
surf_a = ax.plot_surface(
    variable0_grid, variable1_grid, fcnn_grid_a,
    color='red', alpha=0.6, linewidth=0, antialiased=False,
    label='L/d = 75'
)

# Surface for var2 = 60
surf_b = ax.plot_surface(
    variable0_grid, variable1_grid, fcnn_grid_b,
    color='darkorange', alpha=0.6, linewidth=0, antialiased=False,
    label='L/d = 60'
)

# Surface for var2 = 50
surf_c = ax.plot_surface(
    variable0_grid, variable1_grid, fcnn_grid_c,
    color='yellow', alpha=0.6, linewidth=0, antialiased=False,
    label='L/d = 50'
)

# Surface for var2 = 35
surf_d = ax.plot_surface(
    variable0_grid, variable1_grid, fcnn_grid_d,
    color='seagreen', alpha=0.6, linewidth=0, antialiased=False,
    label='L/d = 35'
)

# Surface for var2 = 25
surf_e = ax.plot_surface(
    variable0_grid, variable1_grid, fcnn_grid_e,
    color='lime', alpha=0.6, linewidth=0, antialiased=False,
    label='L/d = 25'
)

# Surface for var2 = 20
surf_f = ax.plot_surface(
    variable0_grid, variable1_grid, fcnn_grid_f,
    color='aqua', alpha=0.6, linewidth=0, antialiased=False,
    label='L/d = 20'
)

# Surface for var2 = 10
surf_g = ax.plot_surface(
    variable0_grid, variable1_grid, fcnn_grid_g,
    color='navy', alpha=0.8, linewidth=0, antialiased=False,
    label='L/d = 10'
)

# # Surface for var2 = 8
# surf_h = ax.plot_surface(
#     variable0_grid, variable1_grid, fcnn_grid_h,
#     color='purple', alpha=1, linewidth=0, antialiased=False,
#     label='L/d = 8'
# )

# Surface for var2 = 5
surf_i = ax.plot_surface(
    variable0_grid, variable1_grid, fcnn_grid_i,
    color='purple', alpha=0.3, linewidth=0, antialiased=False,
    label='L/d = 5'
)

# Add data points
# ax.scatter(
#     X_train[:, 0], X_train[:, 1], y_train,
#     color='blue', label='Training Data', s=10
# )
# ax.scatter(
#     X_test[:, 0], X_test[:, 1], y_test,
#     color='orange', label='Test Data', s=10
# )

ax.set_title("FCNN Predictions")
ax.set_xlabel('log(k)')
ax.set_ylabel('log(Eb/Es)')
ax.set_zlabel('Rb value')

# **Set the z-axis limits from 0 to 1**
ax.set_zlim(0, 1)


# # Custom legend handles
# legend_elements = [
#     # Line2D([0], [0], marker='o', color='w', label='Training Data', markerfacecolor='blue', markersize=10),
#     Line2D([0], [0], color='red', lw=4, label='L/d = 75 Surface'),
#     Line2D([0], [0], color='green', lw=4, label='L/d = 60 Surface'),
#     Line2D([0], [0], color='blue', lw=4, label='L/d = 50 Surface'),
# ]
# ax.legend(handles=legend_elements, loc='best')

ax.view_init(elev=20, azim=55)

plt.show()
