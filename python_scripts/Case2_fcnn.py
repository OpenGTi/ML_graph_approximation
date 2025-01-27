import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import openpyxl
from matplotlib.lines import Line2D
import joblib

# ==========================
# 1. Preprocessing
# ==========================

file_path = r'datafiles/rb_raw_fcnn.xlsx'
xls = pd.ExcelFile(file_path)

variable_0, variable_1, variable_2 = "log_k", "log_Eb_Es", "L_d"
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

# If you intended to filter by origin, uncomment and adjust:
# df = df_raw[df_raw['origin'] == 'Alluvium']
df = df_raw

X = df[[variable_0, variable_1, variable_2]].values
y = df[variable_dependent].values

# Feature scaling for X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future use
joblib.dump(scaler, 'scaler_fcnn.joblib')
print("Scaler has been saved successfully.")

# If you want to scale y (optional), do:
# scaler_y = StandardScaler()
# y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
# For now, we assume y is not scaled:
y_scaled = y

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)

# ==========================
# 2. Define and Train the Neural Network Model
# ==========================

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')  # regression output
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.2e}, Test MAE: {mae:.4f}")

# ==========================
# 3. Predict and Evaluate the Model
# ==========================

y_pred = model.predict(X_test)
y_pred = y_pred.flatten()

mse_fcnn = mean_squared_error(y_test, y_pred)
r2_fcnn = r2_score(y_test, y_pred)
print(f"Neural Network Mean Squared Error: {mse_fcnn:.2e}")
print(f"Neural Network R^2 Score: {r2_fcnn:.4f}")

# Save the model
model.save("fcnn_regression_model.keras")
print("Model saved to fcnn_regression_model.keras")

# ==========================
# 4. Visualization with Matplotlib
# ==========================

# Example 1: 3D Scatter plot of input features vs. dependent variable
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(df[variable_0], df[variable_1], df[variable_dependent],
               c=df[variable_dependent], cmap='viridis', marker='o')
ax.set_title('3D Feature Distribution')
ax.set_xlabel('log(k)')
ax.set_ylabel('log(Eb/Es)')
ax.set_zlabel('Rb value')
fig.colorbar(p, ax=ax, label=variable_dependent)
plt.show()

# Example 2: 3D surface of the model predictions with variable_2 fixed
# ---------------------------------------------------------------
# Here we fix variable_2 at its mean value. This creates a 3D surface
# representing how the model's predictions vary with variable_0 and variable_1.

variable0_min, variable0_max = X[:, 0].min(), X[:, 0].max()
variable1_min, variable1_max = X[:, 1].min(), X[:, 1].max()

variable2_fixed = X[:, 2].mean()  # Fixing variable_2 to its mean

variable0_range = np.linspace(variable0_min, variable0_max, 400)
variable1_range = np.linspace(variable1_min, variable1_max, 400)
variable0_grid, variable1_grid = np.meshgrid(variable0_range, variable1_range)

# Create input grid with variable_2 fixed
X_grid = np.column_stack([
    variable0_grid.ravel(),
    variable1_grid.ravel(),
    np.full_like(variable0_grid.ravel(), variable2_fixed)
])

X_grid_scaled = scaler.transform(X_grid)
fcnn_grid = model.predict(X_grid_scaled).reshape(variable1_grid.shape)

print("Generating 3D plot for model predictions with variable_2 fixed...")
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
fcnn_surface = ax.plot_surface(
    variable0_grid, variable1_grid, fcnn_grid,
    cmap='viridis', alpha=0.6, linewidth=0, antialiased=False
)

ax.scatter(df[variable_0], df[variable_1], df[variable_dependent],
               color='blue', label='Extracted Data', alpha=0.6, marker='o')

ax.set_title("FCNN Predictions (L/d = 75)")
ax.set_xlabel('log(k)')
ax.set_ylabel('log(Eb/Es)')
ax.set_zlabel('Rb value')

legend_elements = [
    #Line2D([0], [0], color='green', lw=4, label='FCNN Predictions'),
    Line2D([0], [0], marker='o', color='w', label='Training Data', markerfacecolor='blue', markersize=10),

]
ax.legend(handles=legend_elements, loc='best')

plt.show()
print("3D plot generated.\n")

# Example 3: Training and Validation Loss Over Epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()

# Example 4: Predicted vs True Values Scatter Plot
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.title('Predicted vs True Values')
plt.xlabel('True R_b')
plt.ylabel('Predicted R_b')
plt.grid(True)
plt.show()

# Example 5: Residuals Histogram
residuals = y_test - y_pred
plt.figure(figsize=(12, 6))
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.title('Distribution of Residuals')
plt.xlabel('Residual (True - Predicted)')
plt.ylabel('Frequency')
plt.show()

# Example 6: Residuals vs Predicted Values
plt.figure(figsize=(12, 6))
plt.scatter(y_pred, residuals, alpha=0.6, edgecolor='k')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted R_b')
plt.ylabel('Residual (True - Predicted)')
plt.grid(True)
plt.show()
