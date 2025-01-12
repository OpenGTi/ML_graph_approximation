import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from joblib import load

# Suppose we have a new DataFrame `df_new` with the same structure and columns as the original data
df_new = pd.read_excel("datafiles/guess.xlsx")

variable_0 = "log_k"
variable_1 = "log_Eb_Es"
variable_2 = "log_L_d"

# Extract features from df_new (make sure columns match the original training features)
X_new = df_new[[variable_0, variable_1, variable_2]].values

# Load the previously fitted scaler (it should have been saved previously, for example using joblib)
# If you did not save it, you must have kept the 'scaler' object in memory or re-fit it on the original training data.
# WARNING: Re-fitting on new data is NOT recommended. You should reuse the original scaler.
# For demonstration:


# Custom optimizer function
def custom_optimizer(obj_func, initial_theta, bounds):
    result = fmin_l_bfgs_b(
        func=obj_func,
        x0=initial_theta,
        bounds=bounds,
        maxiter=30000  # Increased iteration number
    )
    return result[0], result[1]  # Return only the optimized parameters and function value

scaler = load("scaler_gpr_final2.joblib")  # If you saved the scaler before
# If you still have `scaler` from the original code session, just reuse it:
X_new_scaled = scaler.transform(X_new)

# Load the saved model
# model = load_model("fcnn_regression_model_final.h5")
model = joblib.load('gaussian_process_model_final2.joblib')

# Make predictions on the new scaled data
y_new_pred = model.predict(X_new_scaled)
y_new_pred = y_new_pred.reshape(-1,1)

print(f"Prediction for new entries is:\n {y_new_pred}")

# If your targets were not scaled, then `y_future_pred` is directly interpretable.
# If you had scaled y during training, you would need to inverse-transform the predictions:
# y_new_pred = scaler_y.inverse_transform(y_future_pred)
