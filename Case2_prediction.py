import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from joblib import load

# Custom optimizer function
def custom_optimizer(obj_func, initial_theta, bounds):
    result = fmin_l_bfgs_b(
        func=obj_func,
        x0=initial_theta,
        bounds=bounds,
        maxiter=30000  # Increased iteration number
    )
    return result[0], result[1]  # Return only the optimized parameters and function value

def perform_prediction(input_file, output_folder, file_name, model_choice):
    try:
        # Load the dataset
        df_new = pd.read_excel(input_file)

        # Define feature variables based on the model choice
        variable_0 = "log_k"
        variable_1 = "log_Eb_Es"
        variable_2 = "log_L_d" if model_choice == "GPR" else "L_d"

        # Extract features
        X_new = df_new[[variable_0, variable_1, variable_2]].values

        # Load the scaler based on the model choice
        scaler_file = "models/scaler_gpr_final2.joblib" if model_choice == "GPR" else "models/scaler_fcnn.joblib"
        scaler = load(scaler_file)
        X_new_scaled = scaler.transform(X_new)

        # Load the model
        model_file = "models/gaussian_process_model_final2.joblib" if model_choice == "GPR" else "models/fcnn_regression_model.keras"
        if model_choice == "GPR":
            model = joblib.load(model_file)
            y_new_pred = model.predict(X_new_scaled).reshape(-1, 1)
        else:
            model = load_model(model_file)
            y_new_pred = model.predict(X_new_scaled).reshape(-1, 1)

        # Add predictions to the DataFrame
        df_new["Rb"] = y_new_pred

        # Save the results
        output_path = f"{output_folder}/{file_name}.xlsx"
        df_new.to_excel(output_path, index=False)
        messagebox.showinfo("Success", f"Results saved to {output_path}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create the tkinter app
root = tk.Tk()
root.title("Machine Learning Prediction App (powered by Open GTi")
root.geometry("840x512")

# Input file selection
def choose_input_file():
    input_file_path.set(filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")]))

def choose_output_folder():
    output_folder_path.set(filedialog.askdirectory())

input_file_path = tk.StringVar()
output_folder_path = tk.StringVar()
file_name = tk.StringVar()
model_choice = tk.StringVar(value="GPR")

# Input file
tk.Label(root, text="Input File:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=input_file_path, width=40).grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=choose_input_file).grid(row=0, column=2, padx=10, pady=5)

# Output folder
tk.Label(root, text="Output Folder:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=output_folder_path, width=40).grid(row=1, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=choose_output_folder).grid(row=1, column=2, padx=10, pady=5)

# File name
tk.Label(root, text="File Name:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=file_name, width=40).grid(row=2, column=1, padx=10, pady=5)

# ML Models dropdown
tk.Label(root, text="ML Models:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
tk.OptionMenu(root, model_choice, "GPR", "FCNN").grid(row=3, column=1, padx=10, pady=5, sticky="w")

# Calculate button
def on_calculate():
    if not input_file_path.get() or not output_folder_path.get() or not file_name.get():
        messagebox.showwarning("Input Required", "Please provide all inputs before proceeding.")
        return
    perform_prediction(input_file_path.get(), output_folder_path.get(), file_name.get(), model_choice.get())

tk.Button(root, text="Calculate", command=on_calculate).grid(row=4, column=1, padx=10, pady=20)

root.mainloop()



# If your targets were not scaled, then `y_future_pred` is directly interpretable.
# If you had scaled y during training, you would need to inverse-transform the predictions:
# y_new_pred = scaler_y.inverse_transform(y_future_pred)
