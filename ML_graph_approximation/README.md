# README

This repository contains the technical paper titled **_Machine Learning Approach for Approximating Design Parameters from Engineering Graphs_**, published in the Tetra Tech Coffey East Coast Symposium 2025.

## Folder Structure

- **datafiles**  
  Contains all raw data extracted from the engineering graphs studied in the paper.

- **python_scripts**  
  Contains the Python scripts used to derive the required polynomials and non-linear functions. This folder includes two Python files for training Gaussian Process Regression (GPR) and Fully Connected Neural Network (FCNN) models, as well as scripts for plotting and parameter prediction.

- **models**  
  Holds the final machine learning models (GPR and FCNN) along with their corresponding scaler files.

## Instructions

1. **Clone the Repository**  
   You are encouraged to clone the entire repository to your local machine for best use.

2. **Models and Files**  
   - Two trained ML models (GPR and FCNN) are available in the `models` folder.  
   - The GPR model is saved in `.joblib` format, and the FCNN model is saved in `.h5` format.  
   - Each model has an associated scaler file required for predicting Rb values accurately.

3. **Prediction Script**  
   - Use the Python script `Case2_prediction.py` (or an equivalent script) to perform predictions.  
   - Ensure that the model and scaler files are either in the same folder as `Case2_prediction.py` or that you provide the correct file paths in the script.

4. **Performance Note**  
   In the author's opinion, the FCNN model provides better overall prediction performance compared to the GPR model.

5. **Excel-Python Interaction**  
   If you prefer to interact with Python from Excel, note that certain minimum Excel version requirements must be met (e.g., certain builds of Excel 365).

---

## Standard Scripts for Prediction

Below is an example of how to load either the GPR or FCNN model, transform new data, and generate predictions:

```python
from joblib import load
import joblib
from tensorflow.keras.models import load_model

# Load the scaler
scaler = load("name_of_your_scaler_file.joblib")

# Suppose you have new input data stored in X_new
X_new_scaled = scaler.transform(X_new)

# Load the pre-trained ML model (choose either one)
model = joblib.load("name_of_GPR_model_file.joblib")  # GPR example
# model = load_model("name_of_FCNN_model_file.h5")    # FCNN example

# Predict on new data
y_new_pred = model.predict(X_new_scaled)
y_new_pred = y_new_pred.reshape(-1, 1)

print(f"Prediction for new entries is:\n {y_new_pred}")
```

## Additional Resources
Excel minimum requirement:
- ![image](https://github.com/user-attachments/assets/d47e9d70-d06d-4276-a901-cd212319749e)
- Users must have the Python Add-in installed in advance.
- Enterprise and Business Channel: Version 2408 (Build 17928.20114)
- Monthly Enterprise Channel: Version 2408 (Build 17928.20216)
- Family and Personal Channel: Version 2405 (Build 17628.20164)
- Education (Insider Program Current Channel): Version 2405 or higher

## Contact Information
For any inquiries or feedback, please contact the author at [opengti@icloud.com] or [https://www.linkedin.com/in/wai-leung-ng-5b1ab3214/]
