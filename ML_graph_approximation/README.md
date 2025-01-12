# README

This repository contains the technical paper titled "Machine Learning Approach for Approximating Design Parameters from Engineering Graphs" published in Tetra Tech Coffey East Coast Symposium 2025.

## Folder Structure
- **[datafiles]**: This folder contains all raw data extracted from the engineering graphs studied in the paper.
  
- **[python_scripts]**: This folder contains the necessary python scripts that derive the required polynomials and non-linear functions.  There are two pyhton files for training Gaussian Process Regression and Fully Connected Neural Network models.  In addition, codes for plotting and parameter prediction are alos included.
  
- **[models]**: This folder contains the final machine learning models developed by Process Regression and Fully Connected Neural Network methods with their correponding scaler files.
  
## Instructions

1. **Clone the Repository**  
   You are encouraged to clone the entire repository to your local machine.

2. **Models and Files**  
   - Two trained ML models (GPR and FCNN) are available in the `models` folder.  
   - The GPR model is saved in `.joblib` format, and the FCNN model is saved in `.h5` format.  
   - Each model has an associated scaler file required for proper prediction of Rb values.

3. **Prediction Script**  
   - Use the Python script `Case2_prediction.py` to perform predictions.  
   - Ensure that the model and scaler files are either in the same folder as `Case2_prediction.py` **or** provide the correct file paths in the script.

4. **Performance Note**  
   In the author's opinion, the FCNN model provides better overall prediction performance compared to the GPR model.

5. **Excel-Python Interaction**  
   If you prefer to interact with Python from Excel, be aware of the minimum Excel version requirements (e.g., certain versions of Excel 365 may be necessary).

---

## Standard Scripts for Prediction

Below is an example of how to load the GPR model or FCNN model, transform new data, and generate predictions:

```python
from joblib import load
import joblib
from tensorflow.keras.models import load_model

# Load the scaler
scaler = load("name_of_your_scaler_file.joblib")

# Suppose you have your new input data stored in X_new
X_new_scaled = scaler.transform(X_new)

# Load the pre-trained ML model, choose either one
model = joblib.load("name_of_GPR_model_file.joblib")  # for GPR
# model = load_model("name_of_FCNN_model_file.h5")    # for FCNN

# Predict on new data
y_new_pred = model.predict(X_new_scaled)
y_new_pred = y_new_pred.reshape(-1, 1)

print(f"Prediction for new entries is:\n {y_new_pred}")
```

## Additional Resources
Excel minimum requirement:
- ![image](https://github.com/user-attachments/assets/d47e9d70-d06d-4276-a901-cd212319749e)
- Readers need to have Python Add-in installed in advance.
- Enterprise and Business Channel with Version 2408 (Build 17928.20114)
- Monthly Enterprise Channel with Version 2408 (Build 17928.20216)​​​
- Family and Personal Channel with Version 2405 (Build 17628.20164)
- Education users running the Current Channel (Preview) through the Microsoft 365 Insider Program

## Contact Information
For any inquiries or feedback, please contact the author at [opengti@icloud.com] or [https://www.linkedin.com/in/wai-leung-ng-5b1ab3214/]
