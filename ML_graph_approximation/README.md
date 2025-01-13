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

3. **Prediction App**  
   - Use the Python script `Case2_prediction.py` (or an equivalent script) to perform predictions.  
   - You may use the excel file `new_data_example.xlsx` as an example. This excel contain a simple table with all the independent variables.
   - Once `Case2_prediction.py` is opened, follow the step-by-step precedures to perform the prediction. Finally the results will be saved in a seperate excel file.

4. **Performance Note**  
   In the author's opinion, the FCNN model provides better overall prediction performance compared to the GPR model.

5. **Excel-Python Interaction**  
   If you prefer to interact with Python from Excel, note that certain minimum Excel version requirements must be met (e.g., certain builds of Excel 365).

---

## Example

Below is an example of how to load either the GPR or FCNN model, transform new data, and generate predictions:

   - Format of input file:
   - ![image](https://github.com/user-attachments/assets/10ca827f-e772-4729-bbc2-15a5413db0fd)
    
   - Outlook of the app:
   - ![image](https://github.com/user-attachments/assets/bc622fe2-d1d5-47b9-9f33-1de8fcfb1235)
     
   - Output:
   - ![image](https://github.com/user-attachments/assets/3bf163a9-59e9-4aa7-9166-4904bf5cbd83)

## Additional Resources
NIL

## Contact Information
For any inquiries or feedback, please contact the author at [opengti@icloud.com] or [https://www.linkedin.com/in/wai-leung-ng-5b1ab3214/]
