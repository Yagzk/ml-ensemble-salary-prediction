# ml-ensemble-salary-prediction

## Project Overview
The goal of this project is to build a **regression model** that predicts **baseball player salaries (Salary)** using the **Hitters dataset**, and to compare the performance of different machine learning model families.  
The main focus of the assignment is on **ensemble and boosting-based models**, including **Gradient Boosting, XGBoost, LightGBM, and CatBoost**.

---

## What I Did
This notebook implements an **end-to-end machine learning workflow**, covering data preprocessing, modeling, evaluation, and interpretation.

### 1. Data Loading & Exploration
- Loaded the `Hitters.csv` dataset  
- Checked data types, dataset shape, and missing values  
- Visualized the **Salary** distribution and observed that salary values are clustered in lower ranges  

### 2. Missing Value Handling
- Filled missing values in the `Salary` column using the **median**  

### 3. Categorical Encoding
- Converted categorical features (`League`, `Division`, `NewLeague`) into numerical form using **One-Hot Encoding**  

### 4. Outlier Analysis & Winsorization
- Performed outlier analysis on numerical features using **Z-score**  
- For linear models, applied **winsorization (1st–99th percentile clipping)** to reduce sensitivity to extreme values  
- Observed that outlier handling can affect **overfitting and generalization behavior**, especially in linear models  

### 5. Modeling & Evaluation
The following regression models were trained and evaluated:

- **Linear Models**
  - Linear Regression  
  - Ridge Regression  
  - Lasso Regression  

- **Tree-Based Model**
  - Random Forest  

- **Boosting Models**
  - Gradient Boosting  
  - XGBoost  
  - LightGBM  
  - CatBoost  

Models were evaluated using:
- Train / test split  
- **5-fold Cross Validation**  

Evaluation metrics:
- Mean Squared Error (**MSE**)  
- Root Mean Squared Error (**RMSE**)  
- Mean Absolute Error (**MAE**)  
- Coefficient of Determination (**R²**)  

### 6. Hyperparameter Tuning
- Used **GridSearchCV** to find optimal hyperparameters for boosting models and selected regressors  

### 7. Feature Importance
- Visualized **LightGBM feature importance** to interpret which features have the greatest impact on salary predictions  

---

## Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- LightGBM  
- CatBoost  
- Matplotlib / Seaborn  

---

## Dataset
- **Hitters Dataset**  
- Target variable: `Salary`

---

## Notes
This project was developed as part of a machine learning assignment with a focus on **ensemble and boosting methods**.  
The notebook is structured to demonstrate the full machine learning pipeline from raw data to model interpretation.
