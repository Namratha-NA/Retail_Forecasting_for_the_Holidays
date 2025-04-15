# Retail Intelligence: Holiday Demand Forecasting with Machine Learning

## Overview

This project addresses the challenge of forecasting product demand during major holiday seasons by leveraging machine learning techniques on Walmart retail data. The solution helps optimize inventory planning and minimize stockouts or overstocking during high-traffic sales periods like Thanksgiving, Diwali, and Christmas.

The project compares and implements a suite of regression algorithms—ranging from interpretable models like Linear Regression to high-performance models such as XGBoost and Feed-Forward Neural Networks—to uncover nonlinear trends and boost forecast accuracy.

---

## Dataset

- **Source**: [Walmart Retail Sales Dataset](https://www.kaggle.com/datasets)
- **Scope**: Weekly sales data across 40 Walmart stores (2010–2012)
- **Features**:
  - `Weekly_Sales`, `Store`, `Date`, `Holiday_Flag`
  - Economic indicators: `Temperature`, `Fuel Price`, `CPI`, `Unemployment`

---

## Key Highlights

- **Feature Engineering**:
  - Extracted `year`, `month`, `quarter`, and `season` from the `Date` column
  - Applied outlier detection and removed non-informative features (e.g., weekday bias)
  - Encoded categorical and numerical features using Binary Encoder and Standard Scaler

- **Exploratory Analysis**:
  - Identified key economic drivers like `CPI` and `Unemployment` affecting demand
  - Detected strong correlation between holidays and sales surges

- **Modeling Pipeline**:
  - **Linear Regression** – RMSE: 0.0279 | R²: 97.13%
  - **Decision Tree** – RMSE: 0.0489 | R²: 91.22%
  - **Random Forest** – RMSE: 0.0377 | R²: 94.8%
  - **XGBoost** – RMSE: 0.0258 | R²: **97.55%**
  - **Feed-Forward Neural Network** – RMSE: 0.0187 | R²: 96.47%

- **Tuning & Validation**:
  - GridSearchCV for hyperparameter tuning
  - k-Fold Cross-Validation for robustness

---

## Technologies Used

- **Languages**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Keras/TensorFlow, Matplotlib, Seaborn  
- **Modeling Techniques**: Regression, Hyperparameter Tuning, Polynomial Features  
- **Encoding**: Standard Scaler, Binary Encoder, OneHotEncoder  
- **Validation**: RMSE, R², Cross-Validation, Scatter Plots  

---

## Results & Impact

- Identified XGBoost and Feedforward Neural Networks as top-performing models for retail demand prediction
- Enabled strategic inventory planning for holidays and promotional events
- Provided insights into external factors like economic indicators influencing demand

---

## Future Enhancements

- Integrate real-time demand prediction using streaming data  
- Add multi-modal inputs (e.g., marketing campaigns, weather forecasts)  
- Deploy models into retail ERP systems for dynamic inventory updates  
- Build a dashboard for live demand visualization  

---

