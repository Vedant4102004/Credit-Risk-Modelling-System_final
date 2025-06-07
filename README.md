# Credit-Risk-Modelling-System_final

# 📊 Credit Risk Modelling System

## 📌 **Project Overview**
This project provides an advanced **Credit Risk Prediction System** using machine learning models.  
It evaluates the risk level of loan applicants based on **financial history, behavioral patterns, and loan attributes**.  

💡 **Key Features**:
✅ Uses **XGBoost Classifier** for predictions  
✅ **Preprocesses & normalizes data** for better accuracy  
✅ Implements **custom credit scoring & risk assessment**  
✅ **Handles categorical & numerical financial data** effectively  

---

## ⚙️ **Installation Guide**
### **Step 1: Clone the Repository**
```sh
git clone https://github.com/Vedant4102004/Credit-Risk-Modelling-System.git
cd Credit-Risk-Modelling-System
```

### **Step 2: Install Dependencies**
Make sure you have all required Python libraries installed:
```sh
pip install -r requirements.txt
```
**Dependencies Include:**  
- **NumPy, Pandas** (Data manipulation)  
- **Scikit-learn** (Feature processing)  
- **XGBoost** (Machine learning model)  
- **Joblib** (Model saving/loading)  

---

## 🏗 **Model Training & Preprocessing**
### **Step 3: Data Preprocessing**
The dataset undergoes feature engineering & preprocessing:  
✅ **Scaling numerical features** using `MinMaxScaler`  
✅ **Encoding categorical variables** for compatibility  
✅ **Handling missing values** appropriately  

Example preprocessing step:
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # Scale features for training
```

---

### **Step 4: Training the Model**
Train the **XGBoost Classifier** with encoded data:
```python
from xgboost import XGBClassifier

# Define & train the model
model = XGBClassifier()
model.fit(X_train_encoded, y_train)  # Ensure X_train_encoded & y_train are defined
```

Save the trained model for future use:
```python
from joblib import dump
dump(model, "credit_risk_model.joblib")
```

---

## 🏦 **Prediction System**
### **Step 5: Credit Risk Evaluation**
The model predicts **default probability & credit score** using financial inputs.  
📊 **Credit Rating System**:

| Credit Score | Risk Level |
|-------------|------------|
| 300 - 500   | **Poor** |
| 500 - 650   | **Average** |
| 650 - 750   | **Good** |
| 750 - 900   | **Excellent** |

### **Example Prediction Code**
```python
import joblib
import numpy as np

# Load trained model
model = joblib.load("credit_risk_model.joblib")

# Sample customer data (modify as needed)
customer_data = np.array([[30, 1200000, 2500000, 24, 15, 25, 40, 2]])

# Make predictions
credit_risk = model.predict(customer_data)
print("Credit Risk Prediction:", "High Risk" if credit_risk[0] == 1 else "Low Risk")
```

---

## 🚀 **Next Steps**
✅ **Improve feature engineering** with deeper financial insights  
✅ **Deploy the model** for real-world credit evaluation  
✅ **Optimize parameters** for higher accuracy  

---

## 🤝 **Contributing**
If you'd like to improve this project:  
🔹 Fork the repository  
🔹 Work on enhancements  
🔹 Submit a pull request 🚀  

---






