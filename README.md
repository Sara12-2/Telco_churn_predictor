# 📊 Telco Customer Churn Predictor (GUI Version)

This project is a **Customer Churn Prediction System** built with:
- **Python (Tkinter GUI)**
- **LightGBM Classifier**
- **SMOTE for handling imbalance**

It allows you to **train a churn prediction model** using a Telco customer dataset and then make **real-time churn predictions** by entering minimal customer details.

---

## 🚀 Features
- Load your **Telco Customer Dataset (CSV)** and train model instantly.
- Uses **LightGBM** for high-performance classification.
- Handles **imbalanced dataset** with SMOTE oversampling.
- Evaluate model with:
  - Accuracy
  - ROC-AUC Score
  - Precision, Recall, F1-score
- Predict customer churn using just **4 inputs**:
  - tenure  
  - MonthlyCharges  
  - TotalCharges  
  - SeniorCitizen (0 = No, 1 = Yes)

---

## 📦 Requirements
Install dependencies before running:

```bash
pip install pandas scikit-learn imbalanced-learn lightgbm
```
## 🛠️ How to Run

Clone or download this project.

Open terminal in the project folder.

Run:
```bash
python churn_gui.py
```

GUI will open.

## 📂 Dataset

The dataset must include the following columns:

customerID

gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,

MultipleLines, InternetService, Contract, PaymentMethod,

MonthlyCharges, TotalCharges

Churn (Yes/No)

👉 Example dataset: Telco Customer Churn Dataset
## 🎯 Usage

Click 📂 Load CSV & Train Model → select Telco dataset.

Model will be trained and evaluation metrics displayed.

Enter new customer details in input boxes:

tenure → number of months customer has stayed.

MonthlyCharges → monthly billing amount.

TotalCharges → total amount paid till now.

SeniorCitizen → 0 = No, 1 = Yes.

Click 🔍 Predict Churn.

Result will show:

Prediction (Churn or No Churn)

Probability score.

## 🧪 Example Input / Output

### Input:
```bash
tenure = 2
MonthlyCharges = 85
TotalCharges = 170
SeniorCitizen = 1
```
### Output:
```bash
🔮 Prediction: Churn
Probability: 0.67


```
