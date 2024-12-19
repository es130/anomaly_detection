import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import shap

data = pd.read_csv('~/Documents/LLP/llp.csv')

# exploring data set
print("First few rows of the dataset:")
print(data.head())

# Checking for any missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Gets info about the data set
print("\nDataset Information:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

#Data Preprocessing
# Convert transaction_date and transaction_timestamp to datetime
data['transaction_date'] = pd.to_datetime(data['transaction_date'], errors='coerce')
data['transaction_timestamp'] = pd.to_datetime(data['transaction_timestamp'], errors='coerce')

# Removing rows with invalid or missing dates
data = data.dropna(subset=['transaction_date', 'transaction_timestamp'])

# New feature: amount_difference
data['amount_difference'] = data['credit_amount'] - data['debit_amount'] #net funds in transaction

# scaling
features_to_scale = ['debit_amount', 'credit_amount', 'bank_charge', 'amount_difference']
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

print("\nScaled numerical columns:")
print(data[features_to_scale].head())

# These are the features most relevant for finding anomalies
features = ['debit_amount', 'credit_amount', 'bank_charge', 'amount_difference']
X = data[features]

# Anomaly Detection Using Isolation Forest (at 5%)
model = IsolationForest(contamination=0.05, random_state=42)
data['anomaly'] = model.fit_predict(X)

# 1 = normal, -1 = anomaly
data['anomaly'] = data['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

#Actually showing the rows that are flagged as anomalies
print("\nFlagged Anomalies:")
print(data[data['anomaly'] == 'Anomaly'])

#Making the scatter plot to show anomalies against normal transactions
plt.figure(figsize=(10, 6))
plt.scatter(data['debit_amount'], data['credit_amount'], 
            c=(data['anomaly'] == 'Anomaly'), cmap='coolwarm', alpha=0.5)
plt.xlabel('Debit Amount (Scaled)')
plt.ylabel('Credit Amount (Scaled)')
plt.title('Anomaly Detection in Transactions')
plt.show()

# Explain anomalies using SHAP
# SHAP helps explain the model’s anomaly detection decisions
print("\nExplaining anomalies using SHAP:")
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# SHAP Summary Plot
shap.summary_plot(shap_values, X)
