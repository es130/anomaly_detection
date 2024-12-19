# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import shap

# Step 1: Load the Data
# Replace the path with your actual file location
data = pd.read_csv('~/Documents/LLP/llp.csv')

# Step 2: Data Exploration
print("First few rows of the dataset:")
print(data.head())

print("\nMissing values in each column:")
print(data.isnull().sum())

print("\nDataset Information:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

# Step 3: Data Preprocessing
# Convert transaction_date and transaction_timestamp to datetime
data['transaction_date'] = pd.to_datetime(data['transaction_date'], errors='coerce')
data['transaction_timestamp'] = pd.to_datetime(data['transaction_timestamp'], errors='coerce')

# Drop rows with invalid or missing dates
data = data.dropna(subset=['transaction_date', 'transaction_timestamp'])

# Step 4: Feature Engineering
# Add derived feature: amount_difference
data['amount_difference'] = data['credit_amount'] - data['debit_amount']

# Select numerical columns for scaling
features_to_scale = ['debit_amount', 'credit_amount', 'bank_charge', 'amount_difference']
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

print("\nScaled numerical columns:")
print(data[features_to_scale].head())

# Step 5: Anomaly Detection Using Isolation Forest
# Select features for the model
features = ['debit_amount', 'credit_amount', 'bank_charge', 'amount_difference']
X = data[features]

# Train Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
data['anomaly'] = model.fit_predict(X)

# Map anomaly labels: 1 -> Normal, -1 -> Anomaly
data['anomaly'] = data['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

# Step 6: Visualize Anomalies
print("\nFlagged Anomalies:")
print(data[data['anomaly'] == 'Anomaly'])

# Plot anomalies
plt.figure(figsize=(10, 6))
plt.scatter(data['debit_amount'], data['credit_amount'], 
            c=(data['anomaly'] == 'Anomaly'), cmap='coolwarm', alpha=0.5)
plt.xlabel('Debit Amount (Scaled)')
plt.ylabel('Credit Amount (Scaled)')
plt.title('Anomaly Detection in Transactions')
plt.show()

# Step 7: Explain Anomalies Using SHAP
print("\nExplaining anomalies using SHAP:")
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# SHAP Summary Plot
shap.summary_plot(shap_values, X)
