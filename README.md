Fraud Detection Model

Objective
The objective of this project is to proactively detect fraudulent financial transactions using machine learning techniques. With over 6.3M transactions, the model identifies patterns in transaction behavior to differentiate between legitimate and fraudulent activities.

Dataset
Size: 6,362,620 rows × 19 columns
Target Variable: isFraud (1 = Fraudulent, 0 = Legitimate)
Features: Transaction details including step, amount, old & new balances, destination/origin accounts, and transaction type.

Process
1. Data Cleaning & Preprocessing
Removed irrelevant identifiers (nameOrig, nameDest).
Handled missing values (none found).
Checked for outliers and applied transformations.
Scaled features using MinMaxScaler & StandardScaler (based on distribution).

2. Feature Engineering
Created derived features:
tx_per_hour → transaction frequency per account
amount_ratio_org, amount_ratio_dest → ratio of transaction to balance
Negative balance indicators

3. Model Building
Trained a Random Forest Classifier with:
n_estimators=200
class_weight='balanced' to handle class imbalance
n_jobs=-1 for parallel processing

4. Model Evaluation
Accuracy: 99.99%
ROC-AUC: 0.999
PR-AUC: 0.998
Best F1 Score Threshold: 0.645

Confusion Matrix @ best threshold:
[[1270881       0]
 [      4    1639]]

Key Insights
Most Important Features:
amount_ratio_org, newbalanceOrig, oldbalanceOrg, amount

Fraud is strongly linked to balance inconsistencies and transaction type (TRANSFER, CASH_OUT).

Conclusion
The model is highly effective in detecting fraud with near-perfect accuracy.
Findings can be leveraged to:
Flag suspicious transactions in real-time
Strengthen transaction monitoring systems
Reduce financial losses with proactive prevention strategies

Repository Structure
├── Fraud_Detection_Model.ipynb   # Jupyter Notebook with code & analysis
├── fraud_detection_model.pkl     # Trained Random Forest model
├── feature_names.json            # Stored feature names for prediction
├── README.md                     # Project documentation

Future Work
Deploy model via Flask / FastAPI API for real-time detection
Build interactive dashboards using Streamlit

Experiment with XGBoost / LightGBM for further improvements
