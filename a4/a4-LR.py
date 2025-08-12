import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = "heart_attack_prediction_dataset_Original.xlsx"
df = pd.read_excel(file_path)

# Preprocess Blood Pressure: split and average
bp_split = df['Blood Pressure'].str.split('/', expand=True)
df['Systolic BP'] = pd.to_numeric(bp_split[0], errors='coerce')
df['Diastolic BP'] = pd.to_numeric(bp_split[1], errors='coerce')
df['BP_Avg'] = (df['Systolic BP'] + df['Diastolic BP']) / 2

# Select required features
features = ['Age', 'Cholesterol', 'BP_Avg', 'Previous Heart Problems']
df_selected = df[features + ['Heart Attack Risk']].dropna()

# Ensure numeric data
df_selected['Cholesterol'] = pd.to_numeric(df_selected['Cholesterol'], errors='coerce')
df_selected['Previous Heart Problems'] = pd.to_numeric(df_selected['Previous Heart Problems'], errors='coerce')
df_selected = df_selected.dropna()

# Define features and target
X = df_selected[features]
y = df_selected['Heart Attack Risk'].astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=24)

# Logistic Regression model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)

# Evaluation
print("\n----- Logistic Regression Results -----")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['No Attack', 'Attack'],
            yticklabels=['No Attack', 'Attack'])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
