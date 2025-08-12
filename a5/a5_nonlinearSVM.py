import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# ---------------------
# 1. Load datasets
# ---------------------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# ---------------------
# 2. Prepare features and labels
# ---------------------
# Training data
X_train = train_df.drop(columns=['subject', 'Activity'])
y_train = train_df['Activity']

# Test data: if Activity exists, keep it for evaluation
X_test = test_df.drop(columns=['subject', 'Activity'], errors='ignore')
y_test = test_df['Activity'] if 'Activity' in test_df.columns else None

# ---------------------
# 3. Encode target labels
# ---------------------
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# ---------------------
# 4. Train Non-Linear SVM
# ---------------------
model = SVC(kernel='rbf')
model.fit(X_train, y_train_encoded)

# ---------------------
# 5. Predict on test data
# ---------------------
y_test_pred_encoded = model.predict(X_test)
y_test_pred_labels = le.inverse_transform(y_test_pred_encoded)

# ---------------------
# 6. Output Results
# ---------------------
print("✅ Prediction complete.")
print("First 10 predicted activities:")
print(y_test_pred_labels[:10])

import joblib

# Train your model (Random Forest is a good choice)
model = SVC(kernel='rbf')
model.fit(X_train, y_train_encoded)

# Save model and label encoder
joblib.dump(model, 'har_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

