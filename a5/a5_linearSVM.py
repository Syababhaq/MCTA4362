import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
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
X_train = train_df.drop(columns=['subject', 'Activity'])
y_train = train_df['Activity']

X_test = test_df.drop(columns=['subject', 'Activity'], errors='ignore')
y_test = test_df['Activity'] if 'Activity' in test_df.columns else None

# ---------------------
# 3. Encode target labels
# ---------------------
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

if y_test is not None:
    y_test_encoded = le.transform(y_test)

# ---------------------
# 4. Train Linear SVM
# ---------------------
model = SVC(kernel='linear')
model.fit(X_train, y_train_encoded)

# ---------------------
# 5. Predict on test data
# ---------------------
y_pred_encoded = model.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred_encoded)

# ---------------------
# 6. Output Results
# ---------------------
print("✅ Prediction complete.")
print("First 10 predicted activities:")
print(y_pred_labels[:10])

# Evaluate if y_test is available
if y_test is not None:
    print("\n📊 Evaluation Metrics:")
    print(f"Accuracy: {accuracy_score(y_test_encoded, y_pred_encoded):.4f}")
    print(classification_report(y_test_encoded, y_pred_encoded, target_names=le.classes_))

# ---------------------
# 7. Save the model and label encoder
# ---------------------
joblib.dump(model, 'har_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("\n💾 Model and LabelEncoder saved as 'har_model.pkl' and 'label_encoder.pkl'")
