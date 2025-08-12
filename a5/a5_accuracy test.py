import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder           
from sklearn.linear_model import LogisticRegression      # C) Logistic Regression
from sklearn.neighbors import KNeighborsClassifier       # D) KNN
from sklearn.svm import SVC                              # E&F) SVM
from sklearn.naive_bayes import GaussianNB               # G) Naive Bayes
from sklearn.tree import DecisionTreeClassifier          # H) Decision Tree
from sklearn.ensemble import RandomForestClassifier      # I) Random Forest
from sklearn.metrics import accuracy_score, classification_report

import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load your dataset (already loaded in your session)
train_df = pd.read_csv("train.csv")

# Data preparation
X = train_df.drop(columns=['subject', 'Activity'])  # Feature matrix
y = train_df['Activity']                            # Target vector

# Encode target labels (Activity)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Linear SVM": SVC(kernel='linear'),
    "Non-Linear SVM (RBF)": SVC(kernel='rbf'),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# Dictionary to store accuracy results
accuracy_summary = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    accuracy_summary[name] = acc

    # Optional: print detailed classification report
    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

# Create a DataFrame to display the summary table
accuracy_df = pd.DataFrame.from_dict(accuracy_summary, orient='index', columns=['Accuracy'])
accuracy_df = accuracy_df.sort_values(by='Accuracy', ascending=False)

# Print summary table
print("\nSummary of Model Accuracies:")
print(accuracy_df)