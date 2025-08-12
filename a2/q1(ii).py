import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer

# ------------------- QUESTION 1: House Price Prediction -------------------

# Load dataset for house price prediction
data_house = pd.read_csv('housing.csv')  # Ensure correct file path

# Handling missing values
data_house.fillna(data_house.median(numeric_only=True), inplace=True)

# Separate features and target variable
y_house = data_house['median_house_value']
X_house = data_house.drop(columns=['median_house_value'])

# Identify categorical and numerical columns
categorical_cols = ['ocean_proximity'] if 'ocean_proximity' in X_house.columns else []
numeric_cols = X_house.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Apply OneHotEncoder to categorical variables and leave numeric columns unchanged
ct = ColumnTransformer([
    ('encoder', OneHotEncoder(drop='first'), categorical_cols)
], remainder='passthrough')

X_transformed = ct.fit_transform(X_house)
X_house = pd.DataFrame(X_transformed, columns=ct.get_feature_names_out())

# Split into training and testing sets
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_house, y_house, test_size=0.2, random_state=42)

# Standardize features
scaler_h = StandardScaler()
X_train_h_scaled = scaler_h.fit_transform(X_train_h)
X_test_h_scaled = scaler_h.transform(X_test_h)

# --- Multiple Linear Regression ---
lr_h = LinearRegression()
lr_h.fit(X_train_h_scaled, y_train_h)
y_pred_lr_h = lr_h.predict(X_test_h_scaled)
print("House Price - Linear Regression R2 Score:", r2_score(y_test_h, y_pred_lr_h))

# --- Polynomial Regression ---
poly_h = PolynomialFeatures(degree=2)
X_train_h_poly = poly_h.fit_transform(X_train_h_scaled)
X_test_h_poly = poly_h.transform(X_test_h_scaled)
lr_poly_h = LinearRegression()
lr_poly_h.fit(X_train_h_poly, y_train_h)
y_pred_poly_h = lr_poly_h.predict(X_test_h_poly)
print("House Price - Polynomial Regression R2 Score:", r2_score(y_test_h, y_pred_poly_h))

# --- Support Vector Regression ---
svr_h = SVR(kernel='rbf')
svr_h.fit(X_train_h_scaled, y_train_h)
y_pred_svr_h = svr_h.predict(X_test_h_scaled)
print("House Price - SVR R2 Score:", r2_score(y_test_h, y_pred_svr_h))

# --- Plot Predictions vs Actual Values ---
def plot_results(y_test, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.show()

plot_results(y_test_h, y_pred_lr_h, "Linear Regression: Actual vs Predicted")
plot_results(y_test_h, y_pred_poly_h, "Polynomial Regression: Actual vs Predicted")
plot_results(y_test_h, y_pred_svr_h, "SVR: Actual vs Predicted")
