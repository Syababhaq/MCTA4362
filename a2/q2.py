import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = "SolarPowerGenerationKaggle_MissingData.xlsx"
df = pd.read_excel(file_path, sheet_name="spg")

# Select relevant features
selected_features = [
    "shortwave_radiation_backwards_sfc", "total_cloud_cover_sfc",
    "high_cloud_cover_high_cld_lay", "medium_cloud_cover_mid_cld_lay", "low_cloud_cover_low_cld_lay",
    "temperature_2_m_above_gnd", "relative_humidity_2_m_above_gnd", "mean_sea_level_pressure_MSL",
    "total_precipitation_sfc", "snowfall_amount_sfc", "wind_speed_10_m_above_gnd",
    "wind_speed_80_m_above_gnd", "wind_speed_900_mb", "wind_direction_10_m_above_gnd",
    "wind_direction_80_m_above_gnd", "wind_direction_900_mb", "angle_of_incidence",
    "zenith", "azimuth"
]

df_selected = df[selected_features + ["generated_power_kw"]]

# Handle missing values with mean imputation
imputer = SimpleImputer(strategy="mean")
df_imputed = pd.DataFrame(imputer.fit_transform(df_selected), columns=df_selected.columns)

# Scale data using Min-Max Scaling
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_selected.columns)

# Split data into features (X) and target variable (y)
X = df_scaled.drop(columns=["generated_power_kw"])
y = df_scaled["generated_power_kw"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Multiple Linear Regression ---
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)
y_pred_mlr = mlr_model.predict(X_test)
mae_mlr = mean_absolute_error(y_test, y_pred_mlr)
mse_mlr = mean_squared_error(y_test, y_pred_mlr)
r2_mlr = r2_score(y_test, y_pred_mlr)

# --- Polynomial Regression (Degree 2) ---
poly_degree = 2
poly_model = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# --- Support Vector Regression (SVR) ---
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

# Print model evaluation results
print("Multiple Linear Regression:", f"MAE: {mae_mlr:.4f}, MSE: {mse_mlr:.4f}, R2: {r2_mlr:.4f}")
print("Polynomial Regression:", f"MAE: {mae_poly:.4f}, MSE: {mse_poly:.4f}, R2: {r2_poly:.4f}")
print("Support Vector Regression:", f"MAE: {mae_svr:.4f}, MSE: {mse_svr:.4f}, R2: {r2_svr:.4f}")

# --- Plot Predictions vs Actual Values ---
def plot_results(y_test, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.show()

plot_results(y_test, y_pred_mlr, "Linear Regression: Actual vs Predicted")
plot_results(y_test, y_pred_poly, "Polynomial Regression: Actual vs Predicted")
plot_results(y_test, y_pred_svr, "SVR: Actual vs Predicted")
