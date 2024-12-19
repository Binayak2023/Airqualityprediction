import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 1. Logging Configuration
# ================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================================
# 2. Load the Dataset
# ================================

data_path = "data/beijing_air_quality.csv"
df = pd.read_csv(data_path)

logging.info(f"Dataset columns: {df.columns}")

# ================================
# 3. Data Preprocessing
# ================================

# Drop unnecessary columns and rows with missing target values
df.dropna(subset=['PM2.5'], inplace=True)

# Select relevant features and target
selected_features = ['SO2', 'NO2', 'CO', 'O3', 'PM10', 'TEMP', 'WSPM']
X = df[selected_features]
y_regression = df['PM2.5']

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# ================================
# 4. Data Splitting
# ================================

X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X_imputed, y_regression, test_size=0.3, random_state=42
)

# ================================
# 5. Scaling the Data
# ================================

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_reg_scaled = scaler_y.fit_transform(y_train_reg.values.reshape(-1, 1))

# ================================
# 6. Visualization: PM2.5 Distribution
# ================================

plt.figure(figsize=(8, 6))
sns.histplot(y_train_reg, bins=30, kde=True, color='blue')
plt.title("Distribution of PM2.5 Values")
plt.xlabel("PM2.5")
plt.ylabel("Frequency")
plt.savefig("visualizations/pm25_distribution.png")
plt.close()

# ================================
# 7. Gaussian Process Regression
# ================================

logging.info("Training Gaussian Process Regressor on a subset...")

# Reduce dataset size for training to avoid memory issues
subset_size = 2000
X_train_subset = X_train_scaled[:subset_size]
y_train_subset_reg = y_train_reg_scaled[:subset_size]

# Define the kernel for Gaussian Process Regression
kernel_reg = C(1.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e2))

# Train the Gaussian Process Regressor
gp_regressor = GaussianProcessRegressor(kernel=kernel_reg, n_restarts_optimizer=2)
gp_regressor.fit(X_train_subset, y_train_subset_reg.ravel())

# Predict and evaluate
y_pred_reg_scaled = gp_regressor.predict(X_test_scaled)
y_pred_reg = scaler_y.inverse_transform(y_pred_reg_scaled.reshape(-1, 1))

mse = mean_squared_error(y_test_reg, y_pred_reg)
logging.info(f"Mean Squared Error (Regression): {mse:.2f}")

# ================================
# 8. Visualization: Actual vs Predicted PM2.5
# ================================

plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5, color='green')
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted PM2.5")
plt.savefig("visualizations/actual_vs_predicted_pm25.png")
plt.close()

# ================================
# 9. Gaussian Process Classification
# ================================

# Define threshold to categorize 'Good' and 'Poor' air quality
threshold = 75
y_train_class = ['Good' if val <= threshold else 'Poor' for val in y_train_reg]
y_test_class = ['Good' if val <= threshold else 'Poor' for val in y_test_reg]

logging.info("Training Gaussian Process Classifier on a subset...")

# Train Gaussian Process Classifier on a subset
gp_classifier = GaussianProcessClassifier(kernel=RBF(length_scale=1.0), n_restarts_optimizer=2)
gp_classifier.fit(X_train_subset, y_train_class[:subset_size])

# Predict and evaluate
y_pred_class = gp_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test_class, y_pred_class)
logging.info(f"Accuracy (Classification): {accuracy:.4f}")
logging.info("\nClassification Report:\n" + classification_report(y_test_class, y_pred_class))

# ================================
# 10. Visualization: Confusion Matrix
# ================================

conf_matrix = confusion_matrix(y_test_class, y_pred_class)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=['Good', 'Poor'], yticklabels=['Good', 'Poor'])
plt.title('Confusion Matrix for Air Quality Classification')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("visualizations/confusion_matrix.png")
plt.close()

# ================================
# 11. Visualization: Feature Correlation Heatmap
# ================================

plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame(X_imputed, columns=selected_features).corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig("visualizations/feature_correlation.png")
plt.close()

# ================================
# 12. Save Models and Scalers
# ================================

joblib.dump(gp_regressor, "models/gp_regressor.pkl")
joblib.dump(gp_classifier, "models/gp_classifier.pkl")
joblib.dump(scaler_X, "models/scaler_X.pkl")
joblib.dump(scaler_y, "models/scaler_y.pkl")
joblib.dump(imputer, "models/imputer.pkl")

logging.info("Models and scalers saved successfully.")
