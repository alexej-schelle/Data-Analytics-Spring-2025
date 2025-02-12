import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = "mba_decision_dataset.csv"  # Replace with your dataset file path
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Data Preprocessing
df = df.replace({'Male':1, 'Female':2, 'Economics':3, 'Science':4, 'Arts':5, 'Engineering':6, 'Business':7, 'Entrepreneur':8, 'Entrepreneurship':8.5 , 'Analyst':9, 'Engineer':10, 'Consultant':11, 'Manager':12, 'Yes':13, 'No':14, 'Employer':15, 'Loan':16, 'Scholarship':17, 'Self-funded':18, 'Executive':19, 'Marketing Director':20, 'Consutant':21, 'Startup Founder':22, 'Finance Manager':23, 'International':24, 'Domestic':25, 'Networking':26, 'Career Growth':27, 'Skill Enhancement':28, 'Entrepreneuership':29, 'On-Campus':30, 'Online':31, 'Yes':32, 'No':33, 'Other': 34})

# Define input features (X) and target variable (y)
X = df.drop(columns=["Undergraduate GPA"])  # Features
y = df["Undergraduate GPA"]  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVM model
svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svm_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create SVM Model

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Predict on the test set
y_pred = svm_model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the evaluation results
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Define a single new data point
new_data = [[7, 14, 3, 0, 25.0, 50.0, 2000, 50, 0.7, 0.5, 0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]]

# Scale the new data using the same scaler as the training set
new_data_scaled = scaler.transform(new_data)

# Predict the EnergyConsumption for the new data point
predicted_value = svm_model.predict(new_data_scaled)

print(f"Predicted GPA: {predicted_value[0]:.2f}")

# Create Random Forst Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

print(' ')
print(' ')

# Create and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Calculate metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print the evaluation results
print("Random Forest Model Performance:")
print(f"Mean Squared Error (MSE): {mse_rf:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf:.2f}")
print(f"R-squared (R²): {r2_rf:.2f}")

# Define a single new data point
new_data_rf = [[7, 14, 3, 0, 25.0, 50.0, 2000, 50, 0.7, 0.5, 0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]]

# Predict the EnergyConsumption for the new data point using the Random Forest model
predicted_value_rf = rf_model.predict(new_data_rf)

print(f"Predicted GPA: {predicted_value_rf[0]:.2f}")

print(' ')
print(' ')

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Create and train the K-Nearest Neighbors Regressor
knn_model = KNeighborsRegressor(n_neighbors=5)  # Default: k=5
knn_model.fit(X_train_scaled, y_train)  # Use scaled data for KNN

# Predict on the test set
y_pred_knn = knn_model.predict(X_test_scaled)

# Calculate metrics
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test, y_pred_knn)

# Print the evaluation results
print("K-Nearest Neighbors Model Performance:")
print(f"Mean Squared Error (MSE): {mse_knn:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_knn:.2f}")
print(f"R-squared (R²): {r2_knn:.2f}")

# Define a single new data point
new_data_knn = [[7, 14, 3, 0, 25.0, 50.0, 2000, 50, 0.7, 0.5, 0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]]

# Scale the new data using the same scaler
new_data_knn_scaled = scaler.transform(new_data_knn)

# Predict the EnergyConsumption for the new data point
predicted_value_knn = knn_model.predict(new_data_knn_scaled)

print(f"Predicted GPA: {predicted_value_knn[0]:.2f}")

print(' ')
print(' ')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Create and train the Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_gb = gb_model.predict(X_test)

# Calculate metrics
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

# Print the evaluation results
print("Gradient Boosting Model Performance:")
print(f"Mean Squared Error (MSE): {mse_gb:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_gb:.2f}")
print(f"R-squared (R²): {r2_gb:.2f}")

# Define a single new data point (example values)
new_data_gb = [[7, 14, 3, 0, 25.0, 50.0, 2000, 50, 0.7, 0.5, 0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]]

# Predict the EnergyConsumption for the new data point using the Gradient Boosting model
predicted_value_gb = gb_model.predict(new_data_gb)

# Print the predicted value
print(f"Predicted GPA: {predicted_value_gb[0]:.2f}")

print(' ')
print(' ')

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Create and train the Kernel Ridge Regression model
kr_model = KernelRidge(alpha=1.0, kernel='rbf')  # Use Radial Basis Function (RBF) kernel
kr_model.fit(X_train_scaled, y_train)  # Make sure to scale data for kernel methods

# Predict on the test set
y_pred_kr = kr_model.predict(X_test_scaled)

# Calculate metrics
mse_kr = mean_squared_error(y_test, y_pred_kr)
rmse_kr = np.sqrt(mse_kr)
r2_kr = r2_score(y_test, y_pred_kr)

# Print the evaluation results
print("Kernel Ridge Regression Model Performance:")
print(f"Mean Squared Error (MSE): {mse_kr:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_kr:.2f}")
print(f"R-squared (R²): {r2_kr:.2f}")

print(' ')
print(' ')

# Define a single new data point (example values)
new_data_kr = [[7, 14, 3, 0, 25.0, 50.0, 2000, 50, 0.7, 0.5, 0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]]

# Scale the new data using the same scaler (important for kernel methods)
new_data_kr_scaled = scaler.transform(new_data_kr)

# Predict the EnergyConsumption for the new data point using the Kernel Ridge Regression model
predicted_value_kr = kr_model.predict(new_data_kr_scaled)

# Print the predicted value
print(f"Predicted GPA: {predicted_value_kr[0]:.2f}")

print(' ')
print(' ')

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Create and train the Stochastic Gradient Descent Regressor model
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd_model.fit(X_train_scaled, y_train)  # Make sure to scale the data as SGD is sensitive to feature scaling

# Predict on the test set
y_pred_sgd = sgd_model.predict(X_test_scaled)

# Calculate metrics
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
rmse_sgd = np.sqrt(mse_sgd)
r2_sgd = r2_score(y_test, y_pred_sgd)

# Print the evaluation results
print("Stochastic Gradient Descent Model Performance:")
print(f"Mean Squared Error (MSE): {mse_sgd:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_sgd:.2f}")
print(f"R-squared (R²): {r2_sgd:.2f}")

# Define a single new data point (example values)
new_data_sgd = [[7, 14, 3, 0, 25.0, 50.0, 2000, 50, 0.7, 0.5, 0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]]

# Scale the new data using the same scaler (important for SGD)
new_data_sgd_scaled = scaler.transform(new_data_sgd)

# Predict the EnergyConsumption for the new data point using the SGD model
predicted_value_sgd = sgd_model.predict(new_data_sgd_scaled)

# Print the predicted value
print(f"Predicted GPA: {predicted_value_sgd[0]:.2f}")

print(' ')
print(' ')
