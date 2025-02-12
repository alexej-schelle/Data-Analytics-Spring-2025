import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "mba_decision_dataset.csv"  # Replace with your dataset file path
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Step 1: Data Preprocessing
df = df.replace({'Male':1, 'Female':2, 'Economics':3, 'Science':4, 'Arts':5, 'Engineering':6, 'Business':7, 'Entrepreneur':8, 'Entrepreneurship':8.5 , 'Analyst':9, 'Engineer':10, 'Consultant':11, 'Manager':12, 'Yes':13, 'No':14, 'Employer':15, 'Loan':16, 'Scholarship':17, 'Self-funded':18, 'Executive':19, 'Marketing Director':20, 'Consutant':21, 'Startup Founder':22, 'Finance Manager':23, 'International':24, 'Domestic':25, 'Networking':26, 'Career Growth':27, 'Skill Enhancement':28, 'Entrepreneuership':29, 'On-Campus':30, 'Online':31, 'Yes':32, 'No':33, 'Other': 34})

# Step 2: Verify Data
print(df.head())

# Step 3: Model Processing

# Define input features (X) and target variable (y)
X = df.drop(columns=["Undergraduate GPA"])  # Features
y = df["Undergraduate GPA"]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features for better performance
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Build the TensorFlow Model
model = Sequential([
    Dense(2, input_dim=X_train.shape[1], activation="relu"),  # Input layer with 128 neurons
    Dense(1064, activation="relu"),                              # Hidden layer with 64 neurons
    Dense(2, activation="linear")                              # Output layer (1 neuron for regression)
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])  # Loss: Mean Squared Error, Metric: Mean Absolute Error

# Step 3: Train the Model
history = model.fit(X_train, y_train, epochs=500, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Step 4: Evaluate the Model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {loss:.4f}")
print(f"Test Mean Absolute Error (MAE): {mae:.4f}")

new_data = [[7, 14, 3, 0, 25.0, 50.0, 2000, 50, 0.7, 0.5, 0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]]

# Step 5: Make Predictions
predictions = model.predict(new_data)
print("Predictions for GDA Grading:")
print(predictions[:100])  # Print the first 10 predictions

