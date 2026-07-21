import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# Task 1: Data Understanding
# ==========================================
# 1. Load the dataset
df = pd.read_csv('insurance.csv')

# 2. Display the first five records
print("--- First 5 Records ---")
print(df.head())

# 3. Identify Variables
print("\n--- Variable Identification ---")
print("Numerical features: age, bmi, children")
print("Categorical features: sex, smoker, region")
print("Target variable: charges")

# ==========================================
# Task 2: Data Preprocessing
# ==========================================
# Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Encode categorical variables (sex, smoker, region)
# Using drop_first=True to avoid the dummy variable trap
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Split the dataset into 80% training and 20% testing
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# Task 3: Model Development
# ==========================================
# Build and train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the charges for the test dataset
y_pred = model.predict(X_test)

# ==========================================
# Task 4: Model Evaluation
# ==========================================
# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Create Actual vs Predicted scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Insurance Charges")
plt.grid(True)
plt.show()