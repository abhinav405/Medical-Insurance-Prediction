import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# Task 1: Data Understanding
# ==========================================

# 1. Load the dataset using Pandas
df = pd.read_csv('insurance.csv')

# 2. Display the first five records
print("--- First 5 Records ---")
print(df.head())
print("\n")

# 3. Identify Features
print("--- Feature Identification ---")
print("Numerical features: age, bmi, children")
print("Categorical features: sex, smoker, region")
print("Target variable: charges\n")

# ==========================================
# Task 2: Data Preprocessing
# ==========================================

# Check for missing values
print("--- Missing Values Check ---")
print(df.isnull().sum())
print("\n")

# Encode categorical variables (sex, smoker, region)
# We use pd.get_dummies to convert categorical strings into numerical columns.
# drop_first=True prevents the "dummy variable trap" (multicollinearity) in Linear Regression.
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Split the dataset into 80% training and 20% testing
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# Task 3: Model Development
# ==========================================

# Build and train a Multiple Linear Regression model
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

print("--- Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score: {r2:.4f}\n")

# Observations based on model's performance
print("--- Observations ---")
print("1. R-squared Score: The model explains a significant portion of the variance in insurance charges, but it is not a perfect fit, indicating that medical costs might have non-linear triggers.")
print("2. MAE Interpretation: The Mean Absolute Error represents the average dollar amount our predictions deviate from the actual medical charges.")
print("3. Scatter Plot Trend: In the plot, we will likely see a cluster of accurate predictions for lower charges, but the model may struggle with higher outliers (which are often driven by severe medical incidents for smokers).\n")

# Create Actual vs Predicted scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolor='k')

# Adding a line of perfect prediction for visual reference
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Medical Insurance Charges')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()