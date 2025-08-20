import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv("manufacturing_dataset_1000_samples.csv")
print("🔹 Original Data Shape:", df.shape)

# Drop Timestamp if present
if "Timestamp" in df.columns:
    df = df.drop("Timestamp", axis=1)

# Convert categorical (object/string) columns to numeric
df = pd.get_dummies(df, drop_first=True)

# Handle missing values (impute with median for numeric)
df = df.fillna(df.median(numeric_only=True))

print("🔹 Data Shape after Cleaning:", df.shape)
print(df.head())

# Features (X) and Target (y)
X = df.drop("Parts_Per_Hour", axis=1)
y = df["Parts_Per_Hour"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Model Evaluation")
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("🎉 Model trained & saved as model.pkl")
