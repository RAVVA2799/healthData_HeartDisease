import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Step 1: Load the CSV file
file_path = 'C:/Users/Microsoft/OneDrive/Desktop/pythonAssignments/mlProjects/heart_disease.csv'
 # Replace with your actual CSV file path
df = pd.read_csv(file_path)
df=df.iloc[:100]

# ✅ Step 2: Select Features and Target Variable
features = [ "Blood Pressure", "BMI", "Triglyceride Level", "Smoking", "Diabetes"]
target = "CRP Level"

# ✅ Step 3: Encode Categorical Variables
label_encoders = {}
for col in ["Smoking", "Diabetes"]:  # Convert categorical values into numerical labels
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Convert to string before encoding
    label_encoders[col] = le

# ✅ Step 4: Handle Missing Values
df[features] = df[features].fillna(df[features].mean())  # Fill missing values with mean
df[target] = df[target].fillna(df[target].mean())  # Fill missing values in target

# ✅ Step 5: Standardize Numerical Features
scaler = StandardScaler()
df[[ "Blood Pressure", "BMI", "Triglyceride Level"]] = scaler.fit_transform(df[[ "Blood Pressure", "BMI", "Triglyceride Level"]])

# ✅ Step 6: Split Data into Training & Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# ✅ Step 7: Train the Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# ✅ Step 8: Make Predictions
y_pred = lr_model.predict(X_test)

# ✅ Step 9: Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# ✅ Step 10: Plot Actual vs Predicted Cholesterol Levels
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color="blue", alpha=0.6)
plt.xlabel("Actual CRP Level")
plt.ylabel("Predicted CRP Level")
plt.title("Actual vs Predicted CRP Levels")
plt.axline((0, 0), slope=1, color="red", linestyle="--")  # Ideal perfect fit line
plt.show()

# ✅ Step 11: Plot Regression Line (Fitted Line)
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={"color": "blue", "alpha": 0.5}, line_kws={"color": "red"})
plt.xlabel("Actual CRP Level")
plt.ylabel("Predicted CRP Level")
plt.title("Regression Line for CRP Prediction")
plt.show()
