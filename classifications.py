import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = 'C:/Users/Microsoft/OneDrive/Desktop/pythonAssignments/mlProjects/heart_disease.csv'  # Replace with actual path
heart_data = pd.read_csv(file_path)


# Apply the formula logic using pandas
heart_data["Heart Problem"] = heart_data.apply(lambda x: "High" if x["High Blood Pressure"] == "Yes" and x["High LDL Cholesterol"] == "Yes" 
                    else ("Low" if x["High Blood Pressure"] == "No" and x["High LDL Cholesterol"] == "No" else "Medium"), axis=1)


heart_data["Smoking_Drinking_Effects"] = heart_data.apply(lambda x: "Very High" if x["Smoking"] == "Yes" and x["Alcohol Consumption"] == "High" 
                    else ("High" if x["Smoking"] == "Yes" and x["Alcohol Consumption"] == "Medium" else ("Low" if x["Smoking"] =="No"  and x["Alcohol Consumption"] == "Low" else "Medium")), axis=1)

# Save the updated csv  file
heart_data.to_csv("updated_HEART.csv", index=False)

# Select relevant features and target
features = ["Age", "Gender", "Blood Pressure", "Cholesterol Level", "Smoking", "Family Heart Disease", 
            "BMI", "Sleep Hours", "Alcohol Consumption", "Exercise Habits", "CRP Level"]
target = "Stress Level"

# Step 1: Merge Stress Level 3 into 2**
heart_data["Stress Level"] = heart_data["Stress Level"].replace(3, 2)  # Converts all "3" into "2"

# Step 2: Convert Numeric Columns to Correct Data Type**
numeric_cols = ["Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours", "CRP Level"]
heart_data[numeric_cols] = heart_data[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Step 3: Encode Categorical Features**
categorical_cols = ["Gender", "Smoking", "Family Heart Disease", "Alcohol Consumption", "Exercise Habits"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    heart_data[col] = le.fit_transform(heart_data[col].astype(str))  # Convert to string before encoding
    label_encoders[col] = le  # Store encoder

# Step 4: Encode Target Variable (Stress Level)**
le_target = LabelEncoder()
heart_data[target] = le_target.fit_transform(heart_data[target].astype(str))

# Step 5: Handle Missing Values**
heart_data[features] = heart_data[features].fillna(heart_data[features].mean())

# Step 6: Standardize Numerical Features**
scaler = StandardScaler()
heart_data[numeric_cols] = scaler.fit_transform(heart_data[numeric_cols])

# Step 7: Split Data into Training and Testing Sets**
X_train, X_test, y_train, y_test = train_test_split(heart_data[features], heart_data[target], test_size=0.2, random_state=42)

# Step 8: Apply SMOTE to Balance Classes**
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 9: Train Models**
log_reg = LogisticRegression()
dtree = DecisionTreeClassifier()
rf = RandomForestClassifier()

log_reg.fit(X_train_resampled, y_train_resampled)
dtree.fit(X_train_resampled, y_train_resampled)
rf.fit(X_train_resampled, y_train_resampled)

# Step 10: Make Predictions**
y_pred_log = log_reg.predict(X_test)
y_pred_tree = dtree.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Step 11: Evaluate Models**
models = {
    "Logistic Regression": (y_pred_log, log_reg),
    "Decision Tree": (y_pred_tree, dtree),
    "Random Forest": (y_pred_rf, rf),
}

heart_problem_counts =heart_data["Heart Problem"].value_counts()
smoking_alcohol_problem =heart_data["Smoking_Drinking_Effects"].value_counts()

# ✅ Step 3: Display the results
print("Count of High, Medium, and Low values in 'Heart Problem' column:")
print(heart_problem_counts)

print("Count of High, Medium, and Low values in 'Heart Problem' due to Addiction:")
print(smoking_alcohol_problem)

for name, (y_pred, model) in models.items():

    print(f"\n{name} Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # ✅ **Step 12: Plot Confusion Matrix**
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(heart_data[target].unique()), 
                yticklabels=sorted(heart_data[target].unique()))
    plt.xlabel("Stress Level  - HIGH/MEDIUM/LOW")
    plt.ylabel("TOTAL STRESSED PEOPL DUE TO HEART DISEASE")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()




# ✅ Step 3: Plot Pie Chart
plt.figure(figsize=(8, 6))  # Set figure size
colors = [ "orange", "green","red"] 
# Colors for High, Medium, Low
explode = (0.02, 0.02, 0.02)  # Slightly separate "High" for emphasis

# Create the pie chart
plt.pie(heart_problem_counts, labels=heart_problem_counts.index, autopct="%1.1f%%", colors=colors, explode=explode, startangle=140)

# ✅ Step 4: Add Legend on the Right Side
plt.legend(title="Heart Problem Levels", loc="center left", bbox_to_anchor=(1, 0.5))

# ✅ Step 5: Add Title
plt.title("Severity of Heart Problem based on BP/Cholestrol Levels")

# ✅ Step 6: Show the plot
plt.show()

# ✅ Step 3: Plot Pie Chart
plt.figure(figsize=(8, 6))  # Set figure size
colors2 = [ "yellow","orange","red","green"] 
# Colors for High, Medium, Low
explode2 = (0.02, 0.02, 0.02,0.02)  # Slightly separate "High" for emphasis

# Create the pie chart
plt.pie(smoking_alcohol_problem, labels=smoking_alcohol_problem.index, autopct="%1.1f%%", colors=colors2, explode=explode2, startangle=140)

# ✅ Step 4: Add Legend on the Right Side
plt.legend(title="Heart Problem Levels", loc="center left", bbox_to_anchor=(1, 0.5))

# ✅ Step 5: Add Title
plt.title("Severity of Heart Problem based on Smoking/Alcohol Consumption")

# ✅ Step 6: Show the plot
plt.show()