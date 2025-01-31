import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the re-uploaded heart disease dataset

file_path = 'C:/Users/Microsoft/OneDrive/Desktop/pythonAssignments/mlProjects/heart_disease.csv'
heart_data = pd.read_csv(file_path)
heart_data = heart_data.iloc[:1000]
# Display the first few rows and dataset information to understand its structure
# heart_data.head(), heart_data.info()


# Select relevant features
features = ["Age", "Blood Pressure", "Cholesterol Level", "BMI", "Triglyceride Level"]

# Check for missing values
print("Missing values in each column:\n", heart_data[features].isnull().sum())

# Handle missing values (Option: Fill with mean)
heart_data[features] = heart_data[features].fillna(heart_data[features].mean())

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(heart_data[features])

# # Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
heart_data["Cluster"] = kmeans.fit_predict(scaled_data)

# Plot clustering results
plt.figure(figsize=(10, 6))
for cluster in heart_data["Cluster"].unique():
    cluster_data = heart_data[heart_data["Cluster"] == cluster]
    plt.scatter(cluster_data["Age"], cluster_data["Cholesterol Level"], label=f"Cluster {cluster}")

# Add cluster centers to the plot
centers = kmeans.cluster_centers_
scaled_centers = scaler.inverse_transform(centers)
plt.scatter(scaled_centers[:, 0], scaled_centers[:, 2], c="black", marker="x", s=200, label="Centers")

# Add labels and legend
plt.title("Clustering of Patients Based on Health Parameters")
plt.xlabel("Age")
plt.ylabel("Cholesterol Level")
plt.legend()
plt.show()

# or irrelevant columns and use 'Heart Disease Status' as the hue

sns.pairplot(heart_data[["Fasting Blood Sugar", "Blood Pressure", "Cholesterol Level", "BMI", "Triglyceride Level", "Stress Level"]], 
             hue="Stress Level")


# Display the plot
plt.show()