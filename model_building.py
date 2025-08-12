import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("bank-additional-full.csv", sep=";")

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != "y":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Split into features and target
X = df.drop(columns=["y"])
y = df["y"]

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X, y)

# Save model and encoders
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/decision_tree_model.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")
joblib.dump(list(X.columns), "model/features.pkl")

print("✅ Decision Tree model trained and saved successfully (with 'duration').")

