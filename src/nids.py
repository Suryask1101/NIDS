import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# Define dataset path
dataset_path = "C:/SK/py-project/Intrusion-demo-2/data/preprocessed_data.csv"

# Load dataset
if os.path.exists(dataset_path):
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    print("Dataset Loaded Successfully!")
else:
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

# Check for missing values
df.dropna(inplace=True)

# Encode labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Feature selection (remove label column for training)
X = df.drop(columns=['Label'])
y = df['Label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Save model
model_path = "C:/SK/py-project/Intrusion-demo-2/models/nids_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved at {model_path}")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
