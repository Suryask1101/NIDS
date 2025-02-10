import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 📂 Get the absolute path of the project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 📥 Load preprocessed dataset
DATA_FILE = os.path.join(DATA_DIR, "preprocessed_data.csv")
if not os.path.exists(DATA_FILE):
    print(f"❌ ERROR: Preprocessed data not found -> {DATA_FILE}")
    exit()

df = pd.read_csv(DATA_FILE)

# 🔎 Display initial shape
print(f"✅ Data Loaded! Shape: {df.shape}")

# 🏷️ **Detect Label Column Automatically**
label_column = None
for col in df.columns:
    if "label" in col.lower():  # Case-insensitive check
        label_column = col
        break

if not label_column:
    print(f"❌ ERROR: No label column found. Available columns: {df.columns}")
    exit()

print(f"✅ Using label column: {label_column}")

# 🛑 **Feature Selection**
X = df.drop(columns=[label_column])  # Features
y = df[label_column]  # Target

# 🎯 **Train-Test Split (80-20)**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🔄 **Feature Scaling**
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🤖 **Model Training (Random Forest)**
print("🚀 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # Parallel training
model.fit(X_train_scaled, y_train)

# 📊 **Evaluate Performance**
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")
print("\n📌 Classification Report:\n", classification_report(y_test, y_pred))

# 💾 **Save Model & Scaler**
os.makedirs(MODEL_DIR, exist_ok=True)
pickle.dump(model, open(os.path.join(MODEL_DIR, "intrusion_model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb"))
pickle.dump(list(X.columns), open(os.path.join(MODEL_DIR, "feature_names.pkl"), "wb"))

print(f"✅ Model & Scaler Saved in {MODEL_DIR}")
