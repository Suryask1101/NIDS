import os
import pandas as pd

# 📂 Get the absolute path of the project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "preprocessed_data.csv")

# 🗂 Define dataset files
dataset_files = [
    os.path.join(DATA_DIR, "DDoS attacks-LOIC-HTTP.csv"),
    os.path.join(DATA_DIR, "DoS attacks-Hulk.csv"),
    os.path.join(DATA_DIR, "SSH-Bruteforce.csv"),
    os.path.join(DATA_DIR, "FTP-BruteForce.csv"),
]

# ✅ Verify all files exist before proceeding
missing_files = [file for file in dataset_files if not os.path.exists(file)]
if missing_files:
    print(f"❌ ERROR: Missing files -> {missing_files}")
    exit()

# 📥 Load & combine datasets
df_list = [pd.read_csv(file, low_memory=False) for file in dataset_files]
df = pd.concat(df_list, ignore_index=True)

# 🔎 Display initial shape
print(f"✅ Data Loaded! Initial shape: {df.shape}")

# 🏷️ **Ensure Label Column Exists**
label_column = None
for col in df.columns:
    if "label" in col.lower():  # Case-insensitive check
        label_column = col
        break

if not label_column:
    print(f"❌ ERROR: No label column found. Available columns: {df.columns}")
    exit()

print(f"✅ Using label column: {label_column}")

# 🛑 **Drop Duplicates & Missing Values**
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# 🔄 **Categorical Encoding** (Convert Strings to Numbers)
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
categorical_columns.remove(label_column)  # Keep label as is

df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# 🎯 **Save Processed Data**
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Preprocessing Complete! Saved to {OUTPUT_FILE}")
