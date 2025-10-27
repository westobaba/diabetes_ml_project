import pandas as pd
import zipfile
import os
from sklearn.model_selection import train_test_split

def extract_zip(zip_path: str, extract_to: str = "data"):
    """Extracts ZIP file if it hasn't been extracted yet."""
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extracted {zip_path} to {extract_to}/")

def load_data(path: str):
    """
    Load diabetes dataset.
    If path is a ZIP file, it will be extracted automatically.
    """
    if path.endswith(".zip"):
        extract_to = "data"
        extract_zip(path, extract_to)
        # Find the CSV file inside the extracted folder
        for root, _, files in os.walk(extract_to):
            for file in files:
                if file.endswith(".csv"):
                    csv_path = os.path.join(root, file)
                    print(f"📂 Found CSV: {csv_path}")
                    return pd.read_csv(csv_path)
        raise FileNotFoundError("❌ No CSV found inside the ZIP file.")
    elif path.endswith(".csv"):
        print(f"📂 Loading CSV file: {path}")
        return pd.read_csv(path)
    else:
        raise ValueError("❌ Please provide a .csv or .zip file path.")

def preprocess_data(df: pd.DataFrame):
    """Preprocess dataset: drop ID, encode categoricals, and split X/y."""
    if 'patient_id' in df.columns:
        df = df.drop(columns=['patient_id'])

    y = df['diagnosed_diabetes']
    X = df.drop(columns=['diagnosed_diabetes'])

    cat_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train/test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
