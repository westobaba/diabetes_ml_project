# main.py
from src.dataloader import load_data, preprocess_data, split_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import save_model, load_model

# ==== CONFIG ====
# Use raw string for Windows paths
data_path = r"C:\Users\SKRIMGADGETS\Downloads\archive (1).zip"
model_save_path = r"models/random_forest_model.pkl"

# ==== LOAD DATA ====
print("=== Loading and Preprocessing Data ===")
df = load_data(data_path)
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# ==== TRAIN MODEL ====
print("=== Training Model ===")
model = train_model(X_train, y_train, model_type="random_forest")

# Save the trained model
save_model(model, model_save_path)
print(f"✅ Model saved to {model_save_path}")

# ==== EVALUATE MODEL ====
print("=== Evaluating Model ===")
evaluate_model(model, X_test, y_test)
