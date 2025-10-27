# Diabetes ML Project

A machine learning pipeline for predicting diabetes from patient health indicators. This project demonstrates **data loading, preprocessing, model training, evaluation, and model saving** in a clean, modular structure.

---

## 📂 Project Structure

diabetes_ml_project/
│
├── data/ # Extracted dataset folder
├── models/ # Trained model saved here
├── src/
│ ├── init.py
│ ├── dataloader.py # Load and preprocess data
│ ├── train.py # Train model
│ ├── evaluate.py # Evaluate model
│ ├── model.py # Build ML models
│ └── utils.py # Utility functions (save/load model)
├── main.py # Main script to run the pipeline
├── requirements.txt # Python dependencies
└── README.md

yaml
Copy code

---

## 📝 Dataset

- **Name:** Diabetes Health Indicators Dataset  
- **Source:** Generated synthetic dataset (~100,000 patients)  
- **Features:** Demographics, lifestyle, family history, clinical measurements  
- **Target:** `diagnosed_diabetes` (binary classification)  
- **Format:** CSV or ZIP (`diabetes_dataset.csv`)

---

## ⚡ Features

- Binary classification to predict diabetes diagnosis.
- Modular design:
  - `dataloader.py` → load and preprocess data
  - `train.py` → train Random Forest or other ML models
  - `evaluate.py` → evaluate trained model
  - `utils.py` → save/load models
- Supports ZIP datasets and auto-creates required folders.
- Fully reproducible with `main.py`.

---

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/westobaba/diabetes_ml_project.git
cd diabetes_ml_project
Create a virtual environment:

bash
Copy code
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # macOS/Linux
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Download the dataset ZIP and place it anywhere (update path in main.py).

🏃‍♂️ Running the Project
bash
Copy code
python main.py
The pipeline will:

Extract and load the dataset

Preprocess features and labels

Split into training and test sets

Train a Random Forest classifier

Save the trained model to models/

Evaluate and print accuracy & classification report

🔧 Customization
Change model type:

python
Copy code
model = train_model(X_train, y_train, X_test, y_test, model_type="random_forest")
Other models can be added in src/model.py.

Change save path:

python
Copy code
model_save_path = "models/custom_model.pkl"
📊 Results
Test set accuracy: ~0.9997 (synthetic dataset)

Extremely high performance is expected due to the synthetic nature of the data.

🛠 Tech Stack
Python 3.9+

Pandas, scikit-learn, joblib

Modular project structure for ML pipelines

