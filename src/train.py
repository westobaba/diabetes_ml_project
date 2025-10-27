# src/train.py
from sklearn.metrics import classification_report, accuracy_score
from src.model import build_model
from src.utils import save_model

def train_model(X_train, y_train, X_test=None, y_test=None, model_type='random_forest', save_path=None):
    """
    Train a model on preprocessed X/y data.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Optional test features for evaluation
        y_test: Optional test labels for evaluation
        model_type: Type of model to train ('random_forest' supported)
        save_path: Path to save the trained model (optional)

    Returns:
        Trained model
    """
    # Build and train
    model = build_model(model_type)
    model.fit(X_train, y_train)

    # Evaluate if test set provided
    if X_test is not None and y_test is not None:
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"✅ Model: {model_type}")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

    # Save model if path provided
    if save_path:
        save_model(model, save_path)
        print(f"✅ Model saved to {save_path}")

    return model
