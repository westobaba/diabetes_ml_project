# src/evaluate.py
from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    """
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("=== Evaluation Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
