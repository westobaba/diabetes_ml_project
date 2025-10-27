import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance for tree-based models."""
    importance = model.feature_importances_
    feat_imp = pd.Series(importance, index=feature_names).sort_values(ascending=False)[:top_n]
    plt.figure(figsize=(8,6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()

def save_model(model, path='model.pkl'):
    """Save trained model to file."""
    import joblib
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

def load_model(path='model.pkl'):
    """Load model from file."""
    import joblib
    return joblib.load(path)
