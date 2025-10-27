from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model(model_type='random_forest'):
    """Build a sklearn model pipeline."""
    if model_type == 'logistic':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000))
        ])
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=150, random_state=42)
    else:
        raise ValueError("Unsupported model_type. Choose 'logistic' or 'random_forest'.")

    return model
