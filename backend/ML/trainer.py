# backend/ml/trainer.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import uuid
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_model(model_type, X_train, y_train, X_test, y_test):
    """Train a specified model and return metrics."""
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "svm":
        model = SVC(kernel="rbf", probability=True, C=3, gamma="scale", random_state=42)
    elif model_type == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    model_id = str(uuid.uuid4())[:8]
    model_path = os.path.join(MODELS_DIR, f"{model_type}_{model_id}.joblib")
    joblib.dump(model, model_path)

    return {
        "model_type": model_type,
        "accuracy": acc,
        "report": report,
        "model_path": model_path
    }
