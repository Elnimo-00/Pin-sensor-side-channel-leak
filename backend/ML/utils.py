# backend/ml/utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def parse_dataset(file_bytes: bytes):
    """Parses uploaded dataset text into X (features) and y (labels)."""
    text = file_bytes.decode("utf-8").strip()
    rows = [r.strip() for r in text.splitlines() if r.strip()]
    data = [list(map(float, r.replace(",", " ").split())) for r in rows]
    df = pd.DataFrame(data)
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns (label + features)")
    y = df.iloc[:, 0].astype(int)
    X = df.iloc[:, 1:]
    return X, y

def preprocess_data(X, y, test_size=0.2):
    """Splits and scales data."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
