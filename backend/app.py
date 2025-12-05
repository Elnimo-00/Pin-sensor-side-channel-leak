# backend/app.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ML.utils import parse_dataset, preprocess_data
from ML.trainer import train_model, MODELS_DIR

import joblib
import os

app = FastAPI(title="PIN Side-Channel ML API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend folder
FRONTEND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(FRONTEND_PATH):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_PATH), name="frontend")
else:
    print(f"⚠️ Frontend directory not found at: {FRONTEND_PATH}")

@app.get("/")
def read_root():
    return {"message": "PIN Side-Channel ML API is running"}

@app.post("/train")
async def train_endpoint(file: UploadFile = File(...), model_type: str = Form("rf")):
    """Train model on uploaded dataset."""
    try:
        data = await file.read()
        X, y = parse_dataset(data)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        result = train_model(model_type, X_train, y_train, X_test, y_test)

        return JSONResponse({
            "message": "Training complete",
            "accuracy": round(result["accuracy"], 4),
            "report": result["report"],
            "model_file": os.path.basename(result["model_path"]),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...), model_name: str = Form(...)):
    """Predict labels using a saved model."""
    try:
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            return JSONResponse({"error": "Model not found"}, status_code=404)

        model = joblib.load(model_path)
        data = await file.read()
        X, y = parse_dataset(data)

        preds = model.predict(X)
        acc = None
        if y is not None:
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y, preds)

        return JSONResponse({
            "predictions": preds.tolist(),
            "accuracy": round(acc, 4) if acc else None,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
