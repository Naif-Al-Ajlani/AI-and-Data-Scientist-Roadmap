from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from pathlib import Path
import json
import pandas as pd
from joblib import load

ART = Path(__file__).resolve().parents[1] / "artifacts"
MODEL = load(ART / "model.joblib")
FEATURES = json.loads((ART / "feature_names.json").read_text())

app = FastAPI(title="AI Platform V2 — Inference API", version="0.2.0")

class PredictRequest(BaseModel):
    features: Dict[str, float]

class BatchPredictRequest(BaseModel):
    rows: List[Dict[str, float]]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "n_features": len(FEATURES)}

@app.get("/metrics")
def metrics():
    path = ART / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "metrics.json not found; run training")
    return json.loads(path.read_text())

def _validate_and_to_df(d: Dict[str, float]) -> pd.DataFrame:
    missing = [f for f in FEATURES if f not in d]
    extras = [k for k in d.keys() if k not in FEATURES]
    if missing:
        raise HTTPException(400, f"Missing features: {missing}")
    if extras:
        raise HTTPException(400, f"Unknown features: {extras}")
    arr = [d[f] for f in FEATURES]
    return pd.DataFrame([arr], columns=FEATURES)

@app.post("/predict")
def predict(req: PredictRequest):
    X = _validate_and_to_df(req.features)
    proba = float(MODEL.predict_proba(X)[:, 1][0])
    return {"probability": proba, "label": int(proba >= 0.5)}

@app.post("/predict_batch")
def predict_batch(req: BatchPredictRequest):
    if not req.rows:
        raise HTTPException(400, "rows cannot be empty")
    import numpy as np
    X = pd.DataFrame([{k: r.get(k) for k in FEATURES} for r in req.rows], columns=FEATURES)
    probas = MODEL.predict_proba(X)[:, 1]
    labels = (probas >= 0.5).astype(int).tolist()
    return {"probabilities": probas.tolist(), "labels": labels}
