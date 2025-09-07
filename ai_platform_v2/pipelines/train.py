# trains a calibrated binary classifier and writes artifacts/ (with MLflow logging)
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from joblib import dump
import mlflow

ART = Path(__file__).resolve().parents[1] / "artifacts"
ART.mkdir(exist_ok=True, parents=True)

# 1) synthetic dataset (reproducible)
X, y = make_classification(
    n_samples=4000, n_features=8, n_informative=6, n_redundant=2,
    weights=[0.6, 0.4], class_sep=1.2, random_state=42
)
feature_names = [f"f{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["label"] = y

X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names].values, df["label"].values, test_size=0.25,
    stratify=df["label"].values, random_state=7
)

# 2) base pipeline + calibration
base = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500, n_jobs=None))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
probs_cv = cross_val_predict(base, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
auc_cv = roc_auc_score(y_train, probs_cv)

cal = CalibratedClassifierCV(base, method="isotonic", cv=5)

with mlflow.start_run(run_name="logreg_isotonic"):
    cal.fit(X_train, y_train)
    probs = cal.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "cv_auc": float(auc_cv),
        "test_auc": float(roc_auc_score(y_test, probs)),
        "test_f1": float(f1_score(y_test, preds)),
        "brier": float(brier_score_loss(y_test, probs)),
        "n_test": int(len(y_test))
    }
    # log metrics & params
    mlflow.log_params({"model": "logreg", "calibration": "isotonic", "n_features": len(feature_names)})
    mlflow.log_metrics(metrics)

    # persist artifacts
    from joblib import dump
    dump(cal, ART / "model.joblib")
    (ART / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))

    mlflow.log_artifact(str(ART / "feature_names.json"))
    mlflow.log_artifact(str(ART / "metrics.json"))
    mlflow.log_artifact(str(ART / "model.joblib"))

print("Saved artifacts to:", ART)
print("Metrics:", json.dumps(metrics, indent=2))
