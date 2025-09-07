"""
Population Stability Index (PSI) on synthetic recent vs train data.
Usage: python monitoring/drift.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts"

def psi(expected, actual, bins=10):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    q = np.linspace(0, 1, bins+1)
    cuts = np.quantile(expected, q)
    def hist(x):
        h, _ = np.histogram(x, bins=cuts)
        p = h / h.sum()
        # avoid zeros
        p = np.where(p==0, 1e-6, p)
        return p
    pe = hist(expected)
    pa = hist(actual)
    return float(np.sum((pa - pe) * np.log(pa/pe)))

if __name__ == "__main__":
    # Load training feature distribution
    feats = json.loads((ART / "feature_names.json").read_text())
    rng = np.random.default_rng(123)
    # pseudo "train" sample (normal)
    train = pd.DataFrame(rng.normal(0, 1, size=(5000, len(feats))), columns=feats)
    # pseudo "recent" sample with slight shift
    recent = pd.DataFrame(rng.normal(0.1, 1.1, size=(5000, len(feats))), columns=feats)

    out = {}
    for f in feats:
        out[f] = psi(train[f].values, recent[f].values)

    print("PSI per feature (>=0.2 needs attention):")
    for k, v in out.items():
        flag = " !!" if v >= 0.2 else ""
        print(f"  {k}: {v:.3f}{flag}")
