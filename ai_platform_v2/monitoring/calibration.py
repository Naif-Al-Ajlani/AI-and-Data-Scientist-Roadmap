"""
Expected Calibration Error (ECE) on synthetic predictions.
Usage: python monitoring/calibration.py
"""
import numpy as np

def ece(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins-1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # perfectly calibrated probs + noise
    y = rng.integers(0, 2, size=5000)
    p = 0.1 + 0.8 * y + rng.normal(0, 0.05, size=y.size)
    p = np.clip(p, 0, 1)
    print("ECE:", round(ece(p, y), 4))
