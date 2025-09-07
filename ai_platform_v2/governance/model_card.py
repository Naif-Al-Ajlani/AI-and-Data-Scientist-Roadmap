"""
Generate a lightweight model card from artifacts.
Usage: python governance/model_card.py
"""
from pathlib import Path
import json, datetime

ART = Path(__file__).resolve().parents[1] / "artifacts"
OUT = Path(__file__).resolve().parents[1] / "governance" / "model_card.md"

def main():
    m = json.loads((ART / "metrics.json").read_text())
    feats = json.loads((ART / "feature_names.json").read_text())
    OUT.write_text(f"""# Model Card — Logistic Regression (Calibrated)

**Generated:** {datetime.datetime.utcnow().isoformat()}Z

## Intended Use
Binary classification demo for platform validation (thin slice).

## Data
Synthetic 4k samples with 8 features; stratified split (75/25).

## Metrics (test set)
- AUC: {m['test_auc']:.3f}
- F1: {m['test_f1']:.3f}
- Brier: {m['brier']:.4f}
- N test: {m['n_test']}

## Features
{', '.join(feats)}

## Evaluation Protocol
- 5-fold CV AUC on train (stratified), isotonic calibration.
- Test metrics logged; calibration & drift scripts available in /monitoring.

## Risks & Limitations
- Synthetic data; may not reflect real-world shift, biases, or label noise.

## Retraining Triggers
- PSI >= 0.2 on 2+ key features
- ECE >= 0.05
- Online KPI degradation beyond guardrails
""")

if __name__ == "__main__":
    main()
    print("Wrote", OUT)
