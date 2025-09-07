# AI Platform (V2) — Thin Slice + Ops

This repo is a production-grade thin slice for an AI platform that maps to the **AI & Data Scientist Roadmap** (Math/Stats → Econometrics → Coding → EDA → ML/DL → MLOps).

## Quickstart (Local)
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Train and save calibrated model + metrics
python pipelines/train.py

# 2) Serve inference API
uvicorn api.main:app --host 0.0.0.0 --port 8000
#   GET  http://localhost:8000/health
#   GET  http://localhost:8000/metrics
#   POST http://localhost:8000/predict  (see README body for JSON)

# 3) A/B testing helpers
python experiments/power_mde.py
python experiments/analyze_ab.py

# 4) Monitoring
python monitoring/drift.py
python monitoring/calibration.py
```

## Docker
```bash
docker build -t ai-platform-v2 .
docker run --rm -p 8000:8000 ai-platform-v2
```

## Docker Compose (API + Redis + MLflow)
```bash
docker compose -f infra/docker-compose.yml up --build
```

## Feast (optional demo)
```bash
python feature_repo/materialize_demo.py         # generate demo parquet
# Edit feature_repo/feature_store.yaml if needed (Redis host, etc.)
# Then use Feast CLI if installed: `feast apply && feast materialize-incremental 2025-01-01`
```

## Prefect flow (ad-hoc)
```bash
python orchestration/flow.py
```

## Makefile
See `make help` for common tasks.

Generated at: 2025-09-07T00:01:09.515347
