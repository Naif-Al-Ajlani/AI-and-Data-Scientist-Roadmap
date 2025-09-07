.PHONY: help setup train serve test ab drift calib compose-up compose-down

help:
	@echo "Common targets:"
	@echo "  setup        - create venv and install deps"
	@echo "  train        - train model and save artifacts"
	@echo "  serve        - run FastAPI server"
	@echo "  test         - run unit tests"
	@echo "  ab           - run A/B power & analysis demos"
	@echo "  drift        - run drift checks (PSI)"
	@echo "  calib        - run calibration & ECE checks"
	@echo "  compose-up   - start API+Redis+MLflow via docker-compose"
	@echo "  compose-down - stop docker-compose stack"

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train:
	python pipelines/train.py

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

ab:
	python experiments/power_mde.py && python experiments/analyze_ab.py

drift:
	python monitoring/drift.py

calib:
	python monitoring/calibration.py

compose-up:
	docker compose -f infra/docker-compose.yml up --build

compose-down:
	docker compose -f infra/docker-compose.yml down -v
