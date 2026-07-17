.PHONY: help sandbox evals evals-answers gate test serve ask compare docker clean

help:
	@echo "make sandbox        - build the pinned pydantic v1/v2 sandbox venvs"
	@echo "make evals          - retrieval metrics only (offline, no deps)"
	@echo "make evals-answers  - full eval incl. answers + execution grading"
	@echo "make gate           - run the CI gate locally"
	@echo "make test           - pytest"
	@echo "make serve          - run the FastAPI server + demo UI at :8000"
	@echo "make ask Q='...'    - CLI: answer one question (V=v1|v2)"
	@echo "make compare Q='...'- CLI: answer both versions side by side"
	@echo "make docker         - build + run the self-contained demo image"

sandbox:
	bash scripts/setup_sandbox.sh

evals:
	python3 -m docsthatrun.evals.run_evals

evals-answers:
	python3 -m docsthatrun.evals.run_evals --answers --json results/report.json

gate:
	python3 -m docsthatrun.evals.run_evals --answers --gate --json results/report.json

test:
	pytest -q

serve:
	uvicorn app.main:app --reload

# Usage: make ask Q="In Pydantic v2, how do I serialize a model?" V=v2
V ?= v2
ask:
	python3 -m docsthatrun ask "$(Q)" --version $(V)

compare:
	python3 -m docsthatrun compare "$(Q)"

docker:
	docker build -t docsthatrun . && docker run --rm -p 8000:8000 docsthatrun

clean:
	rm -rf .venvs results **/__pycache__ .pytest_cache
