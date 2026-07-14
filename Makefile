.PHONY: help sandbox evals evals-answers gate test serve clean

help:
	@echo "make sandbox        - build the pinned pydantic v1/v2 sandbox venvs"
	@echo "make evals          - retrieval metrics only (offline, no deps)"
	@echo "make evals-answers  - full eval incl. answers + execution grading"
	@echo "make gate           - run the CI gate locally"
	@echo "make test           - pytest"
	@echo "make serve          - run the FastAPI server"

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

clean:
	rm -rf .venvs results **/__pycache__ .pytest_cache
