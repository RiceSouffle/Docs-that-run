# DocsThatRun — self-contained image: API + the two pinned-version sandboxes.
#
#   docker build -t docsthatrun .
#   docker run -p 8000:8000 docsthatrun            # offline MockClient
#   docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... docsthatrun   # real Claude
#
# Then open http://localhost:8000  (the interactive demo UI).
#
# Python is pinned to 3.11: pydantic 1.10.x builds cleanly there, which the
# execution grader needs for the v1 sandbox.
FROM python:3.11-slim

WORKDIR /app

# App-side deps (server + optional real LLM). The core is stdlib-only; these are
# for serving the API and calling Claude.
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29" "anthropic>=0.40"

COPY . .

# Build the pinned pydantic v1.x / v2.x sandbox venvs at image-build time so the
# execution grader works out of the box (no runtime network needed).
RUN bash scripts/setup_sandbox.sh

EXPOSE 8000
ENV DOCSTHATRUN_LLM=auto
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
