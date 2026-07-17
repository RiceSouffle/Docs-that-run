# DocsThatRun — self-contained image: API + the two pinned-version sandboxes.
#
#   docker build -t docsthatrun .
#   docker run -p 8000:8000 docsthatrun                        # offline MockClient
#   docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... docsthatrun  # real Claude
#
# Then open http://localhost:8000  (the interactive demo UI).
#
# Python is pinned to 3.11: pydantic 1.10.x builds cleanly there (needed for the
# v1 sandbox), and RLIMIT_AS memory caps are enforced on Linux.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DOCSTHATRUN_LLM=auto

WORKDIR /app

# App-side deps only (the core is stdlib). Copy the manifest first so this layer
# caches across source changes.
COPY requirements.txt ./
RUN pip install "fastapi>=0.110" "uvicorn[standard]>=0.29" "anthropic>=0.40"

COPY . .

# Build the pinned pydantic v1.x / v2.x sandbox venvs at image-build time so the
# execution grader works out of the box with no runtime network.
RUN bash scripts/setup_sandbox.sh

# Drop privileges: the sandbox runs model-generated code, so the server process
# must not be root. Build artifacts are chowned to the unprivileged user.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
