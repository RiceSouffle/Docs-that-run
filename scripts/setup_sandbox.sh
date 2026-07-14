#!/usr/bin/env bash
# Build the two pinned-version sandbox venvs used by the execution grader.
# Idempotent: skips a venv that already has the right pydantic installed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENVS="$ROOT/.venvs"
PY="${PYTHON:-python3}"

mkdir -p "$VENVS"

make_venv () {
  local name="$1"; shift
  local dir="$VENVS/$name"
  if [ ! -x "$dir/bin/python" ]; then
    echo ">> creating venv $name"
    "$PY" -m venv "$dir"
  fi
  echo ">> installing into $name: $*"
  "$dir/bin/python" -m pip install --quiet --upgrade pip
  "$dir/bin/python" -m pip install --quiet "$@"
}

# Pydantic v1 line.
make_venv "pydantic_v1" "pydantic>=1.10,<2"

# Pydantic v2 line + the split-out settings package (see corpus c_v2_settings).
make_venv "pydantic_v2" "pydantic>=2,<3" "pydantic-settings>=2,<3"

echo ">> sandbox ready:"
"$VENVS/pydantic_v1/bin/python" -c "import pydantic; print('  v1 ->', pydantic.VERSION)"
"$VENVS/pydantic_v2/bin/python" -c "import pydantic; print('  v2 ->', pydantic.VERSION)"
