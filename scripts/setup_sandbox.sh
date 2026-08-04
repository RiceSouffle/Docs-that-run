#!/usr/bin/env bash
# Build the two pinned-version sandbox venvs used by the execution grader.
# Safe to re-run: venv creation is skipped when the interpreter already exists,
# and pip install is a no-op once the requirement is satisfied.
#
# The versions below are pinned exactly, not ranged. "Graded against the pinned
# version" is the project's central claim, and a range quietly breaks it: two
# builds a month apart would grade against different libraries, so the published
# version-lock rate (17/25) and executable-% would drift with no code change and
# no signal. To move to a newer pydantic, bump a pin here, re-run
# `make evals-answers`, and update the numbers in README/GUIDE/DECISIONS/ROADMAP
# together — see DECISIONS.md.
set -euo pipefail

PYDANTIC_V1="pydantic==1.10.26"
PYDANTIC_V2="pydantic==2.13.4"
PYDANTIC_SETTINGS_V2="pydantic-settings==2.11.0"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENVS="$ROOT/.venvs"
PY="${PYTHON:-python3}"

# pydantic 1.10.x ships no wheels for the newest interpreters, so an unsupported
# host would otherwise fail deep inside a source build with a compiler error
# instead of a sentence anyone can act on.
if ! "$PY" -c 'import sys; sys.exit(0 if (3,9) <= sys.version_info < (3,13) else 1)'; then
  echo "error: the sandbox venvs need Python 3.9-3.12 (pydantic 1.10 has no wheels beyond that)." >&2
  echo "       '$PY' is $("$PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')." >&2
  echo "       Point PYTHON at a supported interpreter: PYTHON=python3.11 make sandbox" >&2
  exit 1
fi

mkdir -p "$VENVS"

make_venv () {
  local name="$1"; shift
  local dir="$VENVS/$name"
  if [ ! -x "$dir/bin/python" ]; then
    echo ">> creating venv $name"
    "$PY" -m venv "$dir"
    # Only on first creation: an unconditional upgrade needs the network on
    # every re-run, which contradicts the "safe to re-run" promise above.
    "$dir/bin/python" -m pip install --quiet --upgrade pip
  fi
  echo ">> installing into $name: $*"
  "$dir/bin/python" -m pip install --quiet "$@"
}

# Pydantic v1 line.
make_venv "pydantic_v1" "$PYDANTIC_V1"

# Pydantic v2 line + the split-out settings package (see corpus c_v2_settings).
make_venv "pydantic_v2" "$PYDANTIC_V2" "$PYDANTIC_SETTINGS_V2"

# Verify exactly what sandbox_available() probes — including pydantic_settings
# for v2. Checking only `pydantic` would print "sandbox ready" for a v2 venv the
# grader then refuses to use.
echo ">> sandbox ready:"
"$VENVS/pydantic_v1/bin/python" -c "import pydantic; print('  v1 ->', pydantic.VERSION)"
"$VENVS/pydantic_v2/bin/python" -c "import pydantic, pydantic_settings; print('  v2 ->', pydantic.VERSION, '+ pydantic-settings', pydantic_settings.__version__)"
