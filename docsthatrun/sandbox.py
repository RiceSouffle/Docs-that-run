"""Execution grader.

Runs a code snippet against the *pinned* version of pydantic in an isolated
venv and returns pass/fail. This is the project's core differentiator: an answer
isn't graded on plausibility, it's graded on whether it actually runs against
the version it claims to target.

Set up the venvs once with ``scripts/setup_sandbox.sh`` (or ``make sandbox``).
If a venv is missing, grading returns ``available=False`` rather than crashing,
so retrieval-only evals still run on a machine with no network access.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
_VENV_DIR = os.path.join(_REPO_ROOT, ".venvs")

VENV_PYTHON = {
    "v1": os.path.join(_VENV_DIR, "pydantic_v1", "bin", "python"),
    "v2": os.path.join(_VENV_DIR, "pydantic_v2", "bin", "python"),
}

DEFAULT_TIMEOUT_SECONDS = 20


@dataclass
class ExecResult:
    passed: bool
    available: bool
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "available": self.available,
            "returncode": self.returncode,
            "reason": self.reason,
            "stderr_tail": self.stderr[-400:],
        }


def sandbox_available(version: str) -> bool:
    return os.path.exists(VENV_PYTHON.get(version, ""))


def grade(
    code: str, version: str, timeout: int = DEFAULT_TIMEOUT_SECONDS
) -> ExecResult:
    if version not in VENV_PYTHON:
        return ExecResult(False, False, reason=f"unknown version {version!r}")

    python = VENV_PYTHON[version]
    if not os.path.exists(python):
        return ExecResult(
            False,
            False,
            reason=(
                f"sandbox for {version} not set up — run scripts/setup_sandbox.sh"
            ),
        )
    if not code.strip():
        return ExecResult(False, True, reason="empty code snippet")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    )
    try:
        tmp.write(code)
        tmp.flush()
        tmp.close()
        try:
            proc = subprocess.run(
                [python, tmp.name],
                capture_output=True,
                text=True,
                timeout=timeout,
                # Isolate from the caller's env; empty PYTHONPATH so only the
                # venv's packages are importable.
                env={"PYTHONPATH": "", "PATH": os.environ.get("PATH", "")},
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                False, True, reason=f"timed out after {timeout}s"
            )
        passed = proc.returncode == 0
        return ExecResult(
            passed=passed,
            available=True,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            reason="ok" if passed else "non-zero exit",
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:  # pragma: no cover
            pass


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    demo = "from pydantic import BaseModel\nclass M(BaseModel):\n    x: int\nprint(M(x=1).model_dump())\n"
    print("v2:", grade(demo, "v2").to_dict())
    print("v1:", grade(demo, "v1").to_dict())
    sys.exit(0)
