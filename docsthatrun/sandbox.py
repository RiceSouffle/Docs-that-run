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
import signal
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


_IMPORT_OK: Dict[str, bool] = {}


def _kill_process_tree(proc: "subprocess.Popen", posix: bool) -> None:
    """SIGKILL the whole process group so grandchildren die with the child."""
    try:
        if posix:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:  # pragma: no cover - non-POSIX fallback
            proc.kill()
    except (ProcessLookupError, OSError):  # pragma: no cover - already gone
        try:
            proc.kill()
        except OSError:
            pass


def sandbox_available(version: str) -> bool:
    """True only if the venv exists *and* pydantic actually imports in it.

    Checking for ``bin/python`` alone is not enough: ``setup_sandbox.sh`` creates
    the venv before ``pip install`` runs, so an interrupted setup leaves a
    ``bin/python`` with no pydantic. Reporting that as "available" would grade
    every snippet as a failing ``ModuleNotFoundError`` and misattribute it to
    answer quality. We probe ``import pydantic`` once per version and cache it.
    """
    python = VENV_PYTHON.get(version, "")
    if not python or not os.path.exists(python):
        return False
    if version not in _IMPORT_OK:
        try:
            proc = subprocess.run(
                [python, "-c", "import pydantic"],
                capture_output=True,
                timeout=15,
                env={"PYTHONPATH": "", "PATH": os.environ.get("PATH", "")},
            )
        except (OSError, subprocess.SubprocessError):
            # Transient (fork EAGAIN under load, or the probe timing out while the
            # venv is still being populated). Don't cache — a later call retries,
            # so a momentary hiccup can't permanently disable grading.
            return False
        # Only a clean run gives a *definitive* answer worth caching.
        _IMPORT_OK[version] = proc.returncode == 0
    return _IMPORT_OK[version]


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
        posix = os.name == "posix"
        # Run the snippet in its own session (process group) so a snippet that
        # spawns grandchildren (subprocess.Popen, os.fork, a background thread's
        # process) can't outlive the timeout. subprocess.run only SIGKILLs the
        # *direct* child on timeout, orphaning anything it spawned; we kill the
        # whole group. Env is isolated (empty PYTHONPATH => only the venv's
        # packages are importable).
        popen_kwargs = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"PYTHONPATH": "", "PATH": os.environ.get("PATH", "")},
        )
        if posix:
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen([python, tmp.name], **popen_kwargs)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc, posix)
            # Reap and drain pipes after the kill. Bound this too: a snippet that
            # setsid-escapes the group (double-fork daemon) can keep the pipe's
            # write end open, so an unbounded communicate() would block forever.
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                for pipe in (proc.stdout, proc.stderr):
                    try:
                        if pipe:
                            pipe.close()
                    except OSError:  # pragma: no cover
                        pass
                stdout, stderr = "", ""
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    pass
            return ExecResult(
                False,
                True,
                returncode=proc.returncode,
                stdout=stdout or "",
                stderr=stderr or "",
                reason=f"timed out after {timeout}s",
            )
        passed = proc.returncode == 0
        return ExecResult(
            passed=passed,
            available=True,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
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
