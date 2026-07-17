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
from typing import Dict, List, Optional

from .config import settings

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
_VENV_DIR = os.path.join(_REPO_ROOT, ".venvs")

VENV_PYTHON = {
    "v1": os.path.join(_VENV_DIR, "pydantic_v1", "bin", "python"),
    "v2": os.path.join(_VENV_DIR, "pydantic_v2", "bin", "python"),
}

DEFAULT_TIMEOUT_SECONDS = settings.sandbox_timeout_s


def _launcher_code(cpu_s: int, fsize_bytes: int, as_bytes: int) -> str:
    """Python that self-applies rlimits, then runs the target as __main__.

    Setting the limits *inside* the child (after exec, single-threaded) instead
    of via ``preexec_fn`` avoids the fork-in-a-threaded-server deadlock hazard.
    RLIMIT_AS is skipped on macOS (unreliable there) and applied on Linux/prod;
    it's set generously so a legit pydantic import never false-fails. CPU and
    FSIZE limits stop infinite loops and disk-fill snippets; CORE=0 suppresses
    core dumps. All best-effort — a platform that rejects a limit is not fatal.
    """
    return (
        "import resource,runpy,sys\n"
        "def _l(r,v):\n"
        " try:\n"
        "  s,h=resource.getrlimit(r)\n"
        "  c=v if h==resource.RLIM_INFINITY else min(v,h)\n"
        "  resource.setrlimit(r,(c,c))\n"
        " except Exception: pass\n"
        f"_l(resource.RLIMIT_CPU,{cpu_s})\n"
        f"_l(resource.RLIMIT_FSIZE,{fsize_bytes})\n"
        "_l(resource.RLIMIT_CORE,0)\n"
        f"_AS={as_bytes}\n"
        "if _AS>0 and sys.platform!='darwin':\n"
        " _l(resource.RLIMIT_AS,_AS)\n"
        # Normalize argv so the target sees exactly what `python file.py` gives
        # ([path]); otherwise the launcher's own argv ('-c', path) leaks the path
        # in twice and a snippet using argparse / len(sys.argv) would false-fail.
        "_t=sys.argv[1]\n"
        "sys.argv=[_t]\n"
        "runpy.run_path(_t,run_name='__main__')\n"
    )


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
    code: str,
    version: str,
    timeout: Optional[int] = None,
    cpu_seconds: Optional[int] = None,
    memory_mb: Optional[int] = None,
    file_mb: Optional[int] = None,
) -> ExecResult:
    """Run ``code`` against the pinned-version venv under resource limits.

    ``timeout`` bounds wall-clock; ``cpu_seconds``/``memory_mb``/``file_mb`` cap
    CPU time, address space, and file writes. ``None`` means "use the configured
    default" (see docsthatrun.config). Defence-in-depth: even a self-authored
    snippet can loop, allocate, or write forever — the limits contain all three.
    """
    timeout = timeout if timeout is not None else settings.sandbox_timeout_s
    cpu_seconds = cpu_seconds if cpu_seconds is not None else settings.sandbox_cpu_seconds
    memory_mb = memory_mb if memory_mb is not None else settings.sandbox_memory_mb
    file_mb = file_mb if file_mb is not None else settings.sandbox_file_mb

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
            cmd: List[str] = [
                python,
                "-c",
                _launcher_code(cpu_seconds, file_mb * 1024 * 1024, memory_mb * 1024 * 1024),
                tmp.name,
            ]
        else:  # pragma: no cover - non-POSIX has no rlimits; run directly
            cmd = [python, tmp.name]
        proc = subprocess.Popen(cmd, **popen_kwargs)
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
