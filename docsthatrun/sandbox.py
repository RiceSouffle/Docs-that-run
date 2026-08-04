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
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import settings

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
_VENV_DIR = os.path.join(_REPO_ROOT, ".venvs")

VENV_PYTHON = {
    "v1": os.path.join(_VENV_DIR, "pydantic_v1", "bin", "python"),
    "v2": os.path.join(_VENV_DIR, "pydantic_v2", "bin", "python"),
}

# What each sandbox must be able to import to be considered usable. v2 also needs
# pydantic_settings: v2 split settings into its own distribution, and golden item
# g_v2_settings exercises it. Probing only `pydantic` would call a half-installed
# v2 venv "available" and then charge the resulting ModuleNotFoundError to answer
# quality (bucket `wrong_version_api`) — the exact misattribution this probe
# exists to prevent.
_PROBE_IMPORTS = {
    "v1": "import pydantic",
    "v2": "import pydantic, pydantic_settings",
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

# A failed probe is deliberately not cached as a verdict (the venv may still be
# installing), but it must not be re-forked on every request either: /health and
# /ready probe both versions per call, so a half-built venv turned a health
# check into two process spawns each time — under a monitoring poll that is a
# steady fork storm. Remember only *when* a probe last failed and skip re-probing
# for this many seconds. Self-healing is preserved, just rate-limited.
_IMPORT_FAIL_AT: Dict[str, float] = {}
_PROBE_RETRY_S = 10.0


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


_SIGNAL_REASONS = {
    signal.SIGXCPU: "killed: exceeded the CPU-time limit",
    signal.SIGXFSZ: "killed: exceeded the file-size limit",
    signal.SIGKILL: "killed (SIGKILL)",
    signal.SIGSEGV: "crashed (segmentation fault)",
}


def _exit_reason(returncode: Optional[int], passed: bool) -> str:
    """Explain a non-zero exit. A signal kill leaves NO stderr, so "non-zero
    exit" was the only thing the caller ever saw for a snippet stopped by the
    CPU or file-size limit — naming the signal makes those self-explanatory."""
    if passed:
        return "ok"
    if returncode is not None and returncode < 0:
        sig = -returncode
        try:
            name = signal.Signals(sig).name
        except ValueError:  # pragma: no cover - unknown signal number
            name = f"signal {sig}"
        return _SIGNAL_REASONS.get(sig, f"killed by {name}")
    return "non-zero exit"


def _kill_group(pgid: Optional[int]) -> None:
    """SIGKILL a process group by the id captured when it was spawned.

    Deliberately *not* ``os.getpgid(proc.pid)``: after the direct child exits
    that lookup raises ESRCH on macOS, and after it is reaped it raises
    everywhere — so a late lookup silently kills nothing. The group stays
    addressable by the leader's pid while any member is alive.
    """
    if pgid is None:  # pragma: no cover - non-POSIX
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        # ESRCH is the normal case: the snippet behaved and the group is empty.
        pass


# Only ever surface a bounded tail of a snippet's output. `stderr_tail` is 400
# chars and stdout is not reported at all, so this is generous — its job is to
# guarantee that however much the child wrote, this process reads a fixed
# maximum into memory.
_MAX_CAPTURE_BYTES = 64 * 1024


def _read_tail(path: str, limit: int = _MAX_CAPTURE_BYTES) -> str:
    """Last ``limit`` bytes of ``path``, decoded leniently. Never loads the whole
    file: the interesting part of a traceback is at the end anyway."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            if size > limit:
                handle.seek(size - limit)
            data = handle.read(limit)
    except OSError:  # pragma: no cover - file vanished or unreadable
        return ""
    return data.decode("utf-8", errors="replace")


def sandbox_available(version: str) -> bool:
    """True only if the venv exists *and* pydantic actually imports in it.

    Checking for ``bin/python`` alone is not enough: ``setup_sandbox.sh`` creates
    the venv before ``pip install`` runs, so an interrupted setup leaves a
    ``bin/python`` with no pydantic. Reporting that as "available" would grade
    every snippet as a failing ``ModuleNotFoundError`` and misattribute it to
    answer quality. We probe every package the version's golden set needs and
    cache *success only*.
    """
    python = VENV_PYTHON.get(version, "")
    if not python or not os.path.exists(python):
        return False
    if version in _IMPORT_OK:
        return True
    last_fail = _IMPORT_FAIL_AT.get(version)
    if last_fail is not None and (time.monotonic() - last_fail) < _PROBE_RETRY_S:
        return False  # probed recently and it failed; don't re-fork yet
    try:
        proc = subprocess.run(
            [python, "-c", _PROBE_IMPORTS[version]],
            capture_output=True,
            timeout=15,
            env={"PYTHONPATH": "", "PATH": os.environ.get("PATH", "")},
        )
    except (OSError, subprocess.SubprocessError):
        # Transient (fork EAGAIN under load, or the probe timing out while the
        # venv is still being populated). Don't cache — a later call retries,
        # so a momentary hiccup can't permanently disable grading.
        _IMPORT_FAIL_AT[version] = time.monotonic()
        return False
    if proc.returncode != 0:
        # Import failed — usually a venv whose `pip install` is still running
        # (or was interrupted). Equally transient, equally uncached as a
        # verdict: a probe racing `make sandbox` must not disable grading for
        # the process lifetime. Only success is definitive.
        _IMPORT_FAIL_AT[version] = time.monotonic()
        return False
    _IMPORT_OK[version] = True
    _IMPORT_FAIL_AT.pop(version, None)
    return True


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
    snippet can loop, allocate, write, or *print* forever — the limits contain
    all four.

    Output is captured to temp files rather than pipes. A pipe would be read into
    this process's memory with no ceiling, and neither RLIMIT_AS (which bounds the
    child, not the parent's buffer) nor RLIMIT_FSIZE (which bounds files, not
    pipes) constrains that — a snippet printing gigabytes would grow the *server*
    unboundedly. Redirected to a file, the same write is bounded by RLIMIT_FSIZE
    at the kernel, and we read back only a tail.
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
            reason=(f"sandbox for {version} not set up — run scripts/setup_sandbox.sh"),
        )
    if not code.strip():
        return ExecResult(False, True, reason="empty code snippet")

    # Track handles as they are created so the cleanup below covers a partial
    # failure too: if the temp dir fills after the first file is made, the
    # already-created ones must still be closed and unlinked.
    handles: List["tempfile._TemporaryFileWrapper"] = []
    try:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
        handles.append(tmp)
        out_f = tempfile.NamedTemporaryFile(suffix=".out", delete=False)
        handles.append(out_f)
        err_f = tempfile.NamedTemporaryFile(suffix=".err", delete=False)
        handles.append(err_f)
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
            stdout=out_f,
            stderr=err_f,
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
        # start_new_session makes the child its own session/group leader, so the
        # group id *is* its pid. Capture it now: once the child exits, and
        # certainly once it is reaped, os.getpgid(proc.pid) raises ESRCH — so
        # deriving the group later would silently degrade to a no-op.
        pgid = proc.pid if posix else None
        try:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_group(pgid) if posix else _kill_process_tree(proc, posix)
                # Reap after the kill, bounded: a snippet that setsid-escapes the
                # group (double-fork daemon) could otherwise hold us forever.
                # There is no pipe to drain — output is already on disk.
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    pass
                return ExecResult(
                    False,
                    True,
                    returncode=proc.returncode,
                    stdout=_read_tail(out_f.name),
                    stderr=_read_tail(err_f.name),
                    reason=f"timed out after {timeout}s",
                )
            passed = proc.returncode == 0
            return ExecResult(
                passed=passed,
                available=True,
                returncode=proc.returncode,
                stdout=_read_tail(out_f.name),
                stderr=_read_tail(err_f.name),
                reason=_exit_reason(proc.returncode, passed),
            )
        finally:
            # Sweep the group on EVERY path, not just the timeout. A snippet can
            # spawn a background process and then exit 0 in milliseconds; the
            # direct child is gone, so the wall-clock timeout never fires and
            # nothing would ever reap the descendant. It would outlive the
            # request, holding the (already unlinked) capture files open so their
            # disk is never reclaimed.
            if posix:
                _kill_group(pgid)
    finally:
        for handle in handles:
            try:
                handle.close()
            except OSError:  # pragma: no cover
                pass
            try:
                os.unlink(handle.name)
            except OSError:  # pragma: no cover
                pass


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    demo = "from pydantic import BaseModel\nclass M(BaseModel):\n    x: int\nprint(M(x=1).model_dump())\n"
    print("v2:", grade(demo, "v2").to_dict())
    print("v1:", grade(demo, "v1").to_dict())
    sys.exit(0)
