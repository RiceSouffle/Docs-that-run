"""Sandbox + version-lock tests.

These are skipped automatically when the venvs aren't set up, so `pytest` stays
green on a machine that hasn't run scripts/setup_sandbox.sh. In CI the venvs are
built first, so these run for real and quantify how many golden checks are
crisply version-locked (pass on target, fail on the other version)."""

import os
import sys
import time

import pytest

from docsthatrun.evals.run_evals import load_golden
from docsthatrun.sandbox import grade, sandbox_available

_SANDBOX = sandbox_available("v1") and sandbox_available("v2")
needs_sandbox = pytest.mark.skipif(not _SANDBOX, reason="sandbox venvs not set up")

_OTHER = {"v1": "v2", "v2": "v1"}


@needs_sandbox
def test_all_golden_reference_snippets_pass_on_target():
    for item in load_golden():
        res = grade(item.check, item.version)
        assert res.passed, f"{item.id} failed on target {item.version}: {res.stderr[-300:]}"


@needs_sandbox
def test_at_least_half_golden_are_crisply_version_locked():
    """A crisply version-locked check fails on the *other* version. Some v1 APIs
    survive as deprecated shims in v2, so not all pairs are crisp — we assert a
    healthy majority and print the exact rate for the writeup."""
    locked = 0
    total = 0
    for item in load_golden():
        total += 1
        other = grade(item.check, _OTHER[item.version])
        if not other.passed:
            locked += 1
    rate = locked / total
    print(f"\nversion-locked: {locked}/{total} = {rate:.0%}")
    assert rate >= 0.5, f"only {rate:.0%} of golden checks are version-locked"


# ---- resource limits (defence-in-depth) ------------------------------------

@needs_sandbox
def test_cpu_limit_kills_infinite_loop():
    """A CPU-bound infinite loop is stopped by RLIMIT_CPU (SIGXCPU) well before
    the much larger wall-clock timeout — so a hot loop can't burn a core."""
    res = grade("while True:\n    pass\n", "v2", timeout=30, cpu_seconds=2)
    assert not res.passed


@needs_sandbox
def test_file_size_limit_caps_writes(tmp_path):
    """RLIMIT_FSIZE stops a snippet from filling the disk."""
    target = tmp_path / "big.bin"
    res = grade(
        f"open(r'{target}', 'wb').write(b'x' * (50 * 1024 * 1024))\n",
        "v2",
        file_mb=2,
    )
    assert not res.passed
    size = os.path.getsize(target) if target.exists() else 0
    assert size <= 3 * 1024 * 1024, f"wrote {size} bytes despite a 2 MB cap"


@needs_sandbox
@pytest.mark.skipif(sys.platform == "darwin", reason="RLIMIT_AS unreliable on macOS")
def test_memory_limit_contains_allocation():
    """On Linux, RLIMIT_AS caps address space so a giant allocation fails."""
    res = grade("x = bytearray(3 * 1024 * 1024 * 1024)\n", "v2", memory_mb=256)
    assert not res.passed


@needs_sandbox
def test_argv_matches_direct_execution():
    """The rlimit launcher must leave the snippet with argv == [path] (len 1),
    exactly as `python file.py` would — else argparse/argv snippets false-fail."""
    assert grade("import sys\nassert len(sys.argv) == 1\n", "v2").passed
    assert grade("import argparse\nargparse.ArgumentParser().parse_args()\n", "v2").passed


@needs_sandbox
def test_flood_of_stdout_does_not_balloon_this_process():
    """Output volume is the one axis the rlimits used to miss: stdout went
    through a pipe into an unbounded parent buffer, so a snippet printing
    gigabytes grew the *server*, not the sandbox. Output now goes to a file
    (bounded by RLIMIT_FSIZE) and only a tail is read back."""
    import resource

    def rss_bytes():
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r if sys.platform == "darwin" else r * 1024

    before = rss_bytes()
    res = grade(
        "import sys\nbuf='x'*(1<<20)\nfor _ in range(256): sys.stdout.write(buf)\n",
        "v2",
        file_mb=2,
    )
    growth_mb = (rss_bytes() - before) / (1024 * 1024)
    # 256 MiB written; the parent must not have absorbed it.
    assert growth_mb < 64, f"parent grew {growth_mb:.0f} MB reading child output"
    assert len(res.stdout) <= 64 * 1024
    assert not res.passed  # RLIMIT_FSIZE stops the flood, so the snippet fails


@needs_sandbox
@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
def test_background_process_does_not_outlive_a_passing_grade():
    """The snippet's process group must be swept on EVERY path, not just the
    timeout. A snippet that spawns a background process and exits 0 finishes in
    milliseconds, so the wall-clock timeout never fires — previously nothing
    ever reaped the descendant and it outlived the request."""
    import subprocess as sp

    marker = "docsthatrun_orphan_probe"
    snippet = (
        "import subprocess, sys\n"
        f"subprocess.Popen([sys.executable, '-c', \"import time; time.sleep(120)  # {marker}\"])\n"
        "assert 1 + 1 == 2\n"
    )

    def survivors():
        out = sp.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout
        return [p for p in out.split() if p]

    res = grade(snippet, "v2")
    assert res.passed  # the snippet itself is fine; that's the point
    time.sleep(0.5)
    alive = survivors()
    for pid in alive:  # never leave strays behind, even if we're about to fail
        try:
            os.kill(int(pid), 9)
        except OSError:
            pass
    assert not alive, f"{len(alive)} sandbox descendant(s) outlived a passing grade"


@needs_sandbox
def test_no_temp_file_leak_on_any_exit_path():
    """Capturing output to files means three temp files per grade. Every exit
    path — success, failure, timeout, CPU kill, empty code — must unlink all of
    them, or a long-running server slowly fills its temp directory."""
    import glob
    import tempfile

    td = tempfile.gettempdir()

    def temp_count():
        return sum(
            len(glob.glob(os.path.join(td, pat)))
            for pat in ("tmp*.out", "tmp*.err", "tmp*.py")
        )

    before = temp_count()
    grade("assert 1 == 1\n", "v2")                       # pass
    grade("raise ValueError('x')\n", "v2")               # non-zero exit
    grade("import time\ntime.sleep(30)\n", "v2", timeout=2)          # wall timeout
    grade("while True: pass\n", "v2", timeout=30, cpu_seconds=1)     # CPU kill
    grade("   \n", "v2")                                 # empty snippet
    assert temp_count() == before


@needs_sandbox
def test_v2_probe_requires_pydantic_settings():
    """The v2 golden set uses pydantic_settings, so a v2 venv without it is not
    usable — reporting it available charges the ModuleNotFoundError to answer
    quality instead of to the environment."""
    from docsthatrun.sandbox import _PROBE_IMPORTS

    assert "pydantic_settings" in _PROBE_IMPORTS["v2"]
    assert "pydantic_settings" not in _PROBE_IMPORTS["v1"]  # v1 has no such split


def test_signal_deaths_get_an_explanatory_reason():
    """A signal kill leaves no stderr, so `reason` is the only explanation the
    caller ever gets — "non-zero exit" told them nothing."""
    import signal as sig

    from docsthatrun.sandbox import _exit_reason

    assert _exit_reason(0, True) == "ok"
    assert _exit_reason(1, False) == "non-zero exit"
    assert "CPU" in _exit_reason(-sig.SIGXCPU, False)
    assert "file-size" in _exit_reason(-sig.SIGXFSZ, False)
    assert _exit_reason(None, False) == "non-zero exit"


def test_sandbox_available_does_not_cache_failure(tmp_path, monkeypatch):
    """A probe that races `make sandbox` (venv exists, pydantic mid-install)
    must not disable grading for the process lifetime: failure is re-probed,
    success is cached. Runs without real venvs — the probe python is faked."""
    import docsthatrun.sandbox as sb

    monkeypatch.setattr(sb, "_IMPORT_OK", {})
    monkeypatch.setattr(sb, "_IMPORT_FAIL_AT", {})
    failing = tmp_path / "python_failing"
    failing.write_text("#!/bin/sh\nexit 1\n")
    failing.chmod(0o755)
    monkeypatch.setitem(sb.VENV_PYTHON, "v2", str(failing))
    assert sb.sandbox_available("v2") is False
    assert "v2" not in sb._IMPORT_OK  # the failure was NOT cached as a verdict

    # ...the install finishes (import now succeeds): availability self-heals
    # once the re-probe cooldown has elapsed.
    working = tmp_path / "python_working"
    working.write_text("#!/bin/sh\nexit 0\n")
    working.chmod(0o755)
    monkeypatch.setitem(sb.VENV_PYTHON, "v2", str(working))
    sb._IMPORT_FAIL_AT.clear()  # simulate the cooldown expiring
    assert sb.sandbox_available("v2") is True
    assert sb._IMPORT_OK.get("v2") is True  # success IS cached


def test_failed_probe_is_not_reforked_on_every_call(tmp_path, monkeypatch):
    """/health and /ready probe both versions per request. With a half-built
    venv that used to mean a process spawn per version per request — a fork
    storm under a monitoring poll. Failures are rate-limited, not re-probed."""
    import docsthatrun.sandbox as sb

    monkeypatch.setattr(sb, "_IMPORT_OK", {})
    monkeypatch.setattr(sb, "_IMPORT_FAIL_AT", {})
    failing = tmp_path / "python_failing"
    failing.write_text("#!/bin/sh\nexit 1\n")
    failing.chmod(0o755)
    monkeypatch.setitem(sb.VENV_PYTHON, "v2", str(failing))

    spawns = []
    real_run = sb.subprocess.run
    monkeypatch.setattr(
        sb.subprocess, "run", lambda *a, **k: (spawns.append(1), real_run(*a, **k))[1]
    )
    for _ in range(20):
        assert sb.sandbox_available("v2") is False
    assert len(spawns) == 1, f"probed {len(spawns)} times; expected 1 within the cooldown"
