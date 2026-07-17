"""Sandbox + version-lock tests.

These are skipped automatically when the venvs aren't set up, so `pytest` stays
green on a machine that hasn't run scripts/setup_sandbox.sh. In CI the venvs are
built first, so these run for real and quantify how many golden checks are
crisply version-locked (pass on target, fail on the other version)."""

import os
import sys

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
