"""Sandbox + version-lock tests.

These are skipped automatically when the venvs aren't set up, so `pytest` stays
green on a machine that hasn't run scripts/setup_sandbox.sh. In CI the venvs are
built first, so these run for real and quantify how many golden checks are
crisply version-locked (pass on target, fail on the other version)."""

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
