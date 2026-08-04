"""Session-wide test setup.

This file exists to close two holes that only showed up as *someone else's*
problem — a surprise bill, or a flaky CI run — rather than as a failing test.

1. **No test may reach the real API.** ``tests/test_app.py`` used
   ``os.environ.setdefault``, which is a no-op for a developer who followed
   README's instructions and exported ``ANTHROPIC_API_KEY`` and
   ``DOCSTHATRUN_LLM=anthropic``. For them, ``pytest`` made real, billed Claude
   calls. Here the values are *assigned*, not defaulted, and the key is removed
   from the environment entirely so even a mistake elsewhere can't spend money.

2. **Environment must be set before the first project import.** ``config.py``
   builds its ``settings`` singleton at import time. Previously the env only
   landed in time because ``test_app.py`` happens to sort first alphabetically;
   ``-k``, ``pytest-xdist``, ``pytest-randomly``, or simply a new test file
   sorting earlier would import ``docsthatrun.config`` first,
   ``DOCSTHATRUN_RATE_RPM=0`` would never apply, and the ~30 requests in
   ``test_app.py`` would blow past the burst cap as intermittent 429s. conftest
   is imported before any test module, so this is the right place for it.

The autouse fixtures below restore global state that individual tests mutate,
so test order can't change results.
"""

import logging
import os

import pytest

# --- 1 & 2: must run at import, before anything imports docsthatrun.config ---
os.environ["DOCSTHATRUN_LLM"] = "mock"  # never the real client
os.environ["DOCSTHATRUN_RATE_RPM"] = "0"  # deterministic: no rate limiting
os.environ.pop("ANTHROPIC_API_KEY", None)  # belt and braces: nothing to bill


@pytest.fixture(autouse=True)
def _restore_root_logging():
    """Undo ``configure_logging``'s handler surgery.

    It removes every root handler and installs its own, which leaked into every
    later test (and into pytest's own capture) for the rest of the session.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


@pytest.fixture(autouse=True)
def _reset_answer_cache():
    """Start every test with an empty answer cache.

    Tests were calling ``answer_cache.clear()`` by hand, which meant a new test
    that forgot to inherited whatever the previous one had cached.
    """
    try:
        from app.main import answer_cache
    except Exception:  # fastapi not installed; nothing to reset
        yield
        return
    answer_cache.clear()
    yield
    answer_cache.clear()


@pytest.fixture(autouse=True)
def _reset_sandbox_probe_memo():
    """Keep the sandbox availability memo from leaking across tests.

    Several tests monkeypatch ``VENV_PYTHON`` at a fake interpreter; without
    this, a cached success or failure from one of them changes what a later
    test sees.
    """
    import docsthatrun.sandbox as sb

    ok = dict(sb._IMPORT_OK)
    fail_at = dict(sb._IMPORT_FAIL_AT)
    yield
    sb._IMPORT_OK.clear()
    sb._IMPORT_OK.update(ok)
    sb._IMPORT_FAIL_AT.clear()
    sb._IMPORT_FAIL_AT.update(fail_at)


@pytest.fixture(scope="session")
def sandbox_ready() -> bool:
    """Whether both pinned venvs are usable.

    A fixture rather than a module-level constant so the probe subprocesses run
    at test time, not during collection — importing a test module should not
    fork anything.
    """
    from docsthatrun.sandbox import sandbox_available

    return sandbox_available("v1") and sandbox_available("v2")


@pytest.fixture
def require_sandbox(sandbox_ready):
    """Skip unless both pinned venvs are usable.

    Used via ``pytest.mark.usefixtures`` so the decision is made when the test
    runs, not when its module is imported.
    """
    if not sandbox_ready:
        pytest.skip("sandbox venvs not set up — run scripts/setup_sandbox.sh")
