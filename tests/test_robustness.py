"""Regression tests for the robustness fixes.

Each test pins a bug that was found by the adversarial bug-hunt and fixed:
- the LLM answer parser must degrade to an abstain on truncated/empty/malformed
  output instead of raising (llm._extract_json);
- the CI gate must fail (not silently pass) when the sandbox is up but nothing
  was gradable (run_evals.check_gate);
- the sandbox timeout must kill the whole process group, so a snippet that
  spawns a grandchild can't outlive the timeout (sandbox.grade).
"""

import time

import pytest

from docsthatrun.evals.run_evals import GATE, check_gate
from docsthatrun.llm import _extract_json
from docsthatrun.sandbox import grade, sandbox_available

# ---- LLM answer parsing (llm._extract_json) --------------------------------

def test_extract_json_parses_clean_object():
    obj = _extract_json('{"answer":"a","code":"c","citations":["x"],"abstained":false}')
    assert obj == {"answer": "a", "code": "c", "citations": ["x"], "abstained": False}


@pytest.mark.parametrize(
    "text",
    [
        "",                                   # empty text block (all thinking)
        "   \n  ",                            # whitespace only
        '{"answer": "To serialize you c',     # truncated mid-JSON (max_tokens)
        "not json at all",                    # non-JSON prose
        '["a", "b"]',                         # valid JSON but not an object
    ],
)
def test_extract_json_returns_none_instead_of_raising(text):
    # The old code called json.loads() directly and crashed the /ask request.
    assert _extract_json(text) is None


def test_extract_json_tolerates_fences_and_prose():
    fenced = "```json\n{\"answer\":\"a\",\"code\":\"\",\"citations\":[],\"abstained\":true}\n```"
    assert _extract_json(fenced)["abstained"] is True
    prose = 'Here you go:\n{"answer":"a","code":"","citations":[],"abstained":false} — done'
    assert _extract_json(prose)["answer"] == "a"


def test_extract_json_fallback_is_string_aware():
    # A brace inside a string value must not confuse the prose-stripping fallback.
    txt = 'Sure:\n{"answer":"x","code":"d = {","citations":[],"abstained":false}'
    assert _extract_json(txt)["code"] == "d = {"


# ---- CI gate (run_evals.check_gate) ----------------------------------------

def _base_report(executable_pct, sandbox=True):
    return {
        "retrieval": {"recall_at_5": 1.0, "mrr": 1.0},
        "sandbox_available": sandbox,
        "answers": {
            "executable_pct": executable_pct,
            "unanswerable_abstention": 1.0,
            "answerable_over_abstention": 0.0,
        },
    }


def test_gate_fails_when_sandbox_up_but_nothing_gradable():
    # executable_pct is None when gradable_count == 0. With the sandbox up that
    # is a regression, and the gate must catch it (it used to silently skip).
    failures = check_gate(_base_report(None, sandbox=True))
    assert any("gradable" in f for f in failures)


def test_gate_fails_on_low_executable_pct():
    failures = check_gate(_base_report(GATE["executable_pct_min"] - 0.1))
    assert any("executable_pct" in f for f in failures)


def test_gate_passes_on_healthy_report():
    assert check_gate(_base_report(1.0)) == []


def test_gate_ignores_executable_pct_when_sandbox_down():
    # No sandbox -> executable_pct is not enforced at all.
    assert check_gate(_base_report(None, sandbox=False)) == []


# ---- Sandbox process-group isolation (sandbox.grade) -----------------------

_SANDBOX = sandbox_available("v2")
needs_sandbox = pytest.mark.skipif(not _SANDBOX, reason="sandbox venvs not set up")


@needs_sandbox
def test_timeout_kills_spawned_grandchild(tmp_path):
    """A snippet that spawns a grandchild and blocks must not leave the
    grandchild running after the timeout — the whole process group is killed."""
    marker = tmp_path / "survived.txt"
    snippet = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c',\n"
        f"    \"import time; time.sleep(5); open(r'{marker}','w').write('x')\"])\n"
        "time.sleep(5)\n"
    )
    res = grade(snippet, "v2", timeout=1)
    assert not res.passed and "timed out" in res.reason
    time.sleep(5)  # wait past the grandchild's sleep
    assert not marker.exists(), "grandchild survived the sandbox timeout"
