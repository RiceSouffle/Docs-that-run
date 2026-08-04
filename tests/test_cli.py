"""CLI smoke tests (offline MockClient, no execution to keep them fast)."""

import pytest

from docsthatrun.cli import main

_Q = "In Pydantic v2, how do I serialize a model instance to a dictionary?"
_UNANSWERABLE = "In Pydantic v2, how do I configure the Redis cache backend?"


def test_ask_answerable_exits_zero(capsys):
    rc = main(["ask", _Q, "--version", "v2", "--no-execute", "--client", "mock"])
    out = capsys.readouterr().out
    assert rc == 0
    # "cited:" specifically, not just the id appearing somewhere: the id also
    # shows up in the retrieval table below the answer, so a bare substring
    # check passed even with every citation dropped.
    cited_line = next(ln for ln in out.splitlines() if "cited:" in ln)
    assert "c_v2_dump" in cited_line


def test_ask_rejects_bad_version(capsys):
    rc = main(["ask", _Q, "--version", "v3", "--no-execute", "--client", "mock"])
    assert rc == 2


def test_ask_unanswerable_abstains(capsys):
    rc = main(["ask", _UNANSWERABLE, "--no-execute", "--client", "mock"])
    out = capsys.readouterr().out
    assert rc == 0 and "abstained" in out


def test_compare_shows_both_versions(capsys):
    rc = main(["compare", _Q, "--no-execute", "--client", "mock"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Pydantic v1" in out and "Pydantic v2" in out


def test_no_args_prints_help(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0 and "usage" in out.lower()


@pytest.mark.parametrize("bad", ["0", "-1", "999"])
def test_ask_rejects_out_of_range_top_k(bad, capsys):
    rc = main(["ask", _Q, "--top-k", bad, "--client", "mock"])
    assert rc == 2


def test_ask_exits_1_when_the_sandbox_grade_fails(monkeypatch, capsys):
    """Documented in GUIDE.md as what makes `ask` usable in a script, and never
    tested — every other CLI test passes --no-execute, so this line never ran."""
    import docsthatrun.cli as cli
    from docsthatrun.sandbox import ExecResult

    monkeypatch.setattr(cli, "sandbox_available", lambda v: True)
    monkeypatch.setattr(
        cli.AnswerResult,
        "execution_grade",
        lambda self: (
            setattr(self, "execution", ExecResult(passed=False, available=True, returncode=1)) or self.execution
        ),
    )
    code = cli.main(["ask", "In Pydantic v2, how do I serialize a model instance to a dictionary?", "--client", "mock"])
    assert code == 1


def test_ask_exits_0_when_the_grade_passes(monkeypatch):
    import docsthatrun.cli as cli
    from docsthatrun.sandbox import ExecResult

    monkeypatch.setattr(cli, "sandbox_available", lambda v: True)
    monkeypatch.setattr(
        cli.AnswerResult,
        "execution_grade",
        lambda self: (
            setattr(self, "execution", ExecResult(passed=True, available=True, returncode=0)) or self.execution
        ),
    )
    code = cli.main(["ask", "In Pydantic v2, how do I serialize a model instance to a dictionary?", "--client", "mock"])
    assert code == 0


def test_signal_killed_snippet_prints_a_reason_not_a_bare_fail(monkeypatch, capsys):
    """A CPU-limit or timeout kill is a SIGKILL, so it leaves no stderr at all.
    The CLI only printed stderr, so those rendered as a bare red FAIL with
    nothing explaining it — the exact case ExecResult.reason exists for."""
    import docsthatrun.cli as cli
    from docsthatrun.sandbox import ExecResult

    killed = ExecResult(
        passed=False,
        available=True,
        returncode=-24,
        stderr="",
        reason="killed: exceeded the CPU-time limit",
    )
    monkeypatch.setattr(cli, "sandbox_available", lambda v: True)
    monkeypatch.setattr(
        cli.AnswerResult,
        "execution_grade",
        lambda self: setattr(self, "execution", killed) or self.execution,
    )
    cli.main(["ask", "In Pydantic v2, how do I serialize a model instance to a dictionary?", "--client", "mock"])
    out = capsys.readouterr().out
    assert "CPU-time limit" in out
