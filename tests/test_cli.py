"""CLI smoke tests (offline MockClient, no execution to keep them fast)."""

import pytest

from docsthatrun.cli import main

_Q = "In Pydantic v2, how do I serialize a model instance to a dictionary?"
_UNANSWERABLE = "In Pydantic v2, how do I configure the Redis cache backend?"


def test_ask_answerable_exits_zero(capsys):
    rc = main(["ask", _Q, "--version", "v2", "--no-execute", "--client", "mock"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Answer" in out and "c_v2_dump" in out


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
