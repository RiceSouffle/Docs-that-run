"""Corpus loading and its error paths.

`load_corpus` goes to real trouble to turn a JSONL problem into a message
naming the file and the *real* line number — and none of it was tested, so the
messages could have rotted without anyone noticing. The shipped data is clean,
which is exactly why these need synthetic files.
"""

import pytest

from docsthatrun.corpus import load_corpus, tokenize

_GOOD = '{"id":"c1","version":"v2","topic":"t","title":"T","text":"body","code":""}'


def _corpus(tmp_path, *lines):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_loads_a_well_formed_file(tmp_path):
    chunks = load_corpus(_corpus(tmp_path, _GOOD))
    assert [c.id for c in chunks] == ["c1"]


def test_blank_lines_are_skipped(tmp_path):
    assert len(load_corpus(_corpus(tmp_path, _GOOD, "", "   "))) == 1


def test_malformed_json_names_the_file_and_the_real_line(tmp_path):
    # The real line number matters: blank lines are skipped, so a counter over
    # *parsed* records would point at the wrong place.
    path = _corpus(tmp_path, _GOOD, "", "{not json}")
    with pytest.raises(ValueError) as exc:
        load_corpus(path)
    assert "corpus.jsonl" in str(exc.value)
    assert "corpus.jsonl:3" in str(exc.value)


def test_missing_required_field_is_reported_not_a_bare_keyerror(tmp_path):
    path = _corpus(tmp_path, '{"version":"v2","topic":"t","title":"T","text":"body"}')
    with pytest.raises(ValueError) as exc:
        load_corpus(path)
    assert "missing required field" in str(exc.value)
    assert "id" in str(exc.value)


@pytest.mark.parametrize("value", ["123", '"a string"', "[]", "null"])
def test_non_object_json_line_is_a_clear_error(tmp_path, value):
    """A bare scalar or array parses fine, then blew up on subscripting with a
    TypeError/AttributeError that skipped the file:line message entirely."""
    with pytest.raises(ValueError) as exc:
        load_corpus(_corpus(tmp_path, value))
    assert "corpus.jsonl" in str(exc.value)


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate chunk id"):
        load_corpus(_corpus(tmp_path, _GOOD, _GOOD))


def test_unknown_version_is_rejected(tmp_path):
    bad = _GOOD.replace('"v2"', '"v3"')
    with pytest.raises(ValueError, match="bad version"):
        load_corpus(_corpus(tmp_path, bad))


def test_both_is_a_valid_chunk_version(tmp_path):
    # "both" is not a query version but is a valid chunk version — a chunk that
    # applies regardless of which library version was asked about.
    chunk = load_corpus(_corpus(tmp_path, _GOOD.replace('"v2"', '"both"')))[0]
    assert chunk.version == "both"


def test_tokenize_splits_snake_case_identifiers():
    # `model_dump` must also match a query saying "dump", or version-specific
    # API names would only ever be found by exact spelling.
    assert set(tokenize("model_dump()")) >= {"model_dump", "model", "dump"}
