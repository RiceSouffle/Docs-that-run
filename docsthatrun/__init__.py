"""DocsThatRun — version-aware documentation RAG with execution-graded answers.

The differentiator: generated code snippets are not graded on "looks plausible".
They are executed against the *pinned* version of the target library in an
isolated venv and scored pass/fail. Because Pydantic v1 and v2 removed several
names outright (imports raise), running v2-flavoured code against the v1 sandbox
fails — so execution grading *is* the version-correctness check.
"""

__version__ = "0.1.0"
