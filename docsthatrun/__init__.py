"""DocsThatRun — version-aware documentation RAG with execution-graded answers.

The differentiator: generated code snippets are not graded on "looks plausible".
They are executed against the *pinned* version of the target library in an
isolated venv and scored pass/fail. Because Pydantic v1 and v2 removed several
names outright (imports raise), running v2-flavoured code against the v1 sandbox
fails — so execution grading *is* the version-correctness check.
"""

# The single source of truth for the project version. pyproject.toml reads this
# attribute (`[tool.setuptools.dynamic]`) and app/main.py imports it, so there is
# one place to change. They used to be three separate literals, and this one had
# been left behind at 0.1.0 for two releases while /health reported 0.3.2.
__version__ = "0.4.0"
