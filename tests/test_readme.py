"""The README's code blocks have to reproduce.

The README is the page a stranger reads first, and a block that does not run
is the first thing a sceptical one tries. So every ``python`` block followed
by an output block is executed here and checked against what the README
claims it prints.

Output blocks are wrapped and trimmed with ``[...]``, because the full text
of an audit report does not belong in a README. The check is that every
fragment between the brackets appears verbatim in what the code really
prints, with whitespace collapsed so the line wrapping is free to change.

Blocks run in this process rather than a subprocess. Four subprocess starts
plus four numpy and pandas imports cost more than the rest of this file put
together, and the blocks are ordinary code with nothing to isolate.
"""

import contextlib
import io
import pathlib
import re

import pytest


README = pathlib.Path(__file__).resolve().parent.parent / "README.md"

# A python block, then the block showing what it prints. The bash install
# block has no output block after it and does not match.
_BLOCK = re.compile(r"```python\n(.*?)```\n+```\n(.*?)```", re.DOTALL)

# The elision marker used inside output blocks.
_CUT = "[...]"

# Every module the README demonstrates. If a block is added or lost, the
# count test below fails rather than the file quietly checking nothing.
_EXPECTED_BLOCKS = 4


def _collapse(text: str) -> str:
    """Whitespace-insensitive form, so rewrapping a block is not a failure."""
    return " ".join(text.split())


def _label(code: str) -> str:
    """A test id naming what the block demonstrates."""
    for line in code.splitlines():
        if line.startswith("from evalaudit import"):
            return line.split("import")[-1].strip().replace(", ", "+")
    return "block"


def _blocks():
    return _BLOCK.findall(README.read_text(encoding="utf-8"))


_CASES = _blocks()


def test_readme_still_has_its_blocks():
    """Guards the regex.

    A fence style the pattern stops matching would leave the parametrised
    test with nothing to run and no failure to show for it.
    """
    assert len(_CASES) == _EXPECTED_BLOCKS, (
        f"expected {_EXPECTED_BLOCKS} python blocks with output in README.md, "
        f"found {len(_CASES)}. Update _EXPECTED_BLOCKS if that was deliberate."
    )


@pytest.mark.parametrize(
    "code, shown", _CASES, ids=[_label(code) for code, _ in _CASES]
)
def test_readme_block_reproduces(code, shown):
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            exec(compile(code, str(README), "exec"), {"__name__": "__readme__"})
    except ImportError as exc:
        pytest.skip(
            f"README block needs {exc.name or 'a module'}, which is not "
            f"installed here"
        )

    printed = _collapse(captured.getvalue())
    fragments = [
        _collapse(part) for part in _collapse(shown).split(_CUT)
        if part.strip()
    ]
    assert fragments, "the output block has no text to check"

    missing = [f for f in fragments if f not in printed]
    assert not missing, (
        "README output has drifted from what the code prints.\n"
        + "\n".join(f"  missing: {f[:200]}" for f in missing)
        + f"\n  actual: {printed[:400]}"
    )
