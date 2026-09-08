"""The version is read from installed metadata, never written down twice.

Two published releases of this package both reported ``__version__ ==
"0.1.0"``, because the literal in ``evalaudit/__init__.py`` was edited on a
different schedule from ``pyproject.toml``. Anyone reproducing a consulting
number records the version first, so it has to be the real one.

There is no test here comparing ``__version__`` to ``pyproject.toml`` or to
``importlib.metadata.version``. Both would be tautologies. ``__version__``
*is* that metadata call, and the build backend generates the metadata from
``pyproject.toml``, so all three are the same value by construction and no
such comparison can fail. One of them failed once, against the old literal,
which is exactly what made it look like a real test.

What can fail is the literal coming back, and the fallback being wrong. So
those are the two things tested.
"""

import pathlib
import re
import subprocess
import sys

import evalaudit


INIT = pathlib.Path(evalaudit.__file__)
REPO_ROOT = INIT.parent.parent

# Any dotted numeric triple. Deliberately blunt: it catches "0.1.0" whether
# it was assigned to __version__, parked in a fallback, or left in a comment
# for later. A real version number has no business in this file at all.
_VERSION_SHAPED = re.compile(r"\d+\.\d+\.\d+")


def test_init_carries_no_version_literal():
    """No dotted numeric triple anywhere in the file.

    A fallback such as ``except PackageNotFoundError: __version__ = "0.1.0"``
    is the same bug with an extra branch in front of it, and this catches
    that too.
    """
    source = INIT.read_text(encoding="utf-8")
    found = _VERSION_SHAPED.findall(source)
    assert not found, (
        f"{INIT.name} contains version-shaped literal(s) {found}. The "
        f"version must come from importlib.metadata, not from a string in "
        f"this file. A hard-coded fallback counts."
    )


def test_version_falls_back_to_unknown_when_the_package_is_not_installed():
    """The branch that only runs from a bare source tree.

    Run in a subprocess with ``importlib.metadata.version`` replaced before
    ``evalaudit`` is imported, because the module binds that function at
    import time and the only honest way to exercise the fallback is to
    import the package with it already broken. A subprocess also keeps a
    half-imported evalaudit out of the rest of the session.

    The regex test above does not cover this. A fallback of
    ``"not-installed"`` carries no digits, passes that test, and fails this
    one.
    """
    program = (
        "import importlib.metadata as md\n"
        "def raise_not_found(name):\n"
        "    raise md.PackageNotFoundError(name)\n"
        "md.version = raise_not_found\n"
        "import evalaudit\n"
        "print(evalaudit.__version__)\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert finished.returncode == 0, (
        f"importing evalaudit without metadata failed:\n{finished.stderr}"
    )
    assert finished.stdout.strip() == "unknown", (
        f"expected 'unknown', got {finished.stdout.strip()!r}"
    )
