"""A plain-ASCII run header for the notebook.

Deliberately plain: no ANSI colour, which renders as escape garbage in some
notebook frontends, and no branding.
"""

import platform
import sys

_WIDTH = 72

TITLE = "Intro to Scaled Qualitative Analysis"
STAGES = "normalize -> read into a table -> classify -> visualize"
LICENSE_LINE = (
    "Code: BSD 3-Clause. Data: not BSD -- see SAMPLE_DATA.md for its notice."
)


def _versions():
    """Report versions of the packages that matter, without importing hard."""
    found = []
    for name, module_name in [
        ("pandas", "pandas"), ("numpy", "numpy"),
        ("matplotlib", "matplotlib"), ("seaborn", "seaborn"),
        ("scipy", "scipy"), ("wordcloud", "wordcloud"),
    ]:
        try:
            module = __import__(module_name)
            found.append(f"{name} {getattr(module, '__version__', '?')}")
        except ImportError:
            found.append(f"{name} MISSING")
    return found


def print_header(stage=None):
    """Print the run header. Pass ``stage`` to label the current step."""
    python_version = platform.python_version()

    print("=" * _WIDTH)
    print(TITLE)
    print("-" * _WIDTH)
    print(f"Stages : {STAGES}")
    if stage:
        print(f"Now    : {stage}")
    print(f"Python : {python_version} ({sys.platform})")

    versions = _versions()
    print(f"Packages: {', '.join(versions[:3])}")
    print(f"          {', '.join(versions[3:])}")
    print("-" * _WIDTH)
    print(LICENSE_LINE)
    print("=" * _WIDTH)


__all__ = ["print_header", "TITLE", "STAGES", "LICENSE_LINE"]
