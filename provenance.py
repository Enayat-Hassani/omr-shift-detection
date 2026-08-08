#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What produced a results file.

Everything in `results/` is committed and REPRODUCE.md presents it as something
a reader can check. Until now nothing in those files recorded which code wrote
them, so a stale result and a fresh one were indistinguishable on inspection.
That is not hypothetical: a determinism check in this project once overwrote a
committed results file with a tiny-n run, and it was caught by comparing against
a copy kept outside the repository rather than by anything in the file itself.

WHAT IS STAMPED, AND WHY IT IS NOT THE COMMIT

The stamp is a fingerprint of the source files the run depends on, not the
commit SHA. A SHA changes when history is rewritten -- an amend, a rebase, a
squash -- none of which alter a single line of the code that produced the
numbers, and all of which would falsely invalidate every stamp in the
repository. A content hash answers the question actually being asked: was this
result produced by the code now in the tree?

The commit is recorded as well, but as context. The fingerprint is the evidence.

WHAT IS DELIBERATELY ABSENT

No wall-clock timestamp. Regenerating an unchanged result must produce a
byte-identical file, so that `git diff` being empty is itself the check. A
timestamp would put a diff in every regeneration and destroy that property. The
question a reader has is 'does this match the code', not 'what time was it'.
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from typing import Dict, List, Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _source_files() -> List[str]:
    """The modules this run actually loaded, from inside the repository.

    Not every source file in the tree. A fingerprint over everything would be
    simpler, but it invalidates a result whenever unrelated code changes -- edit
    the cohort screen and the corpus benchmark's stamp goes stale, though the
    two share nothing and re-running would reproduce the file byte for byte. A
    stamp that cries wolf gets ignored, and a ninety-minute benchmark cannot be
    re-run to silence a false alarm.

    Reading `sys.modules` at the moment of writing gives the real dependency
    set: what was imported is what could have affected the numbers. Anything
    outside the repository is excluded -- the interpreter and third-party
    versions are recorded separately, and hashing site-packages would make the
    stamp machine-specific.
    """
    out = set()
    for mod in list(sys.modules.values()):
        path = getattr(mod, "__file__", None)
        if not path or not path.endswith(".py"):
            continue
        path = os.path.abspath(path)
        if not path.startswith(_ROOT + os.sep):
            continue
        rel = os.path.relpath(path, _ROOT)
        # Tests and tools read results; they never produce them.
        if rel.split(os.sep)[0] in ("tests", "tools"):
            continue
        out.add(rel)
    return sorted(out)


def code_fingerprint() -> str:
    """SHA-256 over the loaded source files, path included.

    Path as well as content, so that moving a file registers as a change. Twelve
    hex characters are kept: enough that a collision is not a practical concern
    for a repository of this size, short enough to sit in a text header.

    Two results stamped with different fingerprints were produced by different
    code. Two stamped with the same fingerprint were produced by the same code
    AND the same set of imports, which is the claim that matters.
    """
    h = hashlib.sha256()
    for rel in _source_files():
        h.update(rel.encode())
        with open(os.path.join(_ROOT, rel), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(("git", "-C", _ROOT) + args, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def stamp(**params) -> Dict:
    """The provenance record. `params` are the run's own arguments."""
    commit = _git("rev-parse", "--short", "HEAD")
    # Exactly the files in the fingerprint, and nothing else.
    #
    # Two narrower-than-obvious choices. The whole working tree is wrong: a run
    # writes into `results/`, so the tree is always dirty by the time the stamp
    # is taken, and every results file ever produced would carry 'uncommitted
    # changes'. All `*.py` is also wrong, and less obviously so: editing a file
    # under `tools/` while a benchmark runs would mark that benchmark's source
    # as uncommitted, though the tooling is excluded from the fingerprint and
    # cannot affect a number. Both were observed here before this was narrowed.
    files = _source_files()
    dirty = _git("status", "--porcelain", "--", *files) if files else ""
    return {
        "code_fingerprint": code_fingerprint(),
        "n_source_files": len(_source_files()),
        # Context, not evidence. See the module docstring.
        "commit": commit or "not a git checkout",
        "code_committed": (dirty == "") if dirty is not None else None,
        "python": platform.python_version(),
        "numpy": _numpy_version(),
        "params": {k: params[k] for k in sorted(params)},
    }


def _numpy_version() -> str:
    try:
        import numpy  # noqa: F401
        return numpy.__version__
    except Exception:
        return "absent"


def stamp_text(width: int = 78, **params) -> str:
    """The same record as a comment block, for the plain-text results."""
    s = stamp(**params)
    L = ["-" * width, "PROVENANCE", "-" * width]
    L.append(f"  code fingerprint : {s['code_fingerprint']}  "
             f"(over {s['n_source_files']} loaded modules)")
    L.append(f"  commit           : {s['commit']}"
             + ("" if s["code_committed"] in (None, True)
                else "  (SOURCE HAD UNCOMMITTED CHANGES)"))
    L.append(f"  python           : {s['python']}    numpy: {s['numpy']}")
    if s["params"]:
        rendered = ", ".join(f"{k}={v}" for k, v in s["params"].items())
        L.append(f"  parameters       : {rendered}")
    L.append("  Regenerating from unchanged code reproduces this file exactly;")
    L.append("  an empty `git diff` is the check. See REPRODUCE.md.")
    L.append("-" * width)
    return "\n".join(L)


def write_text(path: str, text: str, **params) -> None:
    """Write a plain-text result with the provenance block appended."""
    with open(path, "w") as fh:
        fh.write(text.rstrip("\n") + "\n\n" + stamp_text(**params) + "\n")


def write_json(path: str, obj, **params) -> None:
    """Write a JSON result with the provenance record under `_provenance`.

    Leading underscore so it sorts and reads as metadata rather than as one of
    the measurements. A list payload is wrapped, since a bare list has nowhere
    to carry a record.
    """
    import json
    if isinstance(obj, dict):
        payload = {"_provenance": stamp(**params), **obj}
    else:
        payload = {"_provenance": stamp(**params), "results": obj}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.write("\n")


def verify(expected: str) -> bool:
    """Whether the tree still matches a fingerprint a results file recorded."""
    return code_fingerprint() == expected


if __name__ == "__main__":
    import json
    json.dump(stamp(), sys.stdout, indent=2)
    sys.stdout.write("\n")
