#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audits all numerical claims in documentation against source result files.

WHY THIS EXISTS
Published figures frequently drift from underlying raw results. This script
automates verification to prevent stale or unverified claims.

WHY A MANIFEST
Parsing prose directly produces false positives. Explicit hand-registration in
`figures_manifest.json` ensures every claim has an intentional, verifiable source.

STATUS CODES (Exits non-zero on failure)
- MISMATCH: Document figure differs from the source file.
- MISSING: Registered claim was removed or reworded in the document.
- NO SOURCE: Result file or expected key does not exist.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
MANIFEST = os.path.join(_HERE, "figures_manifest.json")


# ---------------------------------------------------------------------------
# Getting the measured value out of a results file
# ---------------------------------------------------------------------------

def _dotted(obj: Any, path: str) -> Any:
    """Walk a dotted path through nested dicts and lists."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            if part not in cur:
                raise KeyError(part)
            cur = cur[part]
    return cur


def measured(entry: Dict) -> float:
    """The live value, from JSON by key path or from text by regex."""
    src = os.path.join(_ROOT, entry["source"])
    if not os.path.exists(src):
        raise FileNotFoundError(entry["source"])

    if src.endswith(".json"):
        with open(src) as fh:
            return float(_dotted(json.load(fh), entry["path"]))

    with open(src) as fh:
        text = fh.read()
    m = re.search(entry["pattern"], text, re.MULTILINE)
    if not m:
        raise KeyError(f"pattern {entry['pattern']!r} not found in {entry['source']}")
    return float(m.group(1).replace(",", ""))


# ---------------------------------------------------------------------------
# Rendering it the way the document writes it
# ---------------------------------------------------------------------------

FORMATS = {
    "pct0": lambda v: f"{v * 100:.0f}%",
    "pct1": lambda v: f"{v * 100:.1f}%",
    "pct2": lambda v: f"{v * 100:.2f}%",
    "num2": lambda v: f"{v:.2f}",
    "num3": lambda v: f"{v:.3f}",
    "num4": lambda v: f"{v:.4f}",
    "int": lambda v: f"{int(round(v))}",
    # For sources that already print a percentage: the value is 33, not 0.33.
    "aspct0": lambda v: f"{v:.0f}%",
    "aspct1": lambda v: f"{v:.1f}%",
    "int_comma": lambda v: f"{int(round(v)):,}",
}


def check(entry: Dict) -> Tuple[str, str]:
    """Returns (status, detail). Status is ok / MISMATCH / MISSING / NO SOURCE."""
    doc_path = os.path.join(_ROOT, entry["doc"])
    if not os.path.exists(doc_path):
        return "NO SOURCE", f"document {entry['doc']} not found"
    with open(doc_path) as fh:
        doc = fh.read()

    try:
        value = measured(entry)
    except FileNotFoundError as exc:
        return "NO SOURCE", f"results file missing: {exc}"
    except (KeyError, IndexError, ValueError) as exc:
        return "NO SOURCE", f"cannot read the measurement: {exc}"

    rendered = FORMATS[entry["format"]](value)

    # The claim must appear in the document as the document writes numbers,
    # bounded so that 0.28 does not match inside 0.288.
    if re.search(r"(?<![\d.])" + re.escape(rendered) + r"(?![\d])", doc):
        return "ok", rendered

    claimed = entry.get("claimed")
    if claimed is not None and re.search(
            r"(?<![\d.])" + re.escape(claimed) + r"(?![\d])", doc):
        return "MISMATCH", f"document says {claimed}, results give {rendered}"
    return "MISSING", (f"results give {rendered}, which does not appear in "
                       f"{entry['doc']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--update", action="store_true",
                    help="rewrite each entry's `claimed` field to the measured "
                         "value; review the diff, it does not touch the documents")
    args = ap.parse_args()

    with open(args.manifest) as fh:
        entries: List[Dict] = json.load(fh)["claims"]

    width = max((len(e["label"]) for e in entries), default = 20)
    bad = 0
    by_doc: Dict[str, List] = {}
    for e in entries:
        by_doc.setdefault(e["doc"], []).append(e)

    print("=" * 78)
    print("PUBLISHED FIGURES vs RESULTS FILES")
    print("=" * 78)
    for doc in sorted(by_doc):
        print(f"\n{doc}")
        print("-" * 78)
        for e in by_doc[doc]:
            status, detail = check(e)
            if status != "ok":
                bad += 1
            mark = "  ok  " if status == "ok" else f" {status} "
            print(f"  [{mark:^10}] {e['label']:<{width}}  {detail}")
            if args.update:
                e["claimed"] = detail if status == "ok" else e.get("claimed")

    print("\n" + "=" * 78)
    if bad:
        print(f"{bad} of {len(entries)} published figures do not check out.")
        print("A MISMATCH means one of the two is stale. Find out which before")
        print("editing either.")
    else:
        print(f"All {len(entries)} published figures match their results files.")
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
