"""Turn JSONL into a readable indented JSON file (array of objects).

JSONL is one JSON object per line, hard to read in an editor. This script
reads JSONL and writes a single JSON array with indent=2 (or prints to stdout).

    python pretty_jsonl.py gsm_hard_data/gsm_hard_graphs.jsonl
    python pretty_jsonl.py gsm_hard_data/gsm_hard_graphs.jsonl -o gsm_hard_data/gsm_hard_graphs.json
    python pretty_jsonl.py --stdout gsm_hard_data/gsm_hard_traces.jsonl | less
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="path to .jsonl")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output .json path (default: same name as input with .json extension)",
    )
    p.add_argument("--stdout", action="store_true", help="print to stdout instead of writing a file")
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Not found: {args.input}")

    rows: list[object] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    text = json.dumps(rows, indent=2, ensure_ascii=False) + "\n"

    if args.stdout:
        sys.stdout.write(text)
        return

    out = args.out if args.out is not None else args.input.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} records)")


if __name__ == "__main__":
    main()
