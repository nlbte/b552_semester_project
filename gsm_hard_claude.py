"""Convert gsm_hard_traces.jsonl into reasoning graphs using the Claude API.

Drop-in replacement for traces_to_graphs.py — same input/output format,
same downstream compatibility with motif_analysis.py and visualize_results.py.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...

    python gsm_hard_claude.py
    python gsm_hard_claude.py --traces gsm_hard_data/gsm_hard_traces.jsonl --out gsm_hard_data/gsm_hard_graphs_claude.jsonl
    python gsm_hard_claude.py --limit 10   # test on first 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import anthropic
from pathlib import Path as _Path
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_Path(__file__).resolve().parent / ".env")
except ImportError:
    pass


MODEL = "claude-opus-4-6"

GRAPH_SYSTEM = """You are converting a model's math reasoning trace into a directed acyclic graph of steps.

Output a single JSON object only (no markdown fences, no commentary). Schema:

{
  "question_id": "<same as provided>",
  "steps": [
    {
      "id": "S1",
      "text": "short phrase describing this step",
      "depends_on": [],
      "op": "extract_fact | arithmetic | substitute | conclude | verify | other"
    }
  ],
  "final_answer": "<string, must match provided final_answer exactly>",
  "gold_answer": "<string, must match provided gold_answer exactly>",
  "is_correct": <true or false, must match provided is_correct>
}

Rules:
- Use step ids S1, S2, ... in order of logical flow.
- depends_on lists step ids this step builds on (empty for facts read from the question).
- Keep each text concise; split long rambling into multiple linked steps.
- op must be one of: extract_fact, arithmetic, substitute, conclude, verify, other.
- Preserve the exact final_answer, gold_answer, and is_correct values you are given — do not recompute them.
"""


def build_user_message(row: dict[str, Any]) -> str:
    qid = row.get("id", "")
    final_ans = str(row.get("predicted_answer", ""))
    gold = row.get("gold_answer")
    gold_s = str(gold) if gold is not None else ""
    is_corr = bool(row.get("correct", False))
    question = row.get("question", "")
    trace = row.get("full_trace") or (
        (row.get("thinking_trace") or "") + "\n---\n" + (row.get("answer_text") or "")
    )
    return f"""question_id: {qid}

question:
{question}

final_answer (echo exactly in JSON): {final_ans}
gold_answer (echo exactly in JSON): {gold_s}
is_correct (echo exactly in JSON): {json.dumps(is_corr)}

reasoning_trace_to_segment:
{trace}
"""


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start: end + 1])


def validate_graph(obj: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if "steps" not in obj or not isinstance(obj["steps"], list):
        errs.append("missing or invalid 'steps'")
        return errs
    ids = set()
    for i, s in enumerate(obj["steps"]):
        if not isinstance(s, dict):
            errs.append(f"steps[{i}] is not an object")
            continue
        for k in ("id", "text", "depends_on", "op"):
            if k not in s:
                errs.append(f"steps[{i}] missing '{k}'")
        if "id" in s:
            ids.add(s["id"])
        if "depends_on" in s and not isinstance(s["depends_on"], list):
            errs.append(f"steps[{i}].depends_on must be a list")
    for i, s in enumerate(obj.get("steps", [])):
        if not isinstance(s, dict):
            continue
        for dep in s.get("depends_on", []):
            if dep not in ids:
                errs.append(f"steps[{i}] depends_on unknown id {dep!r}")
    return errs


def run_graph(client: anthropic.Anthropic, row: dict[str, Any], model: str) -> dict[str, Any]:
    with client.messages.stream(
        model=model,
        max_tokens=8192,
        thinking={"type": "adaptive"},
        system=GRAPH_SYSTEM,
        messages=[{"role": "user", "content": build_user_message(row)}],
    ) as stream:
        final = stream.get_final_message()

    text = next((b.text for b in final.content if b.type == "text"), "")
    obj = extract_json_object(text)

    # force canonical fields from the source trace
    obj["question_id"] = row.get("id", obj.get("question_id", ""))
    obj["final_answer"] = str(row.get("predicted_answer", ""))
    if row.get("gold_answer") is not None:
        obj["gold_answer"] = row["gold_answer"]
    obj["is_correct"] = bool(row.get("correct", False))
    obj["_meta"] = {
        "trace_model": row.get("model"),
        "graph_model": model,
        "source_file": "gsm_hard_traces.jsonl",
    }
    return obj


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", default="gsm_hard_data/gsm_hard_traces.jsonl")
    p.add_argument("--out", default="", help="output path (default: gsm_hard_graphs.jsonl in same dir)")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--limit", type=int, default=0, help="max rows (0 = all)")
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--skip-invalid", action="store_true")
    args = p.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.is_file():
        sys.exit(f"not found: {traces_path}")

    out_path = Path(args.out) if args.out else traces_path.parent / "gsm_hard_graphs.jsonl"
    checkpoint_path = out_path.with_name(out_path.stem + "_checkpoint.jsonl")

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    rows: list[dict[str, Any]] = []
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    done_ids: set[str] = set()
    results: list[dict[str, Any]] = []
    if checkpoint_path.is_file():
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = rec.get("question_id") or rec.get("id")
                if qid:
                    done_ids.add(qid)
                results.append(rec)
        print(f"resume: {len(done_ids)} graphs already done")

    n = min(len(rows), args.limit) if args.limit else len(rows)
    print(f"model: {args.model}")
    print(f"input: {traces_path}  ({n} rows)")
    print(f"output: {out_path}")

    errors: list[dict] = []
    for i, row in enumerate(rows[:n]):
        qid = row.get("id", f"row-{i}")
        if qid in done_ids:
            continue
        print(f"[{i + 1}/{n}] {qid}...", end=" ", flush=True)
        try:
            obj = run_graph(client, row, args.model)
            errs = validate_graph(obj)
            if errs:
                obj["_validation_warnings"] = errs
                if args.skip_invalid:
                    print(f"skip (validation): {errs[:2]}")
                    errors.append({"id": qid, "error": "validation", "details": errs})
                    continue
            results.append(obj)
            print(f"steps={len(obj.get('steps', []))} warnings={len(errs)}")
        except Exception as e:
            print(f"error: {e}")
            errors.append({"id": qid, "error": str(e)})

        if args.sleep:
            time.sleep(args.sleep)

        if len(results) % args.checkpoint_every == 0 and results:
            with open(checkpoint_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    if checkpoint_path.is_file():
        checkpoint_path.unlink(missing_ok=True)

    if errors:
        err_path = out_path.with_name(out_path.stem + "_errors.jsonl")
        with open(err_path, "w") as f:
            for e in errors:
                f.write(json.dumps(e) + "\n")
        print(f"errors logged: {err_path}")

    print(f"\nwrote {len(results)} graphs to {out_path}")


if __name__ == "__main__":
    main()
