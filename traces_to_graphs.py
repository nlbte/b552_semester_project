"""Turn gsm_hard_traces.jsonl rows into structured reasoning graphs (JSON).

Reads each trace, calls Ollama Cloud to segment the reasoning into steps with
depends_on edges and op labels. Writes one JSON object per line.

Usage:
    cp .env.example .env    # set OLLAMA_API_KEY (or export in shell)

    python traces_to_graphs.py \\
        --traces gsm_hard_data/gsm_hard_traces.jsonl \\
        --out gsm_hard_data/gsm_hard_graphs.jsonl

    # Smaller / faster model for structuring (optional):
    python traces_to_graphs.py --model gpt-oss:20b --limit 5
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


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


def load_ollama_api_key() -> str:
    env_path = Path(__file__).resolve().parent / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                    v = v[1:-1]
                if k and k not in os.environ:
                    os.environ[k] = v
    key = os.environ.get("OLLAMA_API_KEY")
    if not key:
        key = getpass.getpass(
            "Enter your Ollama API key (https://ollama.com/settings/keys): "
        )
        os.environ["OLLAMA_API_KEY"] = key
    if not key:
        print("No API key provided.", file=sys.stderr)
        sys.exit(1)
    return key


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


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
                errs.append(f"steps[{i}] depends_on references unknown id {dep!r}")
    return errs


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


def run_graph(
    client,
    row: dict[str, Any],
    model: str,
    num_ctx: int,
    num_predict: int,
) -> dict[str, Any]:
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": GRAPH_SYSTEM},
            {"role": "user", "content": build_user_message(row)},
        ],
        options={
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    )

    def _get(obj: Any, key: str, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    msg = _get(response, "message", {}) or {}
    content = _get(msg, "content", "") or ""
    obj = extract_json_object(content)

    # Force canonical fields from trace (avoid model drift)
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
    p.add_argument("--traces", default="gsm_hard_data/gsm_hard_traces.jsonl", help="input traces JSONL")
    p.add_argument("--out", default="", help="output path (default: same dir as traces, gsm_hard_graphs.jsonl)")
    p.add_argument("--model", default="gpt-oss:120b", help="Ollama Cloud model for graph extraction")
    p.add_argument("--num-ctx", type=int, default=16384)
    p.add_argument("--num-predict", type=int, default=8192)
    p.add_argument("--limit", type=int, default=0, help="max rows (0 = all)")
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--skip-invalid", action="store_true", help="skip rows where JSON validation fails after fix")
    args = p.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.is_file():
        raise SystemExit(f"Not found: {traces_path}")

    out_path = Path(args.out) if args.out else traces_path.parent / "gsm_hard_graphs.jsonl"
    checkpoint_path = out_path.with_name(out_path.stem + "_checkpoint.jsonl")

    load_ollama_api_key()
    from ollama import Client

    client = Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )
    model = args.model if args.model.endswith("-cloud") else args.model + "-cloud"
    print(f"Graph model: {model}")
    print(f"Input: {traces_path}")
    print(f"Output: {out_path}")

    rows: list[dict[str, Any]] = []
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    done_ids: set[str] = set()
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
        print(f"Resume: {len(done_ids)} graphs already in checkpoint")

    n = len(rows) if not args.limit else min(args.limit, len(rows))
    results: list[dict[str, Any]] = []
    if checkpoint_path.is_file():
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))

    errors: list[dict[str, Any]] = []

    for i, row in enumerate(rows[:n]):
        qid = row.get("id", f"row-{i}")
        if qid in done_ids:
            continue
        print(f"[{i + 1}/{n}] {qid}...", end=" ", flush=True)
        try:
            obj = run_graph(client, row, model, args.num_ctx, args.num_predict)
            errs = validate_graph(obj)
            if errs:
                obj["_validation_warnings"] = errs
                if args.skip_invalid:
                    print(f"SKIP (validation): {errs[:2]}")
                    errors.append({"id": qid, "error": "validation", "details": errs})
                    continue
            results.append(obj)
            print(f"steps={len(obj.get('steps', []))} warnings={len(errs) if errs else 0}")
        except Exception as e:
            print(f"ERROR: {e}")
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
        print(f"Errors logged: {err_path}")

    print(f"Wrote {len(results)} graphs to {out_path}")


if __name__ == "__main__":
    main()
