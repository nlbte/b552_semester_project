"""Run GSM-Hard math word problems through Ollama Cloud.

Dataset: reasoning-machines/gsm-hard (1319 math word problems with large numbers).
Each row has: input (question), code (reference solution), target (numeric answer).

Usage:
    cp .env.example .env    # then set OLLAMA_API_KEY (or export it in the shell)

    python gsm_hard_ollama.py                               # gpt-oss:120b, 50 samples
    python gsm_hard_ollama.py --sample-size 200 --output-dir runs/hf_draw_01

    # Same fixed problems every time (no HF re-sample): point at your gsm_hard_sample.jsonl manifest.
    # Use --from-sample-jsonl twice if you need two files concatenated in order.
    # By default, outputs go under a model-named subfolder of --output-dir (swap models cleanly).
    # Use --flat-output to write gsm_hard_*.jsonl directly in --output-dir (legacy layout).
    python gsm_hard_ollama.py --from-sample-jsonl gsm_hard_data_batch2/gsm_hard_sample.jsonl \\
        --output-dir runs/trace_t0_v1 --temperature 0 --run-label t0_v1

    python gsm_hard_ollama.py --sample-size 150 --exclude-from gsm_hard_data/gsm_hard_traces.jsonl \\
        --output-dir gsm_hard_data_batch2 --id-prefix b2 --seed 43

Loads OLLAMA_API_KEY from the environment, then from .env in this directory,
then prompts via getpass if still unset.
"""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are solving a math word problem. Work through it step by step, showing your reasoning. Be careful with arithmetic — the numbers may be large.

After your reasoning, provide your final numeric answer on a new line starting with 'FINAL ANSWER: '. The answer must be a single number (no units, no commas)."""


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


def build_prompt(question: str) -> str:
    return question.strip()


def resolve_ollama_cloud_model(raw: str) -> str:
    """Canonical model string sent to Ollama (suffix -cloud unless already present)."""
    s = raw.strip()
    return s if s.endswith("-cloud") else s + "-cloud"


def model_slug_for_path(model: str) -> str:
    """Filesystem-safe slug for grouping outputs by model (e.g. gpt-oss:120b-cloud → gpt-oss-120b-cloud)."""
    s = model.strip().lower().replace(":", "-").replace("/", "-")
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "model"


def load_excluded_question_keys(paths: list[Path]) -> set[str]:
    """Normalize question text from prior jsonl rows (input / question / user_prompt)."""
    keys: set[str] = set()
    for path in paths:
        if not path.is_file():
            print(f"  warning: exclude-from file not found: {path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("input") or obj.get("question") or obj.get("user_prompt")
                if text is not None:
                    keys.add(build_prompt(str(text)))
    return keys


def extract_final_answer(trace: str) -> str | None:
    """Pull the last occurrence of a numeric final answer out of the response."""
    patterns = [
        r"FINAL ANSWER:\s*([-+]?\$?\s*[\d,]+\.?\d*(?:[eE][-+]?\d+)?)",
        r"\\boxed\{\s*([-+]?\$?\s*[\d,]+\.?\d*(?:[eE][-+]?\d+)?)\s*\}",
        r"final answer is[:\s]*\$?\s*([-+]?[\d,]+\.?\d*(?:[eE][-+]?\d+)?)",
        r"answer[:\s]+\$?\s*([-+]?[\d,]+\.?\d*(?:[eE][-+]?\d+)?)",
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, trace, flags=re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip()

    # fallback: last number-looking token in the last non-empty line
    for line in reversed(trace.strip().split("\n")):
        nums = re.findall(r"[-+]?[\d,]+\.?\d*(?:[eE][-+]?\d+)?", line)
        if nums:
            return nums[-1]
    return None


def parse_number(s: str | None) -> float | None:
    if s is None:
        return None
    cleaned = s.replace(",", "").replace("$", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def check_correctness(predicted: str | None, gold: float, rel_tol: float = 1e-4, abs_tol: float = 1e-4) -> bool:
    pred = parse_number(predicted)
    if pred is None:
        return False
    return math.isclose(pred, float(gold), rel_tol=rel_tol, abs_tol=abs_tol)


def load_rows_from_sample_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    """Load {id, input, target} rows from one or more gsm_hard_sample-style JSONL files (order preserved)."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        p = path.expanduser()
        if not p.is_file():
            print(f"error: --from-sample-jsonl file not found: {p}", file=sys.stderr)
            sys.exit(1)
        with open(p, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"{p}:{line_no}: invalid JSON: {e}", file=sys.stderr)
                    sys.exit(1)
                if "input" not in rec and "question" not in rec:
                    print(f"{p}:{line_no}: missing input/question", file=sys.stderr)
                    sys.exit(1)
                if "target" not in rec:
                    print(f"{p}:{line_no}: missing target", file=sys.stderr)
                    sys.exit(1)
                text = rec.get("input") or rec.get("question") or ""
                oid = rec.get("id")
                rows.append({
                    "id": oid if oid is not None else f"manifest-line-{len(rows)}",
                    "input": text,
                    "target": float(rec["target"]),
                })
                if oid is None:
                    print(f"warning: {p}:{line_no} had no id; using {rows[-1]['id']!r}", file=sys.stderr)
    return rows


def collect_trace(
    client,
    prompt_data: dict,
    model: str,
    num_ctx: int = 8192,
    num_predict: int = 4096,
    temperature: float = 0.0,
) -> dict:
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt_data["system_prompt"]},
            {"role": "user", "content": prompt_data["user_prompt"]},
        ],
        options={
            "temperature": temperature,
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
    thinking = _get(msg, "thinking", "") or ""

    if not thinking:
        m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if m:
            thinking = m.group(1).strip()
            content = content[m.end():].strip()

    prompt_tokens = _get(response, "prompt_eval_count", 0) or 0
    output_tokens = _get(response, "eval_count", 0) or 0

    return {
        "id": prompt_data["id"],
        "model": model,
        "question": prompt_data["user_prompt"],
        "thinking_trace": thinking.strip(),
        "answer_text": content.strip(),
        "full_trace": (thinking + "\n---\n" + content).strip(),
        "gold_answer": prompt_data["gold_answer"],
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }


def run(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from ollama import Client

    load_ollama_api_key()

    base_output_dir = Path(args.output_dir).expanduser()
    model = resolve_ollama_cloud_model(args.model)
    slug = model_slug_for_path(model)

    if args.flat_output:
        output_dir = base_output_dir
    else:
        output_dir = base_output_dir / slug

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.flat_output:
        print(f"Output directory (flat): {output_dir}")
    else:
        print(f"Output directory (model '{model}'): {output_dir}")
        print(f"  (--flat-output skips the '{slug}/' subfolder and writes directly under {base_output_dir})")

    manifest_paths = [Path(p).expanduser() for p in args.from_sample_jsonl]

    if manifest_paths:
        print("Manifest mode: loading fixed problem list from JSONL (no HF re-sample).")
        if args.exclude_from:
            print("  note: --exclude-from is ignored in manifest mode.", file=sys.stderr)
        if args.id_prefix:
            print("  note: --id-prefix is ignored in manifest mode (ids come from the files).", file=sys.stderr)
        row_dicts = load_rows_from_sample_jsonl(manifest_paths)
        print(f"  loaded {len(row_dicts)} problems from {len(manifest_paths)} file(s)")
    else:
        print("Loading reasoning-machines/gsm-hard (split=train)...")
        dataset = load_dataset("reasoning-machines/gsm-hard", split="train")
        df = dataset.to_pandas()
        print(f"  total questions: {len(df)}")

        exclude_paths = [Path(p).expanduser() for p in args.exclude_from]
        if exclude_paths:
            excluded = load_excluded_question_keys(exclude_paths)
            df["_qkey"] = df["input"].map(lambda x: build_prompt(str(x)))
            before = len(df)
            df = df.loc[~df["_qkey"].isin(excluded)].drop(columns=["_qkey"]).reset_index(drop=True)
            print(f"  excluded {before - len(df)} already-seen questions ({len(excluded)} unique keys in exclude files)")
            print(f"  pool size after exclusion: {len(df)}")
            if len(df) == 0:
                print("No rows left after exclusion — add fewer --exclude-from files or check paths.", file=sys.stderr)
                sys.exit(1)

        n_take = min(args.sample_size, len(df))
        if n_take < args.sample_size:
            print(
                f"  warning: only {n_take} rows available (asked for {args.sample_size}); sampling all remaining.",
                file=sys.stderr,
            )
        sample = df.sample(n=n_take, random_state=args.seed).reset_index(drop=True)
        pfx = args.id_prefix.strip()
        if pfx:
            sample["id"] = [f"gsm-hard-{pfx}-{i}" for i in range(len(sample))]
        else:
            sample["id"] = [f"gsm-hard-{i}" for i in range(len(sample))]
        print(f"  sample size: {len(sample)}")
        export_cols = ["id", "input", "target"]
        sample[export_cols].to_json(output_dir / "gsm_hard_sample.jsonl", orient="records", lines=True)

        row_dicts = sample.to_dict(orient="records")

    if manifest_paths:
        with open(output_dir / "gsm_hard_sample.jsonl", "w", encoding="utf-8") as f:
            for r in row_dicts:
                f.write(json.dumps({"id": r["id"], "input": r["input"], "target": r["target"]}) + "\n")

    prompts = []
    for row in row_dicts:
        prompts.append({
            "id": row["id"],
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": build_prompt(str(row["input"])),
            "gold_answer": float(row["target"]),
        })
    with open(output_dir / "gsm_hard_prompts.jsonl", "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    print(f"Built {len(prompts)} prompts")

    mode = "manifest" if manifest_paths else "hf_sample"
    run_meta = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "manifest_paths": [str(p) for p in manifest_paths] if manifest_paths else [],
        "output_dir_base": str(base_output_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "flat_output": args.flat_output,
        "model": model,
        "model_slug": slug,
        "run_label": (args.run_label or "").strip(),
        "temperature": args.temperature,
        "seed": args.seed if mode == "hf_sample" else None,
        "sample_size_requested": args.sample_size if mode == "hf_sample" else None,
        "n_prompts": len(prompts),
    }
    meta_path = output_dir / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Wrote {meta_path}")
    print(f"Decoder temperature={args.temperature}")

    client = Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )

    print(f"Using Ollama Cloud model: {model}")

    all_results: list[dict] = []
    all_errors: list[dict] = []
    traces_path = output_dir / "gsm_hard_traces.jsonl"
    checkpoint_path = output_dir / "gsm_hard_traces_checkpoint.jsonl"

    n = min(len(prompts), args.limit) if args.limit else len(prompts)
    for i, prompt_data in enumerate(prompts[:n]):
        print(f"[{i + 1}/{n}] {prompt_data['id']}...", end=" ", flush=True)
        try:
            result = collect_trace(
                client,
                prompt_data,
                model=model,
                num_ctx=args.num_ctx,
                num_predict=args.num_predict,
                temperature=args.temperature,
            )
            extracted = extract_final_answer(result["answer_text"])
            correct = check_correctness(extracted, result["gold_answer"])
            result["predicted_answer"] = extracted
            result["correct"] = correct
            result["_run"] = {
                "model": model,
                "model_slug": slug,
                "temperature": args.temperature,
                "run_label": (args.run_label or "").strip(),
                "mode": mode,
                "manifest_paths": run_meta["manifest_paths"],
                "output_dir": str(output_dir.resolve()),
            }
            all_results.append(result)
            print(f"gold={result['gold_answer']:.2f} pred={extracted} correct={correct} out_tok={result['output_tokens']}")
        except Exception as e:
            print(f"ERROR: {e}")
            all_errors.append({"id": prompt_data["id"], "error": str(e)})

        if args.sleep:
            time.sleep(args.sleep)

        if (i + 1) % args.checkpoint_every == 0:
            with open(checkpoint_path, "w") as f:
                for r in all_results:
                    f.write(json.dumps(r) + "\n")
            correct_so_far = sum(1 for r in all_results if r["correct"])
            print(f"  >> checkpoint: {len(all_results)} traces, {correct_so_far} correct")

    with open(traces_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    if all_errors:
        with open(output_dir / "gsm_hard_errors.jsonl", "w") as f:
            for e in all_errors:
                f.write(json.dumps(e) + "\n")

    print()
    print("=" * 60)
    if all_results:
        correct_count = sum(1 for r in all_results if r["correct"])
        print(
            f"Final: {len(all_results)} traces, {correct_count}/{len(all_results)} correct "
            f"({correct_count / len(all_results) * 100:.1f}%)"
        )
        import pandas as pd
        results_df = pd.DataFrame(all_results)
        print(
            f"Output tokens: mean={results_df['output_tokens'].mean():.0f} "
            f"median={results_df['output_tokens'].median():.0f}"
        )
    print(f"Errors: {len(all_errors)}")
    print(f"Traces saved to: {traces_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt-oss:120b", help="Ollama Cloud model (default: gpt-oss:120b). '-cloud' suffix added automatically.")
    p.add_argument("--sample-size", type=int, default=50, help="random sample size (default: 50)")
    p.add_argument("--limit", type=int, default=0, help="cap number of prompts actually sent (0 = all sampled)")
    p.add_argument("--output-dir", default="gsm_hard_data", help="directory for outputs (default: gsm_hard_data)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--num-predict", type=int, default=4096)
    p.add_argument("--sleep", type=float, default=0.5, help="seconds to sleep between requests")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument(
        "--exclude-from",
        action="append",
        default=[],
        metavar="PATH",
        help="jsonl with input, question, or user_prompt per line, those problems are removed before sampling (repeat flag for multiple files).",
    )
    p.add_argument(
        "--id-prefix",
        default="",
        help="HF mode only: run id prefix, e.g. b2 -> gsm-hard-b2-0 (manifest mode ignores this).",
    )
    p.add_argument(
        "--from-sample-jsonl",
        action="append",
        default=[],
        metavar="PATH",
        help="Use fixed problems from JSONL (repeat for multiple files, merged in order). Skips HF sampling.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Ollama decode temperature (default: 0). Use >0 for stochastic reruns on the same manifest.",
    )
    p.add_argument(
        "--run-label",
        default="",
        help="Tag stored in run_meta.json and each trace under _run (e.g. t0_baseline, t0p3_v2).",
    )
    p.add_argument(
        "--flat-output",
        action="store_true",
        help="Write gsm_hard_*.jsonl directly under --output-dir (no model subfolder). Default: nest under a folder named from the model.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())