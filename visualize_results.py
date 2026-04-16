"""

Loads one or more traces.jsonl files, groups by model, and renders:
    1. Overall accuracy per model
    2. Output-token distribution
    3. Output tokens: correct vs incorrect
    4. Relative error distribution for incorrect answers
    5. A few example wrong answers printed to the console

Usage:
    python visualize_results.py                              # auto-discover *traces*.jsonl in gsm_hard_data
    python visualize_results.py --traces path/to/file.jsonl  # point at specific files
    python visualize_results.py --output-dir gsm_hard_data --out summary.png
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_traces(paths: list[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_source_file"] = Path(p).name
                rows.append(row)
    if not rows:
        raise SystemExit(f"No traces found in: {paths}")
    df = pd.DataFrame(rows)

    for col in ("output_tokens", "prompt_tokens", "gold_answer"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def _pred_num(s):
        if s is None or (isinstance(s, float) and math.isnan(s)):
            return None
        try:
            return float(str(s).replace(",", "").replace("$", "").strip())
        except (ValueError, AttributeError):
            return None

    df["predicted_num"] = df.get("predicted_answer", pd.Series([None] * len(df))).apply(_pred_num)

    def _rel_err(row):
        pred = row["predicted_num"]
        gold = row["gold_answer"]
        if pred is None or gold is None or pd.isna(gold):
            return None
        denom = max(abs(gold), 1e-9)
        return abs(pred - gold) / denom

    df["rel_error"] = df.apply(_rel_err, axis=1)
    if "model" not in df.columns:
        df["model"] = "unknown"
    return df


def render(df: pd.DataFrame, out_path: Path) -> None:
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    acc_by_model = df.groupby("model")["correct"].agg(["sum", "count", "mean"])
    acc_by_model.columns = ["correct", "total", "accuracy"]
    acc_by_model = acc_by_model.sort_values("accuracy")
    bars = ax.barh(acc_by_model.index, acc_by_model["accuracy"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.set_title("GSM-Hard Accuracy by Model")
    for bar, (_, row) in zip(bars, acc_by_model.iterrows()):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{row['correct']:.0f}/{row['total']:.0f} ({row['accuracy']*100:.1f}%)",
            va="center",
            fontsize=9,
        )

    ax = axes[0, 1]
    for m in models:
        sub = df[df["model"] == m]["output_tokens"].dropna()
        if len(sub):
            ax.hist(sub, bins=30, alpha=0.6, label=m, edgecolor="black")
    ax.set_xlabel("Output tokens")
    ax.set_ylabel("Count")
    ax.set_title("Output Token Distribution")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    data, labels = [], []
    for m in models:
        sub = df[df["model"] == m]
        correct_tok = sub[sub["correct"]]["output_tokens"].dropna()
        incorrect_tok = sub[~sub["correct"]]["output_tokens"].dropna()
        if len(correct_tok):
            data.append(correct_tok.values)
            labels.append(f"{m}\ncorrect")
        if len(incorrect_tok):
            data.append(incorrect_tok.values)
            labels.append(f"{m}\nincorrect")
    if data:
        try:
            ax.boxplot(data, tick_labels=labels)
        except TypeError:
            ax.boxplot(data, labels=labels)
    ax.set_ylabel("Output tokens")
    ax.set_title("Tokens: Correct vs Incorrect")
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[1, 1]
    wrong = df[~df["correct"]].copy()
    wrong = wrong[wrong["rel_error"].notna() & (wrong["rel_error"] > 0)]
    if len(wrong):
        wrong["log_rel_err"] = wrong["rel_error"].apply(lambda x: math.log10(max(x, 1e-12)))
        for m in sorted(wrong["model"].unique()):
            sub = wrong[wrong["model"] == m]["log_rel_err"]
            if len(sub):
                ax.hist(sub, bins=20, alpha=0.6, label=m, edgecolor="black")
        ax.set_xlabel("log10(relative error)")
        ax.set_ylabel("Count")
        ax.set_title(f"Relative Error (wrong answers, n={len(wrong)})")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No incorrect answers with numeric predictions", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Relative Error")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


def print_summary(df: pd.DataFrame, n_examples: int) -> None:
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    summary = df.groupby("model").agg(
        total=("correct", "count"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
        out_tok_mean=("output_tokens", "mean"),
        out_tok_median=("output_tokens", "median"),
    )
    summary["accuracy"] = summary["accuracy"].map(lambda x: f"{x*100:.1f}%")
    summary["out_tok_mean"] = summary["out_tok_mean"].map(lambda x: f"{x:.0f}")
    summary["out_tok_median"] = summary["out_tok_median"].map(lambda x: f"{x:.0f}")
    print(summary.to_string())

    if n_examples > 0:
        wrong = df[~df["correct"]]
        if len(wrong):
            print()
            print(f"Example wrong answers (showing {min(n_examples, len(wrong))}):")
            print("-" * 70)
            for _, row in wrong.head(n_examples).iterrows():
                q = row.get("question", "")[:180]
                print(f"[{row.get('id', '?')}] ({row.get('model', '?')})")
                print(f"  Q: {q}{'...' if len(row.get('question', '')) > 180 else ''}")
                print(f"  gold: {row.get('gold_answer')} | pred: {row.get('predicted_answer')} | rel_err: {row.get('rel_error')}")
                print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", nargs="*", help="specific jsonl file(s) to load")
    p.add_argument("--output-dir", default="gsm_hard_data", help="where to look for *traces*.jsonl (default: gsm_hard_data)")
    p.add_argument("--out", default="gsm_hard_data/summary.png", help="figure output path")
    p.add_argument("--examples", type=int, default=3, help="number of wrong-answer examples to print")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.traces:
        paths = args.traces
    else:
        paths = [
            p
            for p in glob.glob(str(Path(args.output_dir) / "*traces*.jsonl"))
            if "checkpoint" not in Path(p).name
        ]
    if not paths:
        raise SystemExit(f"No traces.jsonl files found in {args.output_dir}")

    print(f"Loading traces from {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")

    df = load_traces(paths)
    print(f"Loaded {len(df)} rows across {df['model'].nunique()} model(s)")

    print_summary(df, args.examples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render(df, out_path)


if __name__ == "__main__":
    main()
