import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


ARITHMETIC_OPS = {"arithmetic", "compute", "calculate"}
FACT_OPS = {"extract_fact", "given", "recall"}
CONCLUDE_OPS = {"conclude", "conclusion", "final_answer"}


# build a directed graph from a list of steps with depends_on edges
def build_graph(steps):
    g = nx.DiGraph()
    for step in steps:
        g.add_node(
            step["id"],
            text=step.get("text", ""),
            op=step.get("op", "unknown"),
        )
    for step in steps:
        for dep in step.get("depends_on", []) or []:
            if dep in g.nodes and dep != step["id"]:
                g.add_edge(dep, step["id"])
    return g


# returns the longest dependency chain length (depth) in the dag
def longest_path_length(g):
    if len(g) == 0:
        return 0
    if not nx.is_directed_acyclic_graph(g):
        return -1
    try:
        return nx.dag_longest_path_length(g)
    except Exception:
        return -1


# compute all 14 structural features for a single graph trace
def extract_features(trace):
    steps = trace.get("steps", [])
    g = build_graph(steps)

    ops = [s.get("op", "unknown") for s in steps]
    op_counts = Counter(ops)

    n_steps = len(steps)
    n_arithmetic = sum(op_counts.get(op, 0) for op in ARITHMETIC_OPS)
    n_facts = sum(op_counts.get(op, 0) for op in FACT_OPS)
    n_conclude = sum(op_counts.get(op, 0) for op in CONCLUDE_OPS)

    unsupported_arith = 0
    for s in steps:
        if s.get("op") in ARITHMETIC_OPS and not s.get("depends_on"):
            unsupported_arith += 1

    conclude_nodes = [s["id"] for s in steps if s.get("op") in CONCLUDE_OPS]
    conclude_in_deg = 0
    if conclude_nodes:
        conclude_in_deg = max(g.in_degree(n) for n in conclude_nodes)

    unsupported_conclude = 0
    for cn in conclude_nodes:
        if g.in_degree(cn) == 0:
            unsupported_conclude += 1

    out_degrees = [d for _, d in g.out_degree()]
    in_degrees = [d for _, d in g.in_degree()]
    max_out = max(out_degrees) if out_degrees else 0
    mean_out = float(np.mean(out_degrees)) if out_degrees else 0.0
    mean_in = float(np.mean(in_degrees)) if in_degrees else 0.0

    orphans = 0
    for node in g.nodes():
        op = g.nodes[node].get("op")
        if op in CONCLUDE_OPS:
            continue
        if g.out_degree(node) == 0:
            orphans += 1

    depth = longest_path_length(g)

    frac_arith = n_arithmetic / n_steps if n_steps else 0.0
    frac_facts = n_facts / n_steps if n_steps else 0.0

    return {
        "question_id": trace.get("question_id"),
        "is_correct": bool(trace.get("is_correct")),
        "n_steps": n_steps,
        "depth": depth,
        "n_arithmetic": n_arithmetic,
        "n_facts": n_facts,
        "n_conclude": n_conclude,
        "frac_arithmetic": frac_arith,
        "frac_facts": frac_facts,
        "unsupported_arithmetic": unsupported_arith,
        "unsupported_conclude": unsupported_conclude,
        "conclude_in_degree": conclude_in_deg,
        "max_out_degree": max_out,
        "mean_out_degree": mean_out,
        "mean_in_degree": mean_in,
        "orphan_nodes": orphans,
    }


# load traces from either a jsonl file or a json array
def load_traces(path):
    traces = []
    p = Path(path)
    text = p.read_text()
    stripped = text.strip()
    if stripped.startswith("["):
        traces = json.loads(stripped)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


# run mann-whitney u test on each feature comparing correct vs incorrect traces
def compare_groups(df, feature_cols):
    rows = []
    correct = df[df["is_correct"]]
    wrong = df[~df["is_correct"]]
    for col in feature_cols:
        c = correct[col].dropna().values
        w = wrong[col].dropna().values
        if len(c) < 2 or len(w) < 2:
            rows.append({
                "feature": col,
                "correct_mean": float(np.mean(c)) if len(c) else float("nan"),
                "wrong_mean": float(np.mean(w)) if len(w) else float("nan"),
                "correct_median": float(np.median(c)) if len(c) else float("nan"),
                "wrong_median": float(np.median(w)) if len(w) else float("nan"),
                "u_stat": float("nan"),
                "p_value": float("nan"),
                "n_correct": len(c),
                "n_wrong": len(w),
            })
            continue
        try:
            u, p = stats.mannwhitneyu(c, w, alternative="two-sided")
        except ValueError:
            u, p = float("nan"), float("nan")
        rows.append({
            "feature": col,
            "correct_mean": float(np.mean(c)),
            "wrong_mean": float(np.mean(w)),
            "correct_median": float(np.median(c)),
            "wrong_median": float(np.median(w)),
            "u_stat": float(u) if not math.isnan(u) else float("nan"),
            "p_value": float(p) if not math.isnan(p) else float("nan"),
            "n_correct": len(c),
            "n_wrong": len(w),
        })
    return pd.DataFrame(rows).sort_values("p_value", na_position="last")


# save a grid of boxplots showing each feature split by correctness
def plot_comparison(df, feature_cols, out_path):
    cols_to_plot = [c for c in feature_cols if df[c].nunique() > 1]
    n = len(cols_to_plot)
    if n == 0:
        return
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes = np.atleast_2d(axes).flatten()

    correct_vals = df[df["is_correct"]]
    wrong_vals = df[~df["is_correct"]]

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        data = [correct_vals[col].values, wrong_vals[col].values]
        ax.boxplot(data, tick_labels=[f"correct\nn={len(correct_vals)}", f"wrong\nn={len(wrong_vals)}"])
        ax.set_title(col, fontsize=10)
        ax.grid(True, alpha=0.3)

    for j in range(len(cols_to_plot), len(axes)):
        axes[j].axis("off")

    fig.suptitle("graph features: correct vs incorrect", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# load graphs, extract features, run stats, save csvs and plot
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--graphs",
        default="gsm_hard_data/gsm_hard_graphs.jsonl",
        help="path to graphs jsonl or json array",
    )
    ap.add_argument(
        "--out-dir",
        default="gsm_hard_data",
        help="where to write outputs",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = load_traces(args.graphs)
    print(f"loaded {len(traces)} traces from {args.graphs}")

    feature_rows = [extract_features(t) for t in traces]
    df = pd.DataFrame(feature_rows)

    features_csv = out_dir / "graph_features.csv"
    df.to_csv(features_csv, index=False)
    print(f"wrote per-trace features to {features_csv}")

    feature_cols = [c for c in df.columns if c not in {"question_id", "is_correct"}]
    stats_df = compare_groups(df, feature_cols)
    stats_csv = out_dir / "feature_comparison.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"wrote statistical comparison to {stats_csv}")

    plot_path = out_dir / "feature_comparison.png"
    plot_comparison(df, feature_cols, plot_path)
    print(f"wrote comparison plot to {plot_path}")

    print("\ntop features by p-value:")
    print(stats_df.head(10).to_string(index=False))

    n_correct = int(df["is_correct"].sum())
    n_wrong = int((~df["is_correct"]).sum())
    print(f"\noverall: {n_correct} correct, {n_wrong} incorrect")


if __name__ == "__main__":
    main()