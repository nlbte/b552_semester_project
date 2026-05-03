"""Extended analysis of GSM-Hard reasoning graph features.

Builds on motif_analysis.py with:
  1. Feature correlation matrix + redundancy check (depth vs n_steps)
  2. Multivariate logistic regression with statsmodels (proper p-values)
  3. Composite motif-score per trace + correctness prediction
  4. Out-degree breakdown by node op type (which ops fan out in failing traces)
  5. Error-magnitude split within incorrect traces (high vs low error)
  6. Qualitative case-study graph figures (2 correct, 2 incorrect)

Usage:
    python extended_analysis.py
    python extended_analysis.py --graphs gsm_hard_data/gsm_hard_graphs.jsonl --out-dir gsm_hard_data
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")


# constants

OP_COLORS = {
    "extract_fact": "#4e9af1",
    "arithmetic":   "#f4a261",
    "substitute":   "#a8b5a0",
    "conclude":     "#e76f51",
    "verify":       "#e5c07b",
    "other":        "#aaaaaa",
    "unknown":      "#cccccc",
}

MOTIF_FLAGS = {
    "has_unsupported_arith":   lambda r: r["unsupported_arithmetic"] > 0,
    "has_orphan":              lambda r: r["orphan_nodes"] > 0,
    "has_unsupported_conclude":lambda r: r["unsupported_conclude"] > 0,
    "low_fact_ratio":          lambda r: r["frac_facts"] < 0.25,
    "long_chain":              lambda r: r["depth"] > 5,
}

FEATURE_COLS = [
    "n_steps", "depth", "n_arithmetic", "n_facts", "n_conclude",
    "frac_arithmetic", "frac_facts", "unsupported_arithmetic",
    "unsupported_conclude", "conclude_in_degree", "max_out_degree",
    "mean_out_degree", "mean_in_degree", "orphan_nodes",
]


# helpers

def load_graphs(path: str) -> list[dict]:
    graphs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                graphs.append(json.loads(line))
    return graphs


def build_nx(steps: list[dict]) -> nx.DiGraph:
    g = nx.DiGraph()
    for s in steps:
        g.add_node(s["id"], op=s.get("op", "unknown"), text=s.get("text", ""))
    for s in steps:
        for dep in s.get("depends_on", []) or []:
            if dep in g.nodes and dep != s["id"]:
                g.add_edge(dep, s["id"])
    return g


def parse_number(s) -> float | None:
    if s is None:
        return None
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return None


def analyze_correlation(df: pd.DataFrame, out_dir: Path) -> None:
    print("\n1. feature correlation (checking redundancy)")
    cols = [c for c in FEATURE_COLS if df[c].std() > 0]
    corr = df[cols].corr(method="spearman")

    # highlight pairs with |r| > 0.7
    high = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) > 0.7:
                high.append((a, b, r))
    high.sort(key=lambda x: -abs(x[2]))

    print("highly correlated feature pairs (|spearman r| > 0.7):")
    for a, b, r in high:
        print(f"  {a:30s} ~ {b:30s}  r={r:.3f}")

    # depth vs n_steps specifically
    r, p = stats.spearmanr(df["depth"], df["n_steps"])
    print(f"\ndepth ~ n_steps:  r={r:.3f}  p={p:.4f}")
    print("  → these are largely the same signal" if abs(r) > 0.7 else
          "  → these carry independent information")

    # save heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    plt.colorbar(im, ax=ax, label="spearman r")
    ax.set_title("feature correlation matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_matrix.png", dpi=120)
    plt.close(fig)
    print(f"\nwrote correlation_matrix.png")


def logistic_regression(df: pd.DataFrame, out_dir: Path) -> None:
    print("\n2. multivariate logistic regression")

    # drop zero-variance and highly correlated features
    # keep depth (drop n_steps), keep mean_out_degree (drop mean_in_degree — identical)
    drop = {"n_steps", "mean_in_degree", "n_arithmetic", "n_facts", "n_conclude",
            "unsupported_arithmetic", "unsupported_conclude"}
    cols = [c for c in FEATURE_COLS if c not in drop and df[c].std() > 0]

    X_raw = df[cols].values.astype(float)
    # standardize for interpretable coefficients
    X_std = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-9)
    X = sm.add_constant(X_std)
    y = df["is_correct"].astype(int).values

    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=0, maxiter=200)

        rows = []
        for name, coef, pval in zip(["const"] + cols, result.params, result.pvalues):
            rows.append({"feature": name, "coef": coef, "p_value": pval,
                         "sig": "***" if pval < 0.01 else "**" if pval < 0.05 else
                                "*" if pval < 0.1 else ""})
        res_df = pd.DataFrame(rows).sort_values("p_value")
        print(res_df.to_string(index=False))
        print(f"\npseudo-R²: {result.prsquared:.3f}   AIC: {result.aic:.1f}")
        res_df.to_csv(out_dir / "logistic_regression.csv", index=False)
        print("wrote logistic_regression.csv")
    except Exception as e:
        print(f"logistic regression failed: {e}")


def motif_score_analysis(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    print("\n3. composite motif score")

    for flag, fn in MOTIF_FLAGS.items():
        df[flag] = df.apply(fn, axis=1).astype(int)
    df["motif_count"] = df[[f for f in MOTIF_FLAGS]].sum(axis=1)

    # motif count distribution by correctness
    print("motif count distribution (correct vs incorrect):")
    ct = df.groupby(["motif_count", "is_correct"]).size().unstack(fill_value=0)
    ct.columns = ["incorrect", "correct"]
    ct["total"] = ct.sum(axis=1)
    ct["pct_incorrect"] = (ct["incorrect"] / ct["total"] * 100).round(1)
    print(ct.to_string())

    # each flag's prevalence
    print("\nprevalence of each motif:")
    for flag in MOTIF_FLAGS:
        c = df[df["is_correct"]][flag].mean()
        w = df[~df["is_correct"]][flag].mean()
        print(f"  {flag:30s}  correct={c:.0%}  incorrect={w:.0%}")

    # point-biserial correlation between motif_count and is_correct
    r, p = stats.pointbiserialr(df["motif_count"], df["is_correct"].astype(int))
    print(f"\nmotif_count ~ is_correct:  r={r:.3f}  p={p:.4f}")

    # save
    df.to_csv(out_dir / "motif_scores.csv", index=False)
    print("wrote motif_scores.csv")

    # plot motif count distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, grp, color in [("correct", df[df["is_correct"]], "#4e9af1"),
                               ("incorrect", df[~df["is_correct"]], "#e76f51")]:
        counts = grp["motif_count"].value_counts().sort_index()
        ax.bar(counts.index + (0 if label == "correct" else 0.35),
               counts.values, width=0.35, label=label, color=color, alpha=0.8)
    ax.set_xlabel("motif count")
    ax.set_ylabel("number of traces")
    ax.set_title("composite motif score: correct vs incorrect")
    ax.legend()
    ax.set_xticks(range(df["motif_count"].max() + 1))
    fig.tight_layout()
    fig.savefig(out_dir / "motif_score_distribution.png", dpi=120)
    plt.close(fig)
    print("wrote motif_score_distribution.png")

    return df


def outdegree_by_op(graphs: list[dict], out_dir: Path) -> None:
    print("\n4. out-degree breakdown by node op type")

    rows = []
    for g_data in graphs:
        is_correct = bool(g_data.get("is_correct", False))
        steps = g_data.get("steps", [])
        g = build_nx(steps)
        for node, out_deg in g.out_degree():
            op = g.nodes[node].get("op", "unknown")
            rows.append({"question_id": g_data.get("question_id"),
                         "is_correct": is_correct,
                         "op": op,
                         "out_degree": out_deg})

    node_df = pd.DataFrame(rows)

    print("mean out-degree per op type (correct vs incorrect):")
    pivot = node_df.groupby(["op", "is_correct"])["out_degree"].mean().unstack()
    pivot.columns = ["incorrect", "correct"]
    pivot["diff"] = pivot["incorrect"] - pivot["correct"]
    print(pivot.round(3).sort_values("diff", ascending=False).to_string())

    print("\nhigh out-degree nodes (out_deg >= 3) — op type breakdown:")
    high = node_df[node_df["out_degree"] >= 3]
    print(high.groupby(["op", "is_correct"]).size().unstack(fill_value=0).to_string())

    node_df.to_csv(out_dir / "outdegree_by_op.csv", index=False)
    print("\nwrote outdegree_by_op.csv")

    # plot
    ops = node_df["op"].unique()
    fig, axes = plt.subplots(1, len(ops), figsize=(3 * len(ops), 4), sharey=False)
    if len(ops) == 1:
        axes = [axes]
    for ax, op in zip(axes, sorted(ops)):
        sub = node_df[node_df["op"] == op]
        c = sub[sub["is_correct"]]["out_degree"].values
        w = sub[~sub["is_correct"]]["out_degree"].values
        ax.boxplot([c, w], tick_labels=["correct", "incorrect"])
        ax.set_title(op, fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("out-degree distribution by op type", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "outdegree_by_op.png", dpi=120)
    plt.close(fig)
    print("wrote outdegree_by_op.png")



def error_split(graphs: list[dict], out_dir: Path) -> None:
    print("\n5. error magnitude split within incorrect traces")

    rows = []
    for g in graphs:
        gold = parse_number(g.get("gold_answer"))
        pred = parse_number(g.get("final_answer"))
        if gold is None or pred is None:
            continue
        if abs(gold) < 1e-9:
            rel_err = abs(pred - gold)
        else:
            rel_err = abs(pred - gold) / abs(gold)
        rows.append({
            "question_id": g.get("question_id"),
            "is_correct": bool(g.get("is_correct")),
            "gold": gold,
            "predicted": pred,
            "rel_error": rel_err,
            "log10_error": math.log10(rel_err + 1e-12),
            "steps": len(g.get("steps", [])),
        })

    err_df = pd.DataFrame(rows)
    incorrect = err_df[~err_df["is_correct"]].copy()

    if len(incorrect) == 0:
        print("no incorrect traces found")
        return

    # split at log10 error = 3 (off by >1000x = catastrophic)
    threshold = 3.0
    high_err = incorrect[incorrect["log10_error"] >= threshold]
    low_err  = incorrect[incorrect["log10_error"] < threshold]

    print(f"incorrect traces: {len(incorrect)} total")
    print(f"  high error (log10 >= {threshold}, off by >1000x): {len(high_err)}")
    print(f"  low  error (log10 <  {threshold}):                {len(low_err)}")

    if len(high_err):
        print(f"\nhigh-error traces (catastrophic):")
        print(high_err[["question_id", "gold", "predicted", "rel_error", "steps"]].to_string(index=False))

    if len(low_err):
        print(f"\nlow-error traces (near miss):")
        print(low_err[["question_id", "gold", "predicted", "rel_error", "steps"]].to_string(index=False))

    # compare step counts
    if len(high_err) > 0 and len(low_err) > 0:
        print(f"\nmean steps — high-error: {high_err['steps'].mean():.1f}  "
              f"low-error: {low_err['steps'].mean():.1f}")

    # histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(incorrect["log10_error"].replace(-np.inf, -12), bins=15,
            color="#e76f51", alpha=0.8, edgecolor="white")
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold (log10={threshold})")
    ax.set_xlabel("log10(relative error)")
    ax.set_ylabel("count")
    ax.set_title("error magnitude distribution (incorrect traces only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "error_distribution.png", dpi=120)
    plt.close(fig)
    print("\nwrote error_distribution.png")

    err_df.to_csv(out_dir / "error_split.csv", index=False)
    print("wrote error_split.csv")

    return err_df



def draw_graph(g_data: dict, ax, title: str) -> None:
    steps = g_data.get("steps", [])
    g = build_nx(steps)
    if len(g) == 0:
        ax.text(0.5, 0.5, "empty graph", ha="center", va="center")
        ax.set_title(title, fontsize=9)
        return

    try:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    except Exception:
        try:
            pos = nx.planar_layout(g)
        except Exception:
            pos = nx.spring_layout(g, seed=42)

    node_colors = [OP_COLORS.get(g.nodes[n].get("op", "unknown"), "#cccccc") for n in g.nodes]
    labels = {n: f"{n}\n{g.nodes[n].get('op','?')[:4]}" for n in g.nodes}

    nx.draw(g, pos, ax=ax, labels=labels, node_color=node_colors,
            node_size=800, font_size=6, arrows=True,
            arrowstyle="-|>", arrowsize=15, edge_color="#555555",
            width=1.2, with_labels=True)
    ax.set_title(title, fontsize=9)


def case_studies(graphs: list[dict], df: pd.DataFrame, out_dir: Path) -> None:
    print("\n6. case study graph figures")

    df_idx = df.set_index("question_id")
    g_by_id = {g.get("question_id"): g for g in graphs}

    # pick cases: correct with lowest motif_count+depth (cleanest),
    #             correct with highest depth (long but right),
    #             incorrect with highest motif_count,
    #             incorrect with highest depth
    correct_df = df[df["is_correct"]].copy()
    incorrect_df = df[~df["is_correct"]].copy()

    cases = []
    if len(correct_df):
        clean = correct_df.loc[correct_df["motif_count"].idxmin()]
        cases.append((clean["question_id"], f"correct — cleanest (motif={int(clean['motif_count'])}, depth={int(clean['depth'])})"))
        if len(correct_df) > 1:
            deep = correct_df.loc[correct_df["depth"].idxmax()]
            cases.append((deep["question_id"], f"correct — deepest (depth={int(deep['depth'])}, motif={int(deep['motif_count'])})"))
    if len(incorrect_df):
        worst = incorrect_df.loc[incorrect_df["motif_count"].idxmax()]
        cases.append((worst["question_id"], f"incorrect — most motifs ({int(worst['motif_count'])}, depth={int(worst['depth'])})"))
        if len(incorrect_df) > 1:
            orphan = incorrect_df[incorrect_df["orphan_nodes"] > 0]
            if len(orphan):
                pick = orphan.loc[orphan["motif_count"].idxmax()]
            else:
                pick = incorrect_df.loc[incorrect_df["depth"].idxmax()]
            if pick["question_id"] != worst["question_id"]:
                cases.append((pick["question_id"], f"incorrect — orphan+motifs (orphans={int(pick['orphan_nodes'])}, depth={int(pick['depth'])})"))

    print(f"generating {len(cases)} case study figures:")
    for qid, label in cases:
        print(f"  {qid}: {label}")

    ncols = 2
    nrows = math.ceil(len(cases) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 6))
    axes = np.array(axes).flatten()

    for i, (qid, label) in enumerate(cases):
        g_data = g_by_id.get(qid)
        if g_data:
            draw_graph(g_data, axes[i], label)
        else:
            axes[i].text(0.5, 0.5, f"graph not found: {qid}", ha="center")

    for j in range(len(cases), len(axes)):
        axes[j].axis("off")

    # legend
    legend_patches = [mpatches.Patch(color=c, label=op) for op, c in OP_COLORS.items()
                      if op not in ("unknown",)]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(legend_patches),
               fontsize=8, title="node op type")
    fig.suptitle("case study graphs", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_dir / "case_studies.png", dpi=120)
    plt.close(fig)
    print("wrote case_studies.png")



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs",   default="gsm_hard_data/gsm_hard_graphs.jsonl")
    ap.add_argument("--features", default="gsm_hard_data/graph_features.csv",
                    help="output from motif_analysis.py (auto-generated if missing)")
    ap.add_argument("--out-dir",  default="gsm_hard_data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graphs = load_graphs(args.graphs)
    print(f"loaded {len(graphs)} graphs from {args.graphs}")

    # load or rebuild features df
    features_path = Path(args.features)
    if features_path.is_file():
        df = pd.read_csv(features_path)
        print(f"loaded features from {features_path}")
    else:
        print("features csv not found — run motif_analysis.py first")
        return

    # add motif flags (needed for case studies)
    for flag, fn in MOTIF_FLAGS.items():
        if flag not in df.columns:
            df[flag] = df.apply(fn, axis=1).astype(int)
    if "motif_count" not in df.columns:
        df["motif_count"] = df[[f for f in MOTIF_FLAGS]].sum(axis=1)

    analyze_correlation(df, out_dir)
    logistic_regression(df, out_dir)
    df = motif_score_analysis(df, out_dir)
    outdegree_by_op(graphs, out_dir)
    error_split(graphs, out_dir)
    case_studies(graphs, df, out_dir)

    print("\n✓ all outputs written to", out_dir)


if __name__ == "__main__":
    main()
