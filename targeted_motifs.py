"""Compute three targeted reasoning motifs and run statistical tests.

Motifs:
  late_arithmetic     -- arithmetic node at graph depth >= 4 with no extract_fact ancestor
  verbose_ungrounded  -- n_steps >= 8 AND frac_facts < 0.3
  early_branching     -- node in first half of trace (by sequential position) with out_degree >= 3

Usage:
    python targeted_motifs.py \\
        --graphs gsm_hard_data/gsm_hard_graphs_all.jsonl \\
        --traces gsm_hard_data/gsm_hard_traces.jsonl \\
        --label gpt-oss \\
        --out-csv gsm_hard_data/targeted_motifs_gptoss.csv

    python targeted_motifs.py \\
        --graphs experiments/gemma4-31b-cloud/gsm_hard_graphs.jsonl \\
        --traces experiments/gemma4-31b-cloud/gsm_hard_traces.jsonl \\
        --label gemma4 \\
        --out-csv experiments/gemma4-31b-cloud/targeted_motifs_gemma4.csv \\
        --report experiments/gemma4-31b-cloud/targeted_motifs_gemma4.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


MOTIF_FLAGS = ["late_arithmetic", "verbose_ungrounded", "early_branching"]


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_nx(steps):
    g = nx.DiGraph()
    for s in steps:
        g.add_node(s["id"], op=s.get("op", "unknown"), text=s.get("text", ""))
    for s in steps:
        for dep in s.get("depends_on", []) or []:
            if dep in g.nodes and dep != s["id"]:
                g.add_edge(dep, s["id"])
    return g


def compute_node_depths(g):
    depth = {n: 0 for n in g.nodes()}
    try:
        for n in nx.topological_sort(g):
            for succ in g.successors(n):
                depth[succ] = max(depth[succ], depth[n] + 1)
    except nx.NetworkXUnfeasible:
        pass
    return depth


def graph_longest_path(g):
    if len(g) == 0:
        return 0
    if not nx.is_directed_acyclic_graph(g):
        return -1
    try:
        return nx.dag_longest_path_length(g)
    except Exception:
        return -1


def flag_late_arithmetic(g, steps):
    if len(steps) == 0:
        return False
    if not nx.is_directed_acyclic_graph(g):
        return False
    node_depths = compute_node_depths(g)
    for s in steps:
        if s.get("op") != "arithmetic":
            continue
        nid = s["id"]
        if node_depths.get(nid, 0) < 4:
            continue
        ancestors = nx.ancestors(g, nid)
        has_fact_ancestor = any(g.nodes[a].get("op") == "extract_fact" for a in ancestors)
        if not has_fact_ancestor:
            return True
    return False


def flag_verbose_ungrounded(steps):
    n = len(steps)
    if n == 0:
        return False
    n_facts = sum(1 for s in steps if s.get("op") == "extract_fact")
    frac_facts = n_facts / n
    return n >= 8 and frac_facts < 0.3


def flag_early_branching(g, steps):
    n = len(steps)
    if n == 0:
        return False
    half = max(1, n // 2)
    first_half_ids = {s["id"] for s in steps[:half]}
    for nid in first_half_ids:
        if nid in g.nodes and g.out_degree(nid) >= 3:
            return True
    return False


def extract_row(graph_rec):
    steps = graph_rec.get("steps", [])
    g = build_nx(steps)
    n = len(steps)
    n_facts = sum(1 for s in steps if s.get("op") == "extract_fact")
    frac_f = n_facts / n if n else 0.0
    return {
        "question_id": graph_rec.get("question_id") or graph_rec.get("id"),
        "is_correct": bool(graph_rec.get("is_correct", False)),
        "late_arithmetic": int(flag_late_arithmetic(g, steps)),
        "verbose_ungrounded": int(flag_verbose_ungrounded(steps)),
        "early_branching": int(flag_early_branching(g, steps)),
        "depth": graph_longest_path(g),
        "frac_facts": round(frac_f, 6),
        "n_steps": n,
    }


def _fmt_p(p):
    if math.isnan(p):
        return "—"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def fisher_test(df, flag):
    present = df[df[flag] == 1]
    absent  = df[df[flag] == 0]
    table = [
        [int((~present["is_correct"]).sum()), int(present["is_correct"].sum())],
        [int((~absent["is_correct"]).sum()),  int(absent["is_correct"].sum())],
    ]
    try:
        or_val, p = stats.fisher_exact(table, alternative="greater")
    except Exception:
        or_val, p = float("nan"), float("nan")
    return table, or_val, p


def univariate_logit(df, flag):
    if df[flag].std() == 0 or df[flag].sum() < 5:
        return None
    X = sm.add_constant(df[[flag]].astype(float))
    y = df["is_correct"].astype(int)
    try:
        result = sm.Logit(y, X).fit(disp=0, maxiter=200)
        coef = result.params[flag]
        pval = result.pvalues[flag]
        ci = result.conf_int()
        or_val = math.exp(coef)
        ci_lo = math.exp(ci.iloc[1, 0])
        ci_hi = math.exp(ci.iloc[1, 1])
        return {"coef": coef, "or": or_val, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_value": pval}
    except Exception as e:
        print(f"  logit failed for {flag}: {e}")
        return None


def combined_logit(df, features, label):
    active = [f for f in features if f in df.columns and df[f].std() > 0 and df[f].sum() >= 5]
    if not active:
        print("  no variance in any feature, skipping")
        return None
    X = sm.add_constant(df[active].astype(float))
    y = df["is_correct"].astype(int)
    try:
        result = sm.Logit(y, X).fit(disp=0, maxiter=200)
        ci = result.conf_int()
        rows = []
        for i, name in enumerate(["const"] + active):
            coef = result.params[name]
            pval = result.pvalues[name]
            if name == "const":
                rows.append({
                    "feature": name,
                    "coef": round(coef, 3),
                    "OR": float("nan"),
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "p_value": pval,
                    "sig": "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else "",
                })
            else:
                rows.append({
                    "feature": name,
                    "coef": round(coef, 3),
                    "OR": round(math.exp(coef), 3),
                    "ci_lo": round(math.exp(ci.loc[name].iloc[0]), 3),
                    "ci_hi": round(math.exp(ci.loc[name].iloc[1]), 3),
                    "p_value": pval,
                    "sig": "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else "",
                })
        out_df = pd.DataFrame(rows).sort_values("p_value")
        print(f"\n{label}  (pseudo-R² = {result.prsquared:.3f}  AIC = {result.aic:.1f})")
        print(out_df.to_string(index=False))
        return out_df
    except Exception as e:
        print(f"  combined logit failed: {e}")
        return None


def run_stats(df, label):
    print(f"\nper-motif statistics — {label}")
    summary_rows = []
    for flag in MOTIF_FLAGS:
        n_present = int(df[flag].sum())
        n_correct = int(df[df[flag] == 1]["is_correct"].sum())
        n_incorrect = n_present - n_correct
        table, fisher_or, fisher_p = fisher_test(df, flag)
        uni = univariate_logit(df, flag)
        uni_p    = uni["p_value"] if uni else float("nan")
        uni_or   = uni["or"]      if uni else float("nan")
        uni_cilo = uni["ci_lo"]   if uni else float("nan")
        uni_cihi = uni["ci_hi"]   if uni else float("nan")
        print(f"\n{flag}:")
        print(f"  present={n_present}  correct={n_correct}  incorrect={n_incorrect}")
        print(f"  contingency: {table}")
        print(f"  fisher:  OR={fisher_or:.3f}  p={_fmt_p(fisher_p)}")
        if uni:
            print(f"  logit:   OR={uni_or:.3f} [{uni_cilo:.3f}, {uni_cihi:.3f}]  p={_fmt_p(uni_p)}")
        summary_rows.append({
            "motif":        flag,
            "n_present":    n_present,
            "n_correct":    n_correct,
            "n_incorrect":  n_incorrect,
            "fisher_or":    fisher_or,
            "fisher_p":     fisher_p,
            "logit_or":     uni_or,
            "logit_ci_lo":  uni_cilo,
            "logit_ci_hi":  uni_cihi,
            "logit_p":      uni_p,
        })

    print(f"\ncombined regression: 3 new motifs — {label}")
    combined_3 = combined_logit(df, MOTIF_FLAGS, "3-motif model")

    print(f"\ncombined regression: 3 new motifs + depth + frac_facts — {label}")
    combined_5 = combined_logit(df, MOTIF_FLAGS + ["depth", "frac_facts"], "5-predictor model")

    return pd.DataFrame(summary_rows), combined_3, combined_5


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graphs",   required=True, help="path to gsm_hard_graphs.jsonl")
    p.add_argument("--traces",   required=True, help="path to gsm_hard_traces.jsonl")
    p.add_argument("--out-csv",  required=True, help="output per-trace CSV")
    p.add_argument("--label",    default="model", help="model label for display")
    args = p.parse_args()

    graphs = load_jsonl(args.graphs)
    traces = load_jsonl(args.traces)

    trace_map = {t["id"]: t for t in traces if "id" in t}

    rows = []
    for g in graphs:
        row = extract_row(g)
        qid = row["question_id"]
        if qid in trace_map and "correct" in trace_map[qid]:
            row["is_correct"] = bool(trace_map[qid]["correct"])
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"loaded {len(df)} graphs  |  correct: {df['is_correct'].sum()}  incorrect: {(~df['is_correct']).sum()}")
    print(f"wrote per-trace CSV to {out_csv}")

    for flag in MOTIF_FLAGS:
        pct = df[flag].mean() * 100
        print(f"  {flag}: {int(df[flag].sum())} present ({pct:.1f}%)")

    run_stats(df, args.label)


if __name__ == "__main__":
    main()
