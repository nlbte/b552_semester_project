"""Compute five additional reasoning motifs and run statistical tests.

Motifs:
  has_offschema_node   -- any step with op not in {extract_fact, arithmetic, substitute, conclude, verify}
  orphan_fact          -- extract_fact node with out_degree == 0 (fact extracted but never used)
  linear_chain         -- n_steps >= 7 AND mean_out_degree <= 1.1 AND max_out_degree <= 2
  high_fanin_conclude  -- any conclude node with in_degree >= 3
  failed_verification  -- trace contains at least one verify node

Usage:
    python additional_motifs.py \\
        --graphs gsm_hard_data/gsm_hard_graphs_all.jsonl \\
        --traces gsm_hard_data/gsm_hard_traces.jsonl \\
        --label gpt-oss \\
        --out-csv gsm_hard_data/additional_motifs_gptoss.csv

    python additional_motifs.py \\
        --graphs experiments/gemma4-31b-cloud/gsm_hard_graphs.jsonl \\
        --traces experiments/gemma4-31b-cloud/gsm_hard_traces.jsonl \\
        --label gemma4 \\
        --out-csv experiments/gemma4-31b-cloud/additional_motifs_gemma4.csv \\
        --report experiments/gemma4-31b-cloud/additional_motifs_gemma4.md
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

SCHEMA_OPS = {"extract_fact", "arithmetic", "substitute", "conclude", "verify"}

MOTIF_FLAGS = [
    "has_offschema_node",
    "orphan_fact",
    "linear_chain",
    "high_fanin_conclude",
    "failed_verification",
]


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


def flag_has_offschema_node(steps):
    return any(s.get("op", "unknown") not in SCHEMA_OPS for s in steps)


def flag_orphan_fact(g, steps):
    for s in steps:
        if s.get("op") == "extract_fact":
            nid = s["id"]
            if nid in g.nodes and g.out_degree(nid) == 0:
                return True
    return False


def flag_linear_chain(g, steps):
    n = len(steps)
    if n < 7:
        return False
    if len(g.nodes) == 0:
        return False
    degrees = [g.out_degree(nd) for nd in g.nodes()]
    mean_out = sum(degrees) / len(degrees)
    max_out = max(degrees)
    return mean_out <= 1.1 and max_out <= 2


def flag_high_fanin_conclude(g, steps):
    for s in steps:
        if s.get("op") == "conclude":
            nid = s["id"]
            if nid in g.nodes and g.in_degree(nid) >= 3:
                return True
    return False


def flag_failed_verification(steps):
    return any(s.get("op") == "verify" for s in steps)


def graph_longest_path(g):
    if len(g) == 0:
        return 0
    if not nx.is_directed_acyclic_graph(g):
        return -1
    try:
        return nx.dag_longest_path_length(g)
    except Exception:
        return -1


def extract_row(graph_rec):
    steps = graph_rec.get("steps", [])
    g = build_nx(steps)
    n = len(steps)
    n_facts = sum(1 for s in steps if s.get("op") == "extract_fact")
    frac_f = n_facts / n if n else 0.0

    degrees = [g.out_degree(nd) for nd in g.nodes()] if len(g.nodes()) > 0 else [0]
    mean_out = sum(degrees) / len(degrees)

    lc = int(flag_linear_chain(g, steps))
    return {
        "question_id": graph_rec.get("question_id") or graph_rec.get("id"),
        "is_correct": bool(graph_rec.get("is_correct", False)),
        "has_offschema_node": int(flag_has_offschema_node(steps)),
        "orphan_fact": int(flag_orphan_fact(g, steps)),
        "linear_chain": lc,
        "high_fanin_conclude": int(flag_high_fanin_conclude(g, steps)),
        "failed_verification": int(flag_failed_verification(steps)),
        "linear_chain_x_nsteps": lc * n,
        "depth": graph_longest_path(g),
        "n_steps": n,
        "frac_facts": round(frac_f, 6),
    }


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _fmt_p(p):
    if math.isnan(p):
        return "—"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def fisher_test(df, flag):
    present = df[df[flag] == 1]
    absent = df[df[flag] == 0]
    table = [
        [int((~present["is_correct"]).sum()), int(present["is_correct"].sum())],
        [int((~absent["is_correct"]).sum()), int(absent["is_correct"].sum())],
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
    active = [
        f for f in features
        if f in df.columns and df[f].std() > 0 and df[f].sum() >= 5
    ]
    if not active:
        print("  no variance in any feature, skipping")
        return None
    X = sm.add_constant(df[active].astype(float))
    y = df["is_correct"].astype(int)
    try:
        result = sm.Logit(y, X).fit(disp=0, maxiter=200)
        ci = result.conf_int()
        rows = []
        for name in ["const"] + active:
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
    section(f"per-motif statistics — {label}")
    summary_rows = []
    for flag in MOTIF_FLAGS:
        n_present = int(df[flag].sum())
        n_correct = int(df[df[flag] == 1]["is_correct"].sum())
        n_incorrect = n_present - n_correct
        table, fisher_or, fisher_p = fisher_test(df, flag)
        uni = univariate_logit(df, flag)
        uni_p = uni["p_value"] if uni else float("nan")
        uni_or = uni["or"] if uni else float("nan")
        uni_cilo = uni["ci_lo"] if uni else float("nan")
        uni_cihi = uni["ci_hi"] if uni else float("nan")
        print(f"\n{flag}:")
        print(f"  present={n_present}  correct={n_correct}  incorrect={n_incorrect}")
        print(f"  contingency: {table}")
        print(f"  fisher:  OR={fisher_or:.3f}  p={_fmt_p(fisher_p)}")
        if uni:
            print(f"  logit:   OR={uni_or:.3f} [{uni_cilo:.3f}, {uni_cihi:.3f}]  p={_fmt_p(uni_p)}")
        summary_rows.append({
            "motif": flag,
            "n_present": n_present,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "fisher_or": fisher_or,
            "fisher_p": fisher_p,
            "logit_or": uni_or,
            "logit_ci_lo": uni_cilo,
            "logit_ci_hi": uni_cihi,
            "logit_p": uni_p,
        })

    section(f"combined regression: 5 new motifs — {label}")
    combined_5 = combined_logit(df, MOTIF_FLAGS, "5-motif model")

    section(f"combined regression: 5 new motifs + depth + n_steps + frac_facts + interaction — {label}")
    extended = MOTIF_FLAGS + ["depth", "n_steps", "frac_facts", "linear_chain_x_nsteps"]
    combined_ext = combined_logit(df, extended, "extended model")

    return pd.DataFrame(summary_rows), combined_5, combined_ext


def generate_single_report(label, summary_df, combined_5, combined_ext, out_path):
    def fmt_or(v):
        return "—" if math.isnan(v) else f"{v:.3f}"

    lines = [f"# Additional Motifs — {label}\n"]

    lines.append("## Per-Motif Results\n")
    lines.append("| motif | n_present | n_correct | n_incorrect | fisher OR | fisher p | logit OR | 95% CI | logit p |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in summary_df.iterrows():
        ci = f"[{r['logit_ci_lo']:.3f}, {r['logit_ci_hi']:.3f}]" if not math.isnan(r["logit_ci_lo"]) else "—"
        lines.append(
            f"| `{r['motif']}` | {r['n_present']} | {r['n_correct']} | {r['n_incorrect']}"
            f" | {fmt_or(r['fisher_or'])} | {_fmt_p(r['fisher_p'])}"
            f" | {fmt_or(r['logit_or'])} | {ci} | {_fmt_p(r['logit_p'])} |"
        )
    lines.append("")

    for reg_df, title in [
        (combined_5, "Combined Regression: 5 New Motifs"),
        (combined_ext, "Combined Regression: 5 New Motifs + depth + n_steps + frac_facts + Interaction"),
    ]:
        if reg_df is None:
            continue
        lines.append(f"## {title}\n")
        lines.append("| feature | coef | OR | 95% CI | p-value | sig |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in reg_df.iterrows():
            ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if not math.isnan(r["ci_lo"]) else "—"
            lines.append(
                f"| `{r['feature']}` | {r['coef']:.3f} | {fmt_or(r['OR'])}"
                f" | {ci} | {_fmt_p(r['p_value'])} | {r['sig']} |"
            )
        lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote single-model report to {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graphs", required=True, help="path to gsm_hard_graphs.jsonl")
    p.add_argument("--traces", required=True, help="path to gsm_hard_traces.jsonl")
    p.add_argument("--out-csv", required=True, help="output per-trace CSV")
    p.add_argument("--label", default="model", help="model label for display")
    p.add_argument("--report", default="", help="write per-model markdown report to this path")
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

    summary_df, combined_5, combined_ext = run_stats(df, args.label)

    if args.report:
        generate_single_report(args.label, summary_df, combined_5, combined_ext, args.report)


if __name__ == "__main__":
    main()
