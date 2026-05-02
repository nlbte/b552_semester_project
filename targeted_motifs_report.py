"""Generate a combined cross-model markdown report for targeted motifs.

Takes per-trace CSVs from both model runs (output of targeted_motifs.py)
and produces a single markdown report comparing results across models.

Usage:
    python targeted_motifs_report.py \\
        --gptoss  gsm_hard_data/targeted_motifs_gptoss.csv \\
        --gemma4  experiments/gemma4-31b-cloud/targeted_motifs_gemma4.csv \\
        --out     targeted_motifs_report.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


MOTIF_FLAGS = ["late_arithmetic", "verbose_ungrounded", "early_branching"]

MOTIF_DEFS = {
    "late_arithmetic": (
        "Any arithmetic node whose depth in the DAG (longest path from any source node) "
        "is >= 4, AND whose full ancestor set contains no `extract_fact` node. "
        "This captures arithmetic that has drifted far from the problem's given information."
    ),
    "verbose_ungrounded": (
        "`n_steps >= 8` AND `frac_facts < 0.3`. "
        "Captures long traces that still devote fewer than 30% of their steps "
        "to extracting quantities stated in the problem."
    ),
    "early_branching": (
        "Any node in the first half of the trace (by sequential step position, i.e., "
        "steps S1 through S_floor(n/2)) whose out-degree is >= 3. "
        "Captures a model that fans out into many parallel directions early, "
        "before the reasoning has converged."
    ),
}


def _fmt_p(p):
    if math.isnan(p):
        return "—"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _fmt_or(v):
    return "—" if math.isnan(v) else f"{v:.3f}"


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
        return {
            "or":    math.exp(coef),
            "ci_lo": math.exp(ci.iloc[1, 0]),
            "ci_hi": math.exp(ci.iloc[1, 1]),
            "p":     pval,
        }
    except Exception:
        return None


def combined_logit(df, features):
    active = [f for f in features if f in df.columns and df[f].std() > 0 and df[f].sum() >= 5]
    if not active:
        return None, float("nan")
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
                rows.append({"feature": name, "coef": coef, "OR": float("nan"),
                             "ci_lo": float("nan"), "ci_hi": float("nan"), "p_value": pval})
            else:
                rows.append({"feature": name, "coef": coef,
                             "OR": math.exp(coef),
                             "ci_lo": math.exp(ci.loc[name].iloc[0]),
                             "ci_hi": math.exp(ci.loc[name].iloc[1]),
                             "p_value": pval})
        return pd.DataFrame(rows).sort_values("p_value"), result.prsquared
    except Exception:
        return None, float("nan")


def build_motif_table(df, label):
    rows = []
    for flag in MOTIF_FLAGS:
        n_total   = len(df)
        n_present = int(df[flag].sum())
        n_pct     = n_present / n_total * 100
        n_correct = int(df[df[flag] == 1]["is_correct"].sum())
        n_wrong   = n_present - n_correct
        _, fisher_or, fisher_p = fisher_test(df, flag)
        uni = univariate_logit(df, flag)
        rows.append({
            "motif":       flag,
            "n_present":   n_present,
            "pct":         n_pct,
            "n_correct":   n_correct,
            "n_incorrect": n_wrong,
            "fisher_or":   fisher_or,
            "fisher_p":    fisher_p,
            "logit_or":    uni["or"]    if uni else float("nan"),
            "logit_ci_lo": uni["ci_lo"] if uni else float("nan"),
            "logit_ci_hi": uni["ci_hi"] if uni else float("nan"),
            "logit_p":     uni["p"]     if uni else float("nan"),
        })
    return pd.DataFrame(rows)


def render_reg_table(reg_df):
    lines = []
    lines.append("| feature | coef | OR | 95% CI | p-value | sig |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in reg_df.iterrows():
        sig = "***" if r["p_value"] < 0.01 else "**" if r["p_value"] < 0.05 else "*" if r["p_value"] < 0.1 else ""
        ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if not math.isnan(r["ci_lo"]) else "—"
        lines.append(
            f"| `{r['feature']}` | {r['coef']:.3f} | {_fmt_or(r['OR'])}"
            f" | {ci} | {_fmt_p(r['p_value'])} | {sig} |"
        )
    return "\n".join(lines)


def sig_symbol(p):
    if math.isnan(p):
        return "—"
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return "ns"


def generate_report(df_g, df_m, out_path):
    tbl_g = build_motif_table(df_g, "gpt-oss")
    tbl_m = build_motif_table(df_m, "gemma4")

    reg3_g, r2_3g = combined_logit(df_g, MOTIF_FLAGS)
    reg3_m, r2_3m = combined_logit(df_m, MOTIF_FLAGS)
    reg5_g, r2_5g = combined_logit(df_g, MOTIF_FLAGS + ["depth", "frac_facts"])
    reg5_m, r2_5m = combined_logit(df_m, MOTIF_FLAGS + ["depth", "frac_facts"])

    lines = []

    lines.append("# Targeted Motifs — Cross-Model Report")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Motivation")
    lines.append("")
    lines.append(
        "The five original motif flags (`has_unsupported_arith`, `has_orphan`, "
        "`has_unsupported_conclude`, `low_fact_ratio`, `long_chain`) were defined as "
        "generic structural irregularities. Most were not individually significant — "
        "only `long_chain` (depth > 5) held up in univariate tests, and `has_orphan` "
        "reached marginal significance in gpt-oss. The composite `motif_count` worked "
        "because bad patterns co-occur, not because each flag was individually diagnostic."
    )
    lines.append("")
    lines.append(
        "The three motifs tested here are designed to be more targeted. Each is grounded "
        "in a specific failure mechanism observed in the error analysis: arithmetic that "
        "has drifted away from the problem's given information, long traces that never "
        "anchor to the problem statement, and early over-branching before the reasoning "
        "has converged. Unlike the original flags, these are not just 'looks messy' "
        "indicators — each operationalizes a distinct hypothesis about how reasoning "
        "goes wrong."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Motif Definitions")
    lines.append("")
    for flag, defn in MOTIF_DEFS.items():
        lines.append(f"**`{flag}`** — {defn}")
        lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Per-Model Results")
    lines.append("")

    for label, tbl, df in [("gpt-oss:120b", tbl_g, df_g), ("gemma4:31b", tbl_m, df_m)]:
        n_correct = int(df["is_correct"].sum())
        n_wrong   = int((~df["is_correct"]).sum())
        n_total   = len(df)
        lines.append(f"### {label}  (n={n_total}, correct={n_correct}, incorrect={n_wrong})")
        lines.append("")
        lines.append("| motif | n present | % of traces | correct | incorrect | Fisher OR | Fisher p | Logit OR | 95% CI | Logit p |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in tbl.iterrows():
            ci = (f"[{r['logit_ci_lo']:.3f}, {r['logit_ci_hi']:.3f}]"
                  if not math.isnan(r["logit_ci_lo"]) else "—")
            lines.append(
                f"| `{r['motif']}` | {r['n_present']} | {r['pct']:.1f}%"
                f" | {r['n_correct']} | {r['n_incorrect']}"
                f" | {_fmt_or(r['fisher_or'])} | {_fmt_p(r['fisher_p'])}"
                f" | {_fmt_or(r['logit_or'])} | {ci} | {_fmt_p(r['logit_p'])} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")

    lines.append("## Cross-Model Comparison")
    lines.append("")
    lines.append(
        "The table below shows whether each motif replicates across both models. "
        "Replication is defined as a consistent direction of effect (OR < 1 for "
        "correctness, i.e., motif predicts failure) and p < 0.10 in at least one "
        "test (Fisher or logit) in both models."
    )
    lines.append("")
    lines.append("| motif | gpt-oss Fisher p | gpt-oss Logit OR | gemma4 Fisher p | gemma4 Logit OR | replicates |")
    lines.append("|---|---|---|---|---|---|")

    for _, rg in tbl_g.iterrows():
        flag = rg["motif"]
        rm   = tbl_m[tbl_m["motif"] == flag].iloc[0]
        g_sig = min(rg["fisher_p"], rg["logit_p"])
        m_sig = min(rm["fisher_p"], rm["logit_p"])
        g_dir = rg["logit_or"] < 1 if not math.isnan(rg["logit_or"]) else False
        m_dir = rm["logit_or"] < 1 if not math.isnan(rm["logit_or"]) else False
        replicates = "yes" if (g_sig < 0.10 and m_sig < 0.10 and g_dir and m_dir) else "no"
        lines.append(
            f"| `{flag}` | {_fmt_p(rg['fisher_p'])} | {_fmt_or(rg['logit_or'])}"
            f" | {_fmt_p(rm['fisher_p'])} | {_fmt_or(rm['logit_or'])}"
            f" | {replicates} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Combined Regressions")
    lines.append("")
    lines.append(
        "Two combined regressions are reported for each model. The first includes only "
        "the three new motifs to assess their joint predictive power. The second adds "
        "`depth` and `frac_facts` — the two strongest predictors from the original "
        "feature analysis — as covariates. A new motif adds independent signal if it "
        "remains significant after conditioning on depth and frac_facts."
    )
    lines.append("")
    lines.append(
        "Note that `verbose_ungrounded` is defined partly in terms of `frac_facts` "
        "(it fires when frac_facts < 0.3), so collinearity between these two predictors "
        "is expected in the second regression. Interpret `verbose_ungrounded` coefficients "
        "in the 5-predictor model with caution."
    )
    lines.append("")

    for label, reg3, r2_3, reg5, r2_5 in [
        ("gpt-oss:120b", reg3_g, r2_3g, reg5_g, r2_5g),
        ("gemma4:31b",   reg3_m, r2_3m, reg5_m, r2_5m),
    ]:
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"**3-motif model** (pseudo-R² = {r2_3:.3f})")
        lines.append("")
        if reg3 is not None:
            lines.append(render_reg_table(reg3))
        else:
            lines.append("_regression could not be fit_")
        lines.append("")
        lines.append(f"**5-predictor model** (3 new motifs + depth + frac_facts, pseudo-R² = {r2_5:.3f})")
        lines.append("")
        if reg5 is not None:
            lines.append(render_reg_table(reg5))
        else:
            lines.append("_regression could not be fit_")
        lines.append("")

    lines.append("---")
    lines.append("")

    lines.append("## Discussion")
    lines.append("")

    motif_outcomes = {}
    for flag in MOTIF_FLAGS:
        rg = tbl_g[tbl_g["motif"] == flag].iloc[0]
        rm = tbl_m[tbl_m["motif"] == flag].iloc[0]
        g_sig = min(rg["fisher_p"], rg["logit_p"])
        m_sig = min(rm["fisher_p"], rm["logit_p"])
        g_dir = rg["logit_or"] < 1 if not math.isnan(rg["logit_or"]) else None
        m_dir = rm["logit_or"] < 1 if not math.isnan(rm["logit_or"]) else None
        motif_outcomes[flag] = {
            "g_sig": g_sig, "m_sig": m_sig,
            "g_dir": g_dir, "m_dir": m_dir,
            "g_or": rg["logit_or"], "m_or": rm["logit_or"],
            "g_fp": rg["fisher_p"], "m_fp": rm["fisher_p"],
        }

    o = motif_outcomes

    la = o["late_arithmetic"]
    vu = o["verbose_ungrounded"]
    eb = o["early_branching"]

    lines.append(
        f"`late_arithmetic` shows a Fisher p of {_fmt_p(la['g_fp'])} in gpt-oss and "
        f"{_fmt_p(la['m_fp'])} in gemma4. "
    )
    if la["g_sig"] < 0.10 and la["m_sig"] < 0.10 and la["g_dir"] and la["m_dir"]:
        lines.append(
            "Both models agree on the direction: arithmetic nodes that appear deep in "
            "the reasoning chain without any fact-extraction ancestor are associated "
            "with failure. This is consistent with the interpretation that the model "
            "has lost track of the problem's given quantities and is computing over "
            "intermediate values that are no longer grounded. The effect replicates "
            "across models, which is encouraging given the relatively small number "
            "of incorrect traces (39-41 per model)."
        )
    elif la["g_sig"] < 0.10 or la["m_sig"] < 0.10:
        lines.append(
            "The direction is consistent but the effect is significant in only one "
            "model. This may reflect low statistical power — the motif is rare, and "
            "39-41 incorrect traces is a small sample for detecting individually "
            "weak signals. The effect should be treated as suggestive rather than "
            "confirmed."
        )
    else:
        lines.append(
            "Neither model shows a significant effect for this motif. Either the "
            "hypothesis is wrong, the motif is too rarely triggered to have power, "
            "or the graph extractor does not reliably distinguish ungrounded arithmetic "
            "from grounded arithmetic in its op labeling."
        )
    lines.append("")

    lines.append(
        f"`verbose_ungrounded` fires when n_steps >= 8 AND frac_facts < 0.3. "
        f"It shows Fisher p = {_fmt_p(vu['g_fp'])} (gpt-oss) and "
        f"{_fmt_p(vu['m_fp'])} (gemma4). "
    )
    if vu["g_sig"] < 0.10 and vu["m_sig"] < 0.10 and vu["g_dir"] and vu["m_dir"]:
        lines.append(
            "This replicates across models. However, it is worth noting that `verbose_ungrounded` "
            "is a composite of two features that were already individually significant — "
            "`n_steps` (or `depth`) and `frac_facts`. The motif is therefore best understood "
            "as a convenient threshold rule rather than a new mechanistic signal. In the "
            "5-predictor regression, its coefficient will absorb some of the variance "
            "already explained by `frac_facts`, so the independent contribution "
            "is likely smaller than the univariate result suggests."
        )
    elif vu["g_sig"] < 0.10 or vu["m_sig"] < 0.10:
        lines.append(
            "The motif is significant in one model but not both. Since it directly "
            "operationalizes two already-significant features, partial replication is "
            "expected. It does not provide strong evidence of a distinct failure mechanism "
            "beyond what depth and frac_facts already capture."
        )
    else:
        lines.append(
            "Despite being composed of two individually significant features, the "
            "conjunction fails to reach significance. This likely means the AND condition "
            "selects a small and atypical subset — long traces with low fact ratios may "
            "not be the most failure-prone category relative to either feature alone."
        )
    lines.append("")

    lines.append(
        f"`early_branching` shows Fisher p = {_fmt_p(eb['g_fp'])} (gpt-oss) and "
        f"{_fmt_p(eb['m_fp'])} (gemma4). "
    )
    if eb["g_sig"] < 0.10 and eb["m_sig"] < 0.10 and eb["g_dir"] and eb["m_dir"]:
        lines.append(
            "Both models show that high out-degree in the early steps is associated "
            "with failure. Parallel branching early in the reasoning chain appears to "
            "introduce confusion — the model commits to multiple competing directions "
            "before it has enough grounding to choose between them."
        )
    elif eb["g_sig"] < 0.10 or eb["m_sig"] < 0.10:
        lines.append(
            "The motif is significant in one model. It is plausible that the two models "
            "have different branching styles — gpt-oss's more verbose traces may produce "
            "higher out-degrees at earlier steps, making the motif more detectable there."
        )
    else:
        lines.append(
            "Neither model shows a significant effect. High out-degree early in the trace "
            "may not be uncommon in correct traces as well — some problems genuinely require "
            "extracting multiple independent facts in parallel early on, and the model "
            "correctly does so. The threshold of out_degree >= 3 may be too coarse."
        )
    lines.append("")

    lines.append(
        "In the 5-predictor regressions, any new motif that retains significance after "
        "conditioning on depth and frac_facts provides evidence of an independent "
        "structural signal. If none do, the result still has value: it narrows the "
        "explanation to those two features being sufficient predictors of failure within "
        "this dataset and sample size."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "All motifs are operationally defined on LLM-segmented graphs. The graph "
        "extractor (gpt-oss:120b-cloud or claude-opus-4-6) decides which steps count as "
        "`extract_fact` vs. `arithmetic` vs. `other`, and how dependencies are drawn. "
        "If the extractor is inconsistent — labeling the same type of step differently "
        "across problems — then motif rates partly reflect the extractor's noise rather "
        "than the reasoning model's behavior. There is no ground truth for the graph "
        "structure, only the extractor's interpretation of the trace."
    )
    lines.append("")
    lines.append(
        "Both models were evaluated on n=200 problems with 39-41 incorrect traces. "
        "This gives limited power for detecting individually weak effects. Fisher's exact "
        "test and univariate logistic regression are both underpowered when the motif is "
        "rare (fewer than ~15-20 positive cases). Motifs that fire in fewer than 10% of "
        "traces should be interpreted with extra caution."
    )
    lines.append("")
    lines.append(
        "The combined regressions use unstandardized predictors, so coefficient magnitudes "
        "are not directly comparable across features. `frac_facts` ranges from 0 to 1 while "
        "`depth` and the binary motif flags have very different scales. The odds ratios are "
        "interpretable but the coefficients in the combined model should not be rank-ordered "
        "as importance scores."
    )
    lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote combined report to {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gptoss", required=True, help="per-trace CSV from gpt-oss run")
    p.add_argument("--gemma4", required=True, help="per-trace CSV from gemma4 run")
    p.add_argument("--out",    default="targeted_motifs_report.md", help="output markdown path")
    args = p.parse_args()

    df_g = pd.read_csv(args.gptoss)
    df_m = pd.read_csv(args.gemma4)

    df_g["is_correct"] = df_g["is_correct"].astype(bool)
    df_m["is_correct"] = df_m["is_correct"].astype(bool)

    print(f"gpt-oss:  {len(df_g)} traces  correct={df_g['is_correct'].sum()}  incorrect={(~df_g['is_correct']).sum()}")
    print(f"gemma4:   {len(df_m)} traces  correct={df_m['is_correct'].sum()}  incorrect={(~df_m['is_correct']).sum()}")

    generate_report(df_g, df_m, args.out)


if __name__ == "__main__":
    main()
