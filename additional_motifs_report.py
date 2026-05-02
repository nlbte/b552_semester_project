"""Generate a combined cross-model report for the five additional motifs.

Loads per-model CSVs produced by additional_motifs.py, reruns all statistics,
and writes additional_motifs_report.md.

Usage:
    python additional_motifs_report.py \\
        --gptoss  gsm_hard_data/additional_motifs_gptoss.csv \\
        --gemma4  experiments/gemma4-31b-cloud/additional_motifs_gemma4.csv \\
        --out     additional_motifs_report.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

MOTIF_FLAGS = [
    "has_offschema_node",
    "orphan_fact",
    "linear_chain",
    "high_fanin_conclude",
    "failed_verification",
]

# failed_verification is expected to be diagnostic (reverse direction), noted in report
DIAGNOSTIC_MOTIFS = {"failed_verification"}


def _fmt_p(p):
    if math.isnan(p):
        return "—"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def fmt_or(v):
    return "—" if math.isnan(v) else f"{v:.3f}"


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
        return pd.DataFrame(rows).sort_values("p_value"), result.prsquared, result.aic
    except Exception as e:
        print(f"  combined logit failed ({label}): {e}")
        return None, None, None


def run_model_stats(df, label):
    rows = []
    for flag in MOTIF_FLAGS:
        n_present = int(df[flag].sum())
        n_correct = int(df[df[flag] == 1]["is_correct"].sum())
        n_incorrect = n_present - n_correct
        table, fisher_or, fisher_p = fisher_test(df, flag)
        uni = univariate_logit(df, flag)
        rows.append({
            "motif": flag,
            "model": label,
            "n_present": n_present,
            "pct_present": round(n_present / len(df) * 100, 1),
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "fisher_or": fisher_or,
            "fisher_p": fisher_p,
            "logit_or": uni["or"] if uni else float("nan"),
            "logit_ci_lo": uni["ci_lo"] if uni else float("nan"),
            "logit_ci_hi": uni["ci_hi"] if uni else float("nan"),
            "logit_p": uni["p_value"] if uni else float("nan"),
        })
    return pd.DataFrame(rows)


def replicates(gptoss_row, gemma4_row):
    """Consistent OR direction (OR < 1 for correctness) and p < 0.10 in both models."""
    g_p = min(gptoss_row["fisher_p"], gptoss_row["logit_p"])
    m_p = min(gemma4_row["fisher_p"], gemma4_row["logit_p"])
    g_or = gptoss_row["logit_or"] if not math.isnan(gptoss_row["logit_or"]) else gptoss_row["fisher_or"]
    m_or = gemma4_row["logit_or"] if not math.isnan(gemma4_row["logit_or"]) else gemma4_row["fisher_or"]
    same_dir = (g_or < 1 and m_or < 1) or (g_or > 1 and m_or > 1)
    return same_dir and g_p < 0.10 and m_p < 0.10


def generate_report(gptoss_df, gemma4_df, out_path):
    gptoss_stats = run_model_stats(gptoss_df, "gpt-oss")
    gemma4_stats = run_model_stats(gemma4_df, "gemma4")

    n_gptoss = len(gptoss_df)
    n_gemma4 = len(gemma4_df)
    c_gptoss = int(gptoss_df["is_correct"].sum())
    c_gemma4 = int(gemma4_df["is_correct"].sum())

    lines = ["# Additional Motifs — Cross-Model Report\n", "---\n"]

    lines.append("## Motivation\n")
    lines.append(
        "The five original motifs and three targeted motifs tested previously were grounded in "
        "generic structural irregularities and specific hypotheses about arithmetic drift, "
        "verbose ungrounded reasoning, and early branching. This analysis tests five additional "
        "structural flags: off-schema node labels, extract_fact nodes that are never consumed, "
        "purely linear reasoning chains, high fan-in conclude nodes, and the presence of "
        "explicit verification steps. The last motif (`failed_verification`) is diagnostic "
        "rather than predictive — it tests whether models that attempt to verify their "
        "answers actually do so on harder problems, regardless of whether verification improves accuracy.\n"
    )
    lines.append("---\n")

    lines.append("## Motif Definitions\n")
    lines.append(
        "**`has_offschema_node`** — Any step whose `op` label is not in "
        "`{extract_fact, arithmetic, substitute, conclude, verify}`. "
        "Captures traces where the LLM segmenter (or the reasoning model) produced steps "
        "that do not fit the expected reasoning schema.\n"
    )
    lines.append(
        "**`orphan_fact`** — Any `extract_fact` node with out-degree 0 "
        "(the fact was extracted but never used in any downstream step). "
        "Captures dead-end information gathering.\n"
    )
    lines.append(
        "**`linear_chain`** — `n_steps >= 7` AND `mean_out_degree <= 1.1` AND `max_out_degree <= 2`. "
        "Captures a trace that proceeds in an almost purely sequential manner with minimal branching.\n"
    )
    lines.append(
        "**`high_fanin_conclude`** — Any `conclude` node with in-degree >= 3. "
        "Captures a conclusion that synthesizes three or more upstream steps, "
        "which may indicate a high-integration step or a poorly structured convergence.\n"
    )
    lines.append(
        "**`failed_verification`** — The trace contains at least one `verify` step. "
        "Despite the name, this is diagnostic: it measures whether the model attempted verification, "
        "not whether verification succeeded. If verification steps appear more in incorrect traces, "
        "this suggests the model uses verification as a recovery attempt on problems it is already struggling with.\n"
    )
    lines.append("---\n")

    lines.append("## Per-Model Results\n")
    for stats_df, label, n_total, n_correct in [
        (gptoss_stats, "gpt-oss:120b", n_gptoss, c_gptoss),
        (gemma4_stats, "gemma4:31b", n_gemma4, c_gemma4),
    ]:
        lines.append(f"### {label}  (n={n_total}, correct={n_correct}, incorrect={n_total - n_correct})\n")
        lines.append("| motif | n present | % of traces | correct | incorrect | Fisher OR | Fisher p | Logit OR | 95% CI | Logit p |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in stats_df.iterrows():
            ci = f"[{r['logit_ci_lo']:.3f}, {r['logit_ci_hi']:.3f}]" if not math.isnan(r["logit_ci_lo"]) else "—"
            lines.append(
                f"| `{r['motif']}` | {r['n_present']} | {r['pct_present']}% | {r['n_correct']} | {r['n_incorrect']}"
                f" | {fmt_or(r['fisher_or'])} | {_fmt_p(r['fisher_p'])}"
                f" | {fmt_or(r['logit_or'])} | {ci} | {_fmt_p(r['logit_p'])} |"
            )
        lines.append("")

    lines.append("---\n")

    lines.append("## Cross-Model Comparison\n")
    lines.append(
        "Replication is defined as consistent OR direction and p < 0.10 in both models "
        "(Fisher or logit). `failed_verification` is excluded from the replication criterion "
        "because it is diagnostic rather than predictive.\n"
    )
    lines.append("| motif | gpt-oss Fisher p | gpt-oss Logit OR | gemma4 Fisher p | gemma4 Logit OR | replicates |")
    lines.append("|---|---|---|---|---|---|")
    for flag in MOTIF_FLAGS:
        g_row = gptoss_stats[gptoss_stats["motif"] == flag].iloc[0]
        m_row = gemma4_stats[gemma4_stats["motif"] == flag].iloc[0]
        if flag in DIAGNOSTIC_MOTIFS:
            rep_str = "diagnostic"
        else:
            rep_str = "yes" if replicates(g_row, m_row) else "no"
        lines.append(
            f"| `{flag}` | {_fmt_p(g_row['fisher_p'])} | {fmt_or(g_row['logit_or'])}"
            f" | {_fmt_p(m_row['fisher_p'])} | {fmt_or(m_row['logit_or'])} | {rep_str} |"
        )
    lines.append("")
    lines.append("---\n")

    lines.append("## Combined Regressions\n")
    lines.append(
        "Two combined regressions are reported for each model. The first includes only the five new motifs. "
        "The second adds `depth`, `n_steps`, `frac_facts`, and the `linear_chain_x_nsteps` interaction term. "
        "The interaction term tests whether linear chains are only harmful (or helpful) in combination with "
        "longer traces. A motif that retains significance in the extended model provides evidence of an "
        "independent structural signal beyond the basic trace-length and fact-ratio features.\n"
    )

    for df, label in [(gptoss_df, "gpt-oss:120b"), (gemma4_df, "gemma4:31b")]:
        lines.append(f"### {label}\n")

        result_5 = combined_logit(df, MOTIF_FLAGS, "5-motif")
        if result_5[0] is not None:
            reg_df, r2, aic = result_5
            lines.append(f"**5-motif model** (pseudo-R² = {r2:.3f})\n")
            lines.append("| feature | coef | OR | 95% CI | p-value | sig |")
            lines.append("|---|---|---|---|---|---|")
            for _, r in reg_df.iterrows():
                ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if not math.isnan(r["ci_lo"]) else "—"
                lines.append(
                    f"| `{r['feature']}` | {r['coef']:.3f} | {fmt_or(r['OR'])}"
                    f" | {ci} | {_fmt_p(r['p_value'])} | {r['sig']} |"
                )
            lines.append("")

        ext_features = MOTIF_FLAGS + ["depth", "n_steps", "frac_facts", "linear_chain_x_nsteps"]
        result_ext = combined_logit(df, ext_features, "extended")
        if result_ext[0] is not None:
            reg_df, r2, aic = result_ext
            lines.append(f"**Extended model** (5 new motifs + depth + n_steps + frac_facts + interaction, pseudo-R² = {r2:.3f})\n")
            lines.append("| feature | coef | OR | 95% CI | p-value | sig |")
            lines.append("|---|---|---|---|---|---|")
            for _, r in reg_df.iterrows():
                ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if not math.isnan(r["ci_lo"]) else "—"
                lines.append(
                    f"| `{r['feature']}` | {r['coef']:.3f} | {fmt_or(r['OR'])}"
                    f" | {ci} | {_fmt_p(r['p_value'])} | {r['sig']} |"
                )
            lines.append("")

    lines.append("---\n")

    lines.append("## Discussion\n")
    lines.append(
        "`has_offschema_node` captures traces with steps whose `op` label falls outside the five-category schema. "
        "This flag merges two conceptually different phenomena: steps labeled `other` (the segmenter could not classify "
        "the step) and steps labeled `substitute` (a schema-legal but infrequent operation). "
        "If the flag is significant, it may reflect the segmenter's uncertainty rather than the reasoning model's behavior. "
        "However, if `substitute` steps appear disproportionately in incorrect traces, that would suggest a different failure mode.\n"
    )
    lines.append(
        "`orphan_fact` fires when a fact is extracted but never referenced downstream. "
        "With only ~3 occurrences per model in the probe data, this motif is expected to hit the min-count guard "
        "and logistic regression will not be run. Fisher's exact test will be reported but should be interpreted "
        "with extreme caution given the rarity.\n"
    )
    lines.append(
        "`linear_chain` flags traces with `n_steps >= 7`, `mean_out_degree <= 1.1`, and `max_out_degree <= 2`. "
        "These traces proceed in an almost purely sequential manner. The probe showed this fires in ~43% of traces "
        "with roughly equal rates in correct and incorrect traces, suggesting linear chains are not "
        "individually diagnostic of failure. The `linear_chain_x_nsteps` interaction in the extended regression "
        "tests whether long linear chains specifically (not just linear chains generally) predict failure.\n"
    )
    lines.append(
        "`high_fanin_conclude` requires a conclude node with in-degree >= 3. "
        "The threshold was set at 3 (rather than 4) because the probe showed the maximum observed conclude "
        "in-degree across both datasets is 3 — a threshold of 4 would produce zero positive cases. "
        "With in-degree >= 3, the motif fires in approximately 5 traces in gpt-oss and 1 in gemma4, "
        "so the min-count guard will likely apply for gemma4.\n"
    )
    lines.append(
        "`failed_verification` is the diagnostic motif. The probe showed verify steps appear in "
        "approximately 39% of incorrect gpt-oss traces vs. 23% of correct traces, "
        "and 21% of incorrect gemma4 traces vs. 15% of correct traces. "
        "This reverses the expected direction: if verification were a positive reasoning strategy, "
        "we would expect it to appear more in correct traces. The observed pattern instead suggests "
        "that models invoke verification as a compensatory behavior on problems they are already struggling with. "
        "This is consistent with the interpretation that explicit self-checking in chain-of-thought "
        "is triggered by difficulty, not by rigor.\n"
    )
    lines.append("---\n")

    lines.append("## Limitations\n")
    lines.append(
        "The `has_offschema_node` flag mixes `other` and `substitute` steps. "
        "If `substitute` steps are schema-legal and mechanistically distinct from `other` steps, "
        "combining them into one binary flag may dilute the signal from truly off-schema behavior. "
        "Separate flags for `other` vs. `substitute` would require additional schema redesign.\n"
    )
    lines.append(
        "`orphan_fact` and `high_fanin_conclude` (for gemma4) will trigger the min-count guard "
        "and logistic regression results will not be computed. Fisher's exact test will still be reported "
        "but its power is severely limited at n < 5 positive cases.\n"
    )
    lines.append(
        "The `failed_verification` result should not be interpreted as 'verification causes failure.' "
        "The causality likely runs the other way: hard problems elicit verification attempts. "
        "Disentangling this would require controlling for problem difficulty, which the current dataset "
        "does not support at scale (see the within-problem analysis, which has only 14 discordant pairs).\n"
    )

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote cross-model report to {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gptoss", required=True, help="path to additional_motifs_gptoss.csv")
    p.add_argument("--gemma4", required=True, help="path to additional_motifs_gemma4.csv")
    p.add_argument("--out", required=True, help="output markdown report path")
    args = p.parse_args()

    gptoss_df = pd.read_csv(args.gptoss)
    gemma4_df = pd.read_csv(args.gemma4)

    gptoss_df["is_correct"] = gptoss_df["is_correct"].astype(bool)
    gemma4_df["is_correct"] = gemma4_df["is_correct"].astype(bool)

    print(f"gpt-oss: {len(gptoss_df)} rows, correct={gptoss_df['is_correct'].sum()}")
    print(f"gemma4:  {len(gemma4_df)} rows, correct={gemma4_df['is_correct'].sum()}")

    for flag in MOTIF_FLAGS:
        for df, label in [(gptoss_df, "gpt-oss"), (gemma4_df, "gemma4")]:
            if flag in df.columns:
                print(f"  {label} {flag}: {int(df[flag].sum())} present")

    generate_report(gptoss_df, gemma4_df, args.out)


if __name__ == "__main__":
    main()
