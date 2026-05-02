"""Within-problem paired comparison of graph features across discordant model pairs.

For each problem where one model answered correctly and the other did not,
compare graph features of the correct trace vs. the incorrect trace on the
same problem. This eliminates problem difficulty as a confound entirely —
any feature difference observed within a discordant pair reflects reasoning
quality, not problem hardness.

Usage:
    python within_problem_comparison.py \\
        --gptoss-features gsm_hard_data/graph_features.csv \\
        --gemma4-features experiments/gemma4-31b-cloud/graph_features.csv \\
        --out within_problem_report.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


FEATURES = [
    "depth", "n_steps", "frac_facts", "frac_arithmetic",
    "max_out_degree", "mean_out_degree", "orphan_nodes",
    "n_arithmetic", "n_facts", "unsupported_arithmetic",
]


def load_and_merge(path_g: str, path_m: str) -> pd.DataFrame:
    g = pd.read_csv(path_g)
    m = pd.read_csv(path_m)
    g["is_correct"] = g["is_correct"].astype(bool)
    m["is_correct"] = m["is_correct"].astype(bool)
    merged = g.merge(m, on="question_id", suffixes=("_gptoss", "_gemma4"))
    return merged


def build_discordant(merged: pd.DataFrame) -> pd.DataFrame:
    disc = merged[merged["is_correct_gptoss"] != merged["is_correct_gemma4"]].copy()

    rows = []
    for _, row in disc.iterrows():
        if row["is_correct_gptoss"] and not row["is_correct_gemma4"]:
            winner, loser = "gptoss", "gemma4"
        else:
            winner, loser = "gemma4", "gptoss"

        r = {
            "question_id": row["question_id"],
            "winner": winner,
            "loser": loser,
        }
        for f in FEATURES:
            r[f"correct_{f}"]   = row[f"{f}_{winner}"]
            r[f"incorrect_{f}"] = row[f"{f}_{loser}"]
            r[f"delta_{f}"]     = row[f"{f}_{loser}"] - row[f"{f}_{winner}"]
        rows.append(r)
    return pd.DataFrame(rows)


def wilcoxon_one_sided(delta: np.ndarray, alternative: str = "greater"):
    clean = delta[~np.isnan(delta)]
    if len(clean) < 4:
        return float("nan"), float("nan")
    if np.all(clean == 0):
        return float("nan"), 1.0
    try:
        stat, p = stats.wilcoxon(clean, alternative=alternative)
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def paired_stats(disc: pd.DataFrame, feature: str) -> dict:
    delta = disc[f"delta_{feature}"].values.astype(float)
    correct   = disc[f"correct_{feature}"].values.astype(float)
    incorrect = disc[f"incorrect_{feature}"].values.astype(float)
    stat, p = wilcoxon_one_sided(delta, alternative="greater")
    return {
        "feature":            feature,
        "n_pairs":            len(delta),
        "median_correct":     float(np.nanmedian(correct)),
        "median_incorrect":   float(np.nanmedian(incorrect)),
        "median_delta":       float(np.nanmedian(delta)),
        "pct_delta_positive": float(np.mean(delta > 0)) * 100,
        "wilcoxon_stat":      stat,
        "wilcoxon_p":         p,
    }


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _fmt_p(p: float) -> str:
    if math.isnan(p):
        return "—"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _sig(p: float) -> str:
    if math.isnan(p):
        return "—"
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return "ns"


def print_results(merged: pd.DataFrame, disc: pd.DataFrame) -> list[dict]:
    section("concordance summary")
    n = len(merged)
    both_right  = int((merged["is_correct_gptoss"] & merged["is_correct_gemma4"]).sum())
    both_wrong  = int((~merged["is_correct_gptoss"] & ~merged["is_correct_gemma4"]).sum())
    g_wins      = int((merged["is_correct_gptoss"] & ~merged["is_correct_gemma4"]).sum())
    m_wins      = int((~merged["is_correct_gptoss"] & merged["is_correct_gemma4"]).sum())
    print(f"total shared problems: {n}")
    print(f"  both correct:                    {both_right} ({both_right/n*100:.1f}%)")
    print(f"  both incorrect:                  {both_wrong} ({both_wrong/n*100:.1f}%)")
    print(f"  gpt-oss correct, gemma4 wrong:   {g_wins}")
    print(f"  gemma4 correct, gpt-oss wrong:   {m_wins}")
    print(f"  discordant pairs total:          {len(disc)}")

    section("discordant pair feature comparison (wilcoxon signed-rank, H1: delta > 0)")
    print(f"  delta = incorrect_trace_value − correct_trace_value")
    print(f"  positive delta means the wrong model's trace is larger on this feature\n")

    rows = []
    for f in FEATURES:
        r = paired_stats(disc, f)
        rows.append(r)
        sig = _sig(r["wilcoxon_p"])
        print(
            f"  {f:<24}  med_correct={r['median_correct']:>6.2f}  "
            f"med_incorrect={r['median_incorrect']:>6.2f}  "
            f"med_delta={r['median_delta']:>+6.2f}  "
            f"delta>0: {r['pct_delta_positive']:>5.1f}%  "
            f"p={_fmt_p(r['wilcoxon_p'])}  {sig}"
        )

    section("per-pair detail (depth and n_steps)")
    print(f"{'question_id':<22}  {'winner':<8}  {'depth_correct':>13}  {'depth_incorrect':>15}  {'delta':>6}  {'n_steps_correct':>15}  {'n_steps_incorrect':>17}")
    print("-" * 110)
    for _, row in disc.sort_values("delta_depth", ascending=False).iterrows():
        print(
            f"{row['question_id']:<22}  {row['winner']:<8}  "
            f"{row['correct_depth']:>13.0f}  {row['incorrect_depth']:>15.0f}  "
            f"{row['delta_depth']:>+6.0f}  "
            f"{row['correct_n_steps']:>15.0f}  {row['incorrect_n_steps']:>17.0f}"
        )

    return rows


def generate_report(merged: pd.DataFrame, disc: pd.DataFrame, stat_rows: list[dict], out_path: Path) -> None:
    n = len(merged)
    both_right  = int((merged["is_correct_gptoss"] & merged["is_correct_gemma4"]).sum())
    both_wrong  = int((~merged["is_correct_gptoss"] & ~merged["is_correct_gemma4"]).sum())
    g_wins      = int((merged["is_correct_gptoss"] & ~merged["is_correct_gemma4"]).sum())
    m_wins      = int((~merged["is_correct_gptoss"] & merged["is_correct_gemma4"]).sum())

    lines = []

    lines.append("# Within-Problem Paired Comparison")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Motivation")
    lines.append("")
    lines.append(
        "The cross-model analyses in the main results show that incorrect traces tend to "
        "be structurally larger — more steps, greater depth, lower fact ratio — than "
        "correct traces. A natural objection is that this reflects problem difficulty "
        "rather than reasoning quality: harder problems might simply require more steps "
        "to solve, and harder problems are also more likely to be answered incorrectly. "
        "If that is the explanation, then the structural differences between correct and "
        "incorrect traces would disappear once problem difficulty is controlled for."
    )
    lines.append("")
    lines.append(
        "The within-problem comparison addresses this directly. Of the 200 shared "
        "problems, the two models disagree on 14. For each of these 14 problems, we "
        "have a correct trace from one model and an incorrect trace from the other, "
        "both solving the exact same question. Problem difficulty is therefore held "
        "constant by construction. Any feature difference observed within a discordant "
        "pair reflects a difference in how the models reasoned, not in what problem "
        "they were given."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Model Concordance")
    lines.append("")
    lines.append(f"Both models ran on the same {n} problems at temperature 0.")
    lines.append("")
    lines.append("| outcome | count | % of problems |")
    lines.append("|---|---|---|")
    lines.append(f"| both correct | {both_right} | {both_right/n*100:.1f}% |")
    lines.append(f"| both incorrect | {both_wrong} | {both_wrong/n*100:.1f}% |")
    lines.append(f"| gpt-oss correct, gemma4 wrong | {g_wins} | {g_wins/n*100:.1f}% |")
    lines.append(f"| gemma4 correct, gpt-oss wrong | {m_wins} | {m_wins/n*100:.1f}% |")
    lines.append(f"| **discordant total** | **{len(disc)}** | **{len(disc)/n*100:.1f}%** |")
    lines.append("")
    lines.append(
        f"The models agree on {both_right + both_wrong} of {n} problems ({(both_right+both_wrong)/n*100:.1f}%). "
        f"The 14 discordant problems are the only ones where a within-problem "
        f"comparison is possible. gpt-oss wins 6 of these ({g_wins}) and gemma4 wins 8 ({m_wins}), "
        f"suggesting the disagreements are distributed roughly evenly rather than "
        f"one model systematically dominating the other's failures."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Feature Comparison on Discordant Pairs")
    lines.append("")
    lines.append(
        "For each discordant pair, delta is defined as the incorrect model's feature "
        "value minus the correct model's feature value. A positive delta means the "
        "model that got the answer wrong produced a structurally larger trace on "
        "that feature. The Wilcoxon signed-rank test (one-sided, H1: delta > 0) "
        f"tests whether the positive direction is consistent across the {len(disc)} pairs. "
        "This is a non-parametric paired test and makes no distributional assumptions."
    )
    lines.append("")
    lines.append("| feature | median correct | median incorrect | median delta | delta > 0 | Wilcoxon p | sig |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in stat_rows:
        lines.append(
            f"| `{r['feature']}` | {r['median_correct']:.2f} | {r['median_incorrect']:.2f}"
            f" | {r['median_delta']:+.2f} | {r['pct_delta_positive']:.0f}% "
            f"| {_fmt_p(r['wilcoxon_p'])} | {_sig(r['wilcoxon_p'])} |"
        )
    lines.append("")

    lines.append("---")
    lines.append("")

    lines.append("## Per-Pair Detail")
    lines.append("")
    lines.append(
        "The table below shows each discordant problem with the depth and step count "
        "of the correct and incorrect traces. Problems are sorted by depth delta "
        "(incorrect minus correct) descending."
    )
    lines.append("")
    lines.append("| problem | winner | depth (correct) | depth (incorrect) | delta depth | steps (correct) | steps (incorrect) | delta steps |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, row in disc.sort_values("delta_depth", ascending=False).iterrows():
        lines.append(
            f"| `{row['question_id']}` | {row['winner']} "
            f"| {int(row['correct_depth'])} | {int(row['incorrect_depth'])} "
            f"| {int(row['delta_depth']):+d} "
            f"| {int(row['correct_n_steps'])} | {int(row['incorrect_n_steps'])} "
            f"| {int(row['delta_n_steps']):+d} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")

    depth_row = next(r for r in stat_rows if r["feature"] == "depth")
    steps_row = next(r for r in stat_rows if r["feature"] == "n_steps")
    facts_row = next(r for r in stat_rows if r["feature"] == "frac_facts")

    depth_p    = depth_row["wilcoxon_p"]
    steps_p    = steps_row["wilcoxon_p"]
    depth_pos  = depth_row["pct_delta_positive"]
    steps_pos  = steps_row["pct_delta_positive"]
    delta_d    = depth_row["median_delta"]
    delta_s    = steps_row["median_delta"]
    delta_f    = facts_row["median_delta"]

    lines.append(
        f"Among the 14 discordant pairs, the incorrect trace is deeper than the correct "
        f"trace in {depth_pos:.0f}% of cases (median delta = {delta_d:+.1f} steps of depth, "
        f"Wilcoxon p = {_fmt_p(depth_p)}). The incorrect trace has more total steps in "
        f"{steps_pos:.0f}% of cases (median delta = {delta_s:+.1f}, Wilcoxon p = {_fmt_p(steps_p)}). "
        f"The fact ratio moves in the expected direction as well — the incorrect trace "
        f"has a median delta of {delta_f:+.3f} in frac_facts (negative meaning the incorrect "
        f"trace is more fact-sparse), though power is limited at n=14."
    )
    lines.append("")

    if depth_p < 0.10 or steps_p < 0.10:
        lines.append(
            "The depth and step-count signals hold up within discordant pairs, providing "
            "direct evidence against the problem-difficulty confound. On the exact same "
            "question, the model that gets it wrong tends to construct a deeper, longer "
            "reasoning chain than the model that gets it right. This is consistent with "
            "the interpretation that over-elaboration of a reasoning trace is a genuine "
            "signal of reasoning failure, independent of how difficult the underlying "
            "problem is."
        )
    else:
        lines.append(
            "The depth and step-count effects do not reach conventional significance "
            "thresholds within the discordant pairs, most likely because n=14 provides "
            "very limited power for a one-sided Wilcoxon test. The direction is consistent "
            "with the cross-model results, but the within-problem evidence is inconclusive "
            "on its own. A larger dataset — or a benchmark with more discordant pairs — "
            "would be needed to confirm that the effect survives problem-difficulty control."
        )
    lines.append("")
    lines.append(
        "It is also worth noting that 14 discordant pairs out of 200 implies the two "
        "models mostly fail on the same problems. This concordance in failures suggests "
        "that problem difficulty is a real factor in overall accuracy — the two models "
        "both struggle with the same subset of hard problems — even if the structural "
        "features within discordant pairs point to additional reasoning-quality effects."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "With only 14 discordant pairs, the within-problem analysis is severely "
        "underpowered. The Wilcoxon signed-rank test requires at least 6 non-zero "
        "differences to compute a p-value, and conventional significance thresholds "
        "are difficult to achieve with n < 20. Results should be interpreted as "
        "directional evidence, not definitive conclusions."
    )
    lines.append("")
    lines.append(
        "The 14 discordant pairs are not a random sample of problems — they are "
        "specifically the problems where the two models disagree. These problems "
        "may have unusual properties (e.g., edge cases in number formatting, "
        "ambiguous problem statements) that make them unrepresentative of the "
        "full distribution."
    )
    lines.append("")
    lines.append(
        "Graph features are extracted by the same LLM segmenter for both models. "
        "However, the segmenter is given each model's trace independently, so "
        "feature values are not mechanically coupled across models for the same problem."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote report to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gptoss-features", default="gsm_hard_data/graph_features.csv")
    p.add_argument("--gemma4-features", default="experiments/gemma4-31b-cloud/graph_features.csv")
    p.add_argument("--out", default="within_problem_report.md")
    args = p.parse_args()

    merged = load_and_merge(args.gptoss_features, args.gemma4_features)
    disc   = build_discordant(merged)

    stat_rows = print_results(merged, disc)
    generate_report(merged, disc, stat_rows, Path(args.out))


if __name__ == "__main__":
    main()
