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


# merge both models' feature csvs on question_id
def load_and_merge(path_g, path_m):
    g = pd.read_csv(path_g)
    m = pd.read_csv(path_m)
    g["is_correct"] = g["is_correct"].astype(bool)
    m["is_correct"] = m["is_correct"].astype(bool)
    return g.merge(m, on="question_id", suffixes=("_gptoss", "_gemma4"))


# pull only rows where models disagree, label winner/loser per pair
def build_discordant(merged):
    disc = merged[merged["is_correct_gptoss"] != merged["is_correct_gemma4"]].copy()
    rows = []
    for _, row in disc.iterrows():
        winner, loser = ("gptoss", "gemma4") if row["is_correct_gptoss"] else ("gemma4", "gptoss")
        r = {"question_id": row["question_id"], "winner": winner, "loser": loser}
        for f in FEATURES:
            r[f"correct_{f}"]   = row[f"{f}_{winner}"]
            r[f"incorrect_{f}"] = row[f"{f}_{loser}"]
            r[f"delta_{f}"]     = row[f"{f}_{loser}"] - row[f"{f}_{winner}"]
        rows.append(r)
    return pd.DataFrame(rows)


# one-sided wilcoxon signed-rank test on a delta array
def wilcoxon_one_sided(delta, alternative="greater"):
    clean = delta[~np.isnan(delta)]
    if len(clean) < 4 or np.all(clean == 0):
        return float("nan"), 1.0
    try:
        stat, p = stats.wilcoxon(clean, alternative=alternative)
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


# compute summary stats and wilcoxon p for one feature across all discordant pairs
def paired_stats(disc, feature):
    delta     = disc[f"delta_{feature}"].values.astype(float)
    correct   = disc[f"correct_{feature}"].values.astype(float)
    incorrect = disc[f"incorrect_{feature}"].values.astype(float)
    stat, p   = wilcoxon_one_sided(delta)
    return {
        "feature":            feature,
        "median_correct":     float(np.nanmedian(correct)),
        "median_incorrect":   float(np.nanmedian(incorrect)),
        "median_delta":       float(np.nanmedian(delta)),
        "pct_delta_positive": float(np.mean(delta > 0)) * 100,
        "wilcoxon_p":         p,
    }


def _fmt_p(p):
    if math.isnan(p): return "—"
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"


def _sig(p):
    if math.isnan(p): return "—"
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return "ns"


# print concordance summary, per-feature wilcoxon results, and per-pair depth/steps table
def print_results(merged, disc):
    n          = len(merged)
    both_right = int((merged["is_correct_gptoss"] & merged["is_correct_gemma4"]).sum())
    both_wrong = int((~merged["is_correct_gptoss"] & ~merged["is_correct_gemma4"]).sum())
    g_wins     = int((merged["is_correct_gptoss"] & ~merged["is_correct_gemma4"]).sum())
    m_wins     = int((~merged["is_correct_gptoss"] & merged["is_correct_gemma4"]).sum())

    print(f"\nconcordance ({n} problems)")
    print(f"  both correct: {both_right}  both wrong: {both_wrong}  gptoss wins: {g_wins}  gemma4 wins: {m_wins}  discordant: {len(disc)}")

    print(f"\nfeature deltas on {len(disc)} discordant pairs (wilcoxon H1: delta > 0)")
    rows = []
    for f in FEATURES:
        r = paired_stats(disc, f)
        rows.append(r)
        print(
            f"  {f:<24}  med_correct={r['median_correct']:>5.2f}  "
            f"med_incorrect={r['median_incorrect']:>5.2f}  "
            f"delta={r['median_delta']:>+5.2f}  "
            f"delta>0: {r['pct_delta_positive']:>4.0f}%  "
            f"p={_fmt_p(r['wilcoxon_p'])}  {_sig(r['wilcoxon_p'])}"
        )

    print("\nper-pair depth / n_steps")
    for _, row in disc.sort_values("delta_depth", ascending=False).iterrows():
        print(
            f"  {row['question_id']:<22}  winner={row['winner']:<8}  "
            f"depth {row['correct_depth']:.0f}->{row['incorrect_depth']:.0f} ({row['delta_depth']:+.0f})  "
            f"steps {row['correct_n_steps']:.0f}->{row['incorrect_n_steps']:.0f} ({row['delta_n_steps']:+.0f})"
        )

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gptoss-features", default="results_gptoss/graph_features.csv")
    p.add_argument("--gemma4-features", default="results_gemma4/graph_features.csv")
    args = p.parse_args()

    merged = load_and_merge(args.gptoss_features, args.gemma4_features)
    disc   = build_discordant(merged)
    print_results(merged, disc)


if __name__ == "__main__":
    main()
