"""Fisher's exact + logistic regression with odds ratios.

Usage:
    python stats_tests.py
    python stats_tests.py --data-dir experiments/gemma4-31b-cloud
"""

import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="gsm_hard_data")
args = ap.parse_args()
data_dir = Path(args.data_dir)

df = pd.read_csv(data_dir / "motif_scores.csv")

# fisher's exact: motif_count >= 2 vs correctness
high  = df[df["motif_count"] >= 2]
low   = df[df["motif_count"] <  2]
table = [
    [int((~high["is_correct"]).sum()), int(high["is_correct"].sum())],
    [int((~low["is_correct"]).sum()),  int(low["is_correct"].sum())],
]
odds_ratio, p_fisher = stats.fisher_exact(table, alternative="greater")
print("── fisher's exact: motif_count ≥ 2 predicts incorrectness ──")
print(f"  contingency table:  {table}")
print(f"  odds ratio: {odds_ratio:.2f}   p = {p_fisher:.4f}")

# logistic regression with odds ratios
features = ["depth", "frac_facts", "orphan_nodes", "max_out_degree",
            "unsupported_arithmetic", "unsupported_conclude"]

# drop zero-variance features
features = [f for f in features if df[f].std() > 0]

X = sm.add_constant(df[features].astype(float))
y = df["is_correct"].astype(int)

result = sm.Logit(y, X).fit(disp=0, maxiter=200)

out = pd.DataFrame({
    "feature":    ["const"] + features,
    "OR":         np.exp(result.params).round(3),
    "coef":       result.params.round(3),
    "p_value":    result.pvalues.round(4),
    "sig":        ["***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                   for p in result.pvalues],
}).sort_values("p_value")

print(f"\n── logistic regression (pseudo-R² = {result.prsquared:.3f}) ──")
print(out.to_string(index=False))
out.to_csv(data_dir / "logistic_odds_ratios.csv", index=False)
print(f"\nwrote {data_dir}/logistic_odds_ratios.csv")
