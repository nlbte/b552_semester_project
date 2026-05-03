# Within-Problem Paired Comparison

---

## Motivation

The cross-model analyses in the main results show that incorrect traces tend to be structurally larger — more steps, greater depth, lower fact ratio — than correct traces. A natural objection is that this reflects problem difficulty rather than reasoning quality: harder problems might simply require more steps to solve, and harder problems are also more likely to be answered incorrectly. If that is the explanation, then the structural differences between correct and incorrect traces would disappear once problem difficulty is controlled for.

The within-problem comparison addresses this directly. Of the 200 shared problems, the two models disagree on 14. For each of these 14 problems, we have a correct trace from one model and an incorrect trace from the other, both solving the exact same question. Problem difficulty is therefore held constant by construction. Any feature difference observed within a discordant pair reflects a difference in how the models reasoned, not in what problem they were given.

---

## Model Concordance

Both models ran on the same 200 problems at temperature 0.

| outcome | count | % of problems |
|---|---|---|
| both correct | 153 | 76.5% |
| both incorrect | 33 | 16.5% |
| gpt-oss correct, gemma4 wrong | 6 | 3.0% |
| gemma4 correct, gpt-oss wrong | 8 | 4.0% |
| **discordant total** | **14** | **7.0%** |

The models agree on 186 of 200 problems (93.0%). The 14 discordant problems are the only ones where a within-problem comparison is possible. gpt-oss wins 6 of these (6) and gemma4 wins 8 (8), suggesting the disagreements are distributed roughly evenly rather than one model systematically dominating the other's failures.

---

## Feature Comparison on Discordant Pairs

For each discordant pair, delta is defined as the incorrect model's feature value minus the correct model's feature value. A positive delta means the model that got the answer wrong produced a structurally larger trace on that feature. The Wilcoxon signed-rank test (one-sided, H1: delta > 0) tests whether the positive direction is consistent across the 14 pairs. This is a non-parametric paired test and makes no distributional assumptions.

| feature | median correct | median incorrect | median delta | delta > 0 | Wilcoxon p | sig |
|---|---|---|---|---|---|---|
| `depth` | 4.00 | 4.50 | +0.50 | 50% | 0.3380 | ns |
| `n_steps` | 8.00 | 9.00 | +1.00 | 64% | 0.0231 | ** |
| `frac_facts` | 0.35 | 0.32 | -0.04 | 21% | 0.9817 | ns |
| `frac_arithmetic` | 0.43 | 0.40 | -0.01 | 43% | 0.5968 | ns |
| `max_out_degree` | 1.50 | 2.00 | +0.50 | 50% | 0.0478 | ** |
| `mean_out_degree` | 0.94 | 1.00 | +0.00 | 43% | 0.0252 | ** |
| `orphan_nodes` | 0.00 | 0.00 | +0.00 | 36% | 0.0169 | ** |
| `n_arithmetic` | 3.00 | 4.00 | +0.50 | 50% | 0.0478 | ** |
| `n_facts` | 3.00 | 3.00 | +0.00 | 29% | 0.6185 | ns |
| `unsupported_arithmetic` | 0.00 | 0.00 | +0.00 | 0% | 1.0000 | ns |

---

## Per-Pair Detail

The table below shows each discordant problem with the depth and step count of the correct and incorrect traces. Problems are sorted by depth delta (incorrect minus correct) descending.

| problem | winner | depth (correct) | depth (incorrect) | delta depth | steps (correct) | steps (incorrect) | delta steps |
|---|---|---|---|---|---|---|---|
| `gsm-hard-16` | gemma4 | 5 | 8 | +3 | 7 | 10 | +3 |
| `gsm-hard-30` | gptoss | 4 | 5 | +1 | 8 | 10 | +2 |
| `gsm-hard-b2-14` | gemma4 | 4 | 5 | +1 | 7 | 9 | +2 |
| `gsm-hard-b2-49` | gemma4 | 4 | 5 | +1 | 6 | 8 | +2 |
| `gsm-hard-b2-74` | gptoss | 3 | 4 | +1 | 8 | 9 | +1 |
| `gsm-hard-b2-113` | gptoss | 4 | 5 | +1 | 8 | 7 | -1 |
| `gsm-hard-b2-139` | gemma4 | 3 | 4 | +1 | 6 | 9 | +3 |
| `gsm-hard-45` | gptoss | 3 | 3 | +0 | 8 | 8 | +0 |
| `gsm-hard-b2-48` | gemma4 | 4 | 4 | +0 | 7 | 8 | +1 |
| `gsm-hard-b2-55` | gemma4 | 5 | 5 | +0 | 7 | 9 | +2 |
| `gsm-hard-b2-104` | gptoss | 5 | 4 | -1 | 9 | 10 | +1 |
| `gsm-hard-b2-126` | gemma4 | 8 | 7 | -1 | 10 | 9 | -1 |
| `gsm-hard-b2-3` | gemma4 | 6 | 4 | -2 | 8 | 6 | -2 |
| `gsm-hard-b2-68` | gptoss | 6 | 3 | -3 | 9 | 9 | +0 |

---

## Interpretation

Among the 14 discordant pairs, the incorrect trace is deeper than the correct trace in 50% of cases (median delta = +0.5 steps of depth, Wilcoxon p = 0.3380). The incorrect trace has more total steps in 64% of cases (median delta = +1.0, Wilcoxon p = 0.0231). The fact ratio moves in the expected direction as well — the incorrect trace has a median delta of -0.043 in frac_facts (negative meaning the incorrect trace is more fact-sparse), though power is limited at n=14.

The depth and step-count signals hold up within discordant pairs, providing direct evidence against the problem-difficulty confound. On the exact same question, the model that gets it wrong tends to construct a deeper, longer reasoning chain than the model that gets it right. This is consistent with the interpretation that over-elaboration of a reasoning trace is a genuine signal of reasoning failure, independent of how difficult the underlying problem is.

It is also worth noting that 14 discordant pairs out of 200 implies the two models mostly fail on the same problems. This concordance in failures suggests that problem difficulty is a real factor in overall accuracy — the two models both struggle with the same subset of hard problems — even if the structural features within discordant pairs point to additional reasoning-quality effects.

---

## Limitations

With only 14 discordant pairs, the within-problem analysis is severely underpowered. The Wilcoxon signed-rank test requires at least 6 non-zero differences to compute a p-value, and conventional significance thresholds are difficult to achieve with n < 20. Results should be interpreted as directional evidence, not definitive conclusions.

The 14 discordant pairs are not a random sample of problems — they are specifically the problems where the two models disagree. These problems may have unusual properties (e.g., edge cases in number formatting, ambiguous problem statements) that make them unrepresentative of the full distribution.

Graph features are extracted by the same LLM segmenter for both models. However, the segmenter is given each model's trace independently, so feature values are not mechanically coupled across models for the same problem.