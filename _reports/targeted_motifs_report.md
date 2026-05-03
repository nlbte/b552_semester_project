# Targeted Motifs — Cross-Model Report

---

## Motivation

The five original motif flags (`has_unsupported_arith`, `has_orphan`, `has_unsupported_conclude`, `low_fact_ratio`, `long_chain`) were defined as generic structural irregularities. Most were not individually significant — only `long_chain` (depth > 5) held up in univariate tests, and `has_orphan` reached marginal significance in gpt-oss. The composite `motif_count` worked because bad patterns co-occur, not because each flag was individually diagnostic.

The three motifs tested here are designed to be more targeted. Each is grounded in a specific failure mechanism observed in the error analysis: arithmetic that has drifted away from the problem's given information, long traces that never anchor to the problem statement, and early over-branching before the reasoning has converged. Unlike the original flags, these are not just 'looks messy' indicators — each operationalizes a distinct hypothesis about how reasoning goes wrong.

---

## Motif Definitions

**`late_arithmetic`** — Any arithmetic node whose depth in the DAG (longest path from any source node) is >= 4, AND whose full ancestor set contains no `extract_fact` node. This captures arithmetic that has drifted far from the problem's given information.

**`verbose_ungrounded`** — `n_steps >= 8` AND `frac_facts < 0.3`. Captures long traces that still devote fewer than 30% of their steps to extracting quantities stated in the problem.

**`early_branching`** — Any node in the first half of the trace (by sequential step position, i.e., steps S1 through S_floor(n/2)) whose out-degree is >= 3. Captures a model that fans out into many parallel directions early, before the reasoning has converged.

---

## Per-Model Results

### gpt-oss:120b  (n=200, correct=159, incorrect=41)

| motif | n present | % of traces | correct | incorrect | Fisher OR | Fisher p | Logit OR | 95% CI | Logit p |
|---|---|---|---|---|---|---|---|---|---|
| `late_arithmetic` | 1 | 0.5% | 1 | 0 | 0.000 | 1.0000 | — | — | — |
| `verbose_ungrounded` | 33 | 16.5% | 15 | 18 | 7.513 | 1.55e-06 | 0.133 | [0.059, 0.301] | 1.21e-06 |
| `early_branching` | 27 | 13.5% | 18 | 9 | 2.203 | 0.0689 | 0.454 | [0.187, 1.102] | 0.0811 |

### gemma4:31b  (n=200, correct=161, incorrect=39)

| motif | n present | % of traces | correct | incorrect | Fisher OR | Fisher p | Logit OR | 95% CI | Logit p |
|---|---|---|---|---|---|---|---|---|---|
| `late_arithmetic` | 0 | 0.0% | 0 | 0 | — | 1.0000 | — | — | — |
| `verbose_ungrounded` | 34 | 17.0% | 21 | 13 | 3.333 | 0.0041 | 0.300 | [0.134, 0.673] | 0.0035 |
| `early_branching` | 30 | 15.0% | 23 | 7 | 1.312 | 0.3608 | 0.762 | [0.301, 1.930] | 0.5663 |

---

## Cross-Model Comparison

The table below shows whether each motif replicates across both models. Replication is defined as a consistent direction of effect (OR < 1 for correctness, i.e., motif predicts failure) and p < 0.10 in at least one test (Fisher or logit) in both models.

| motif | gpt-oss Fisher p | gpt-oss Logit OR | gemma4 Fisher p | gemma4 Logit OR | replicates |
|---|---|---|---|---|---|
| `late_arithmetic` | 1.0000 | — | 1.0000 | — | no |
| `verbose_ungrounded` | 1.55e-06 | 0.133 | 0.0041 | 0.300 | yes |
| `early_branching` | 0.0689 | 0.454 | 0.3608 | 0.762 | no |

---

## Combined Regressions

Two combined regressions are reported for each model. The first includes only the three new motifs to assess their joint predictive power. The second adds `depth` and `frac_facts` — the two strongest predictors from the original feature analysis — as covariates. A new motif adds independent signal if it remains significant after conditioning on depth and frac_facts.

Note that `verbose_ungrounded` is defined partly in terms of `frac_facts` (it fires when frac_facts < 0.3), so collinearity between these two predictors is expected in the second regression. Interpret `verbose_ungrounded` coefficients in the 5-predictor model with caution.

### gpt-oss:120b

**3-motif model** (pseudo-R² = 0.116)

| feature | coef | OR | 95% CI | p-value | sig |
|---|---|---|---|---|---|
| `const` | 1.834 | — | — | 1.16e-15 | *** |
| `verbose_ungrounded` | -2.018 | 0.133 | [0.055, 0.320] | 6.53e-06 | *** |
| `early_branching` | 0.005 | 1.005 | [0.351, 2.879] | 0.9918 |  |

**5-predictor model** (3 new motifs + depth + frac_facts, pseudo-R² = 0.134)

| feature | coef | OR | 95% CI | p-value | sig |
|---|---|---|---|---|---|
| `const` | 2.540 | — | — | 0.0386 | ** |
| `verbose_ungrounded` | -1.262 | 0.283 | [0.085, 0.945] | 0.0401 | ** |
| `depth` | -0.279 | 0.757 | [0.540, 1.060] | 0.1055 |  |
| `frac_facts` | 1.232 | 3.429 | [0.041, 286.903] | 0.5853 |  |
| `early_branching` | 0.125 | 1.133 | [0.388, 3.303] | 0.8194 |  |

### gemma4:31b

**3-motif model** (pseudo-R² = 0.041)

| feature | coef | OR | 95% CI | p-value | sig |
|---|---|---|---|---|---|
| `const` | 1.673 | — | — | 2.75e-14 | *** |
| `verbose_ungrounded` | -1.230 | 0.292 | [0.126, 0.681] | 0.0044 | *** |
| `early_branching` | 0.105 | 1.111 | [0.407, 3.028] | 0.8376 |  |

**5-predictor model** (3 new motifs + depth + frac_facts, pseudo-R² = 0.063)

| feature | coef | OR | 95% CI | p-value | sig |
|---|---|---|---|---|---|
| `const` | 3.278 | — | — | 0.0066 | *** |
| `depth` | -0.358 | 0.699 | [0.500, 0.977] | 0.0358 | ** |
| `verbose_ungrounded` | -0.758 | 0.469 | [0.155, 1.422] | 0.1810 |  |
| `early_branching` | 0.251 | 1.285 | [0.462, 3.576] | 0.6307 |  |
| `frac_facts` | -0.344 | 0.709 | [0.008, 61.504] | 0.8800 |  |

---

## Discussion

`late_arithmetic` shows a Fisher p of 1.0000 in gpt-oss and 1.0000 in gemma4. 
Neither model shows a significant effect for this motif. Either the hypothesis is wrong, the motif is too rarely triggered to have power, or the graph extractor does not reliably distinguish ungrounded arithmetic from grounded arithmetic in its op labeling.

`verbose_ungrounded` fires when n_steps >= 8 AND frac_facts < 0.3. It shows Fisher p = 1.55e-06 (gpt-oss) and 0.0041 (gemma4). 
This replicates across models. However, it is worth noting that `verbose_ungrounded` is a composite of two features that were already individually significant — `n_steps` (or `depth`) and `frac_facts`. The motif is therefore best understood as a convenient threshold rule rather than a new mechanistic signal. In the 5-predictor regression, its coefficient will absorb some of the variance already explained by `frac_facts`, so the independent contribution is likely smaller than the univariate result suggests.

`early_branching` shows Fisher p = 0.0689 (gpt-oss) and 0.3608 (gemma4). 
The motif is significant in one model. It is plausible that the two models have different branching styles — gpt-oss's more verbose traces may produce higher out-degrees at earlier steps, making the motif more detectable there.

In the 5-predictor regressions, any new motif that retains significance after conditioning on depth and frac_facts provides evidence of an independent structural signal. If none do, the result still has value: it narrows the explanation to those two features being sufficient predictors of failure within this dataset and sample size.

---

## Limitations

All motifs are operationally defined on LLM-segmented graphs. The graph extractor (gpt-oss:120b-cloud or claude-opus-4-6) decides which steps count as `extract_fact` vs. `arithmetic` vs. `other`, and how dependencies are drawn. If the extractor is inconsistent — labeling the same type of step differently across problems — then motif rates partly reflect the extractor's noise rather than the reasoning model's behavior. There is no ground truth for the graph structure, only the extractor's interpretation of the trace.

Both models were evaluated on n=200 problems with 39-41 incorrect traces. This gives limited power for detecting individually weak effects. Fisher's exact test and univariate logistic regression are both underpowered when the motif is rare (fewer than ~15-20 positive cases). Motifs that fire in fewer than 10% of traces should be interpreted with extra caution.

The combined regressions use unstandardized predictors, so coefficient magnitudes are not directly comparable across features. `frac_facts` ranges from 0 to 1 while `depth` and the binary motif flags have very different scales. The odds ratios are interpretable but the coefficients in the combined model should not be rank-ordered as importance scores.
