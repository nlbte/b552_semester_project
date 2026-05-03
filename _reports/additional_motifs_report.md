# Additional Motifs — Cross-Model Report

---

## Motivation

The five original motifs and three targeted motifs tested previously were grounded in generic structural irregularities and specific hypotheses about arithmetic drift, verbose ungrounded reasoning, and early branching. This analysis tests five additional structural flags: off-schema node labels, extract_fact nodes that are never consumed, purely linear reasoning chains, high fan-in conclude nodes, and the presence of explicit verification steps. The last motif (`failed_verification`) is diagnostic rather than predictive — it tests whether models that attempt to verify their answers actually do so on harder problems, regardless of whether verification improves accuracy.

---

## Motif Definitions

**`has_offschema_node`** — Any step whose `op` label is not in `{extract_fact, arithmetic, substitute, conclude, verify}`. Captures traces where the LLM segmenter (or the reasoning model) produced steps that do not fit the expected reasoning schema.

**`orphan_fact`** — Any `extract_fact` node with out-degree 0 (the fact was extracted but never used in any downstream step). Captures dead-end information gathering.

**`linear_chain`** — `n_steps >= 7` AND `mean_out_degree <= 1.1` AND `max_out_degree <= 2`. Captures a trace that proceeds in an almost purely sequential manner with minimal branching.

**`high_fanin_conclude`** — Any `conclude` node with in-degree >= 3. Captures a conclusion that synthesizes three or more upstream steps, which may indicate a high-integration step or a poorly structured convergence.

**`failed_verification`** — The trace contains at least one `verify` step. Despite the name, this is diagnostic: it measures whether the model attempted verification, not whether verification succeeded. If verification steps appear more in incorrect traces, this suggests the model uses verification as a recovery attempt on problems it is already struggling with.

---

## Per-Model Results

### gpt-oss:120b  (n=200, correct=159, incorrect=41)

| motif | n present | % of traces | correct | incorrect | Fisher OR | Fisher p | Logit OR | 95% CI | Logit p |
|---|---|---|---|---|---|---|---|---|---|
| `has_offschema_node` | 34 | 17.0% | 13 | 21 | 11.792 | 3.97e-09 | 0.085 | [0.037, 0.195] | 6.89e-09 |
| `orphan_fact` | 3 | 1.5% | 3 | 0 | 0.000 | 1.0000 | — | — | — |
| `linear_chain` | 86 | 43.0% | 70 | 16 | 0.814 | 0.7737 | 1.229 | [0.609, 2.478] | 0.5645 |
| `high_fanin_conclude` | 5 | 2.5% | 2 | 3 | 6.197 | 0.0595 | 0.161 | [0.026, 1.000] | 0.0500 |
| `failed_verification` | 52 | 26.0% | 36 | 16 | 2.187 | 0.0291 | 0.457 | [0.221, 0.948] | 0.0355 |

### gemma4:31b  (n=200, correct=161, incorrect=39)

| motif | n present | % of traces | correct | incorrect | Fisher OR | Fisher p | Logit OR | 95% CI | Logit p |
|---|---|---|---|---|---|---|---|---|---|
| `has_offschema_node` | 28 | 14.0% | 10 | 18 | 12.943 | 1.34e-08 | 0.077 | [0.031, 0.190] | 2.27e-08 |
| `orphan_fact` | 3 | 1.5% | 3 | 0 | 0.000 | 1.0000 | — | — | — |
| `linear_chain` | 89 | 44.5% | 70 | 19 | 1.235 | 0.3394 | 0.810 | [0.402, 1.632] | 0.5551 |
| `high_fanin_conclude` | 1 | 0.5% | 0 | 1 | inf | 0.1950 | — | — | — |
| `failed_verification` | 32 | 16.0% | 24 | 8 | 1.473 | 0.2631 | 0.679 | [0.279, 1.653] | 0.3936 |

---

## Cross-Model Comparison

Replication is defined as consistent OR direction and p < 0.10 in both models (Fisher or logit). `failed_verification` is excluded from the replication criterion because it is diagnostic rather than predictive.

| motif | gpt-oss Fisher p | gpt-oss Logit OR | gemma4 Fisher p | gemma4 Logit OR | replicates |
|---|---|---|---|---|---|
| `has_offschema_node` | 3.97e-09 | 0.085 | 1.34e-08 | 0.077 | yes |
| `orphan_fact` | 1.0000 | — | 1.0000 | — | no |
| `linear_chain` | 0.7737 | 1.229 | 0.3394 | 0.810 | no |
| `high_fanin_conclude` | 0.0595 | 0.161 | 0.1950 | — | no |
| `failed_verification` | 0.0291 | 0.457 | 0.2631 | 0.679 | diagnostic |

---

## Combined Regressions

Two combined regressions are reported for each model. The first includes only the five new motifs. The second adds `depth`, `n_steps`, `frac_facts`, and the `linear_chain_x_nsteps` interaction term. The interaction term tests whether linear chains are only harmful (or helpful) in combination with longer traces. A motif that retains significance in the extended model provides evidence of an independent structural signal beyond the basic trace-length and fact-ratio features.

### gpt-oss:120b

**5-motif model** (pseudo-R² = 0.204)

| feature | coef | OR | 95% CI | p-value | sig |
|---|---|---|---|---|---|
| `const` | 2.208 | — | — | 7.24e-11 | *** |
| `has_offschema_node` | -2.566 | 0.077 | [0.032, 0.186] | 1.39e-08 | *** |
| `failed_verification` | -0.970 | 0.379 | [0.157, 0.917] | 0.0315 | ** |
| `linear_chain` | 0.266 | 1.305 | [0.577, 2.949] | 0.5225 |  |
| `high_fanin_conclude` | -0.214 | 0.807 | [0.079, 8.298] | 0.8570 |  |

**Extended model** (5 new motifs + depth + n_steps + frac_facts + interaction, pseudo-R² = 0.321)

| feature | coef | OR | 95% CI | p-value | sig |
|---|---|---|---|---|---|
| `has_offschema_node` | -2.497 | 0.082 | [0.030, 0.225] | 1.18e-06 | *** |
| `n_steps` | -0.469 | 0.626 | [0.446, 0.879] | 0.0068 | *** |
| `frac_facts` | 5.327 | 205.912 | [1.149, 36905.814] | 0.0442 | ** |
| `const` | 2.892 | — | — | 0.0585 | * |
| `failed_verification` | -0.762 | 0.467 | [0.170, 1.283] | 0.1397 |  |
| `depth` | 0.300 | 1.349 | [0.801, 2.275] | 0.2607 |  |
| `linear_chain` | 1.593 | 4.918 | [0.089, 270.653] | 0.4360 |  |
| `linear_chain_x_nsteps` | -0.156 | 0.856 | [0.545, 1.342] | 0.4971 |  |
| `high_fanin_conclude` | -0.046 | 0.955 | [0.068, 13.503] | 0.9731 |  |

### gemma4:31b

**5-motif model** (pseudo-R² = 0.173)

| feature | coef | OR | 95% CI | p-value | sig |
|---|---|---|---|---|---|
| `const` | 2.198 | — | — | 6.74e-11 | *** |
| `has_offschema_node` | -2.566 | 0.077 | [0.031, 0.190] | 2.73e-08 | *** |
| `linear_chain` | -0.352 | 0.703 | [0.314, 1.574] | 0.3914 |  |
| `failed_verification` | -0.350 | 0.705 | [0.249, 1.991] | 0.5088 |  |

**Extended model** (5 new motifs + depth + n_steps + frac_facts + interaction, pseudo-R² = 0.278)

| feature | coef | OR | 95% CI | p-value | sig |
|---|---|---|---|---|---|
| `has_offschema_node` | -2.496 | 0.082 | [0.029, 0.237] | 3.64e-06 | *** |
| `n_steps` | -0.734 | 0.480 | [0.317, 0.727] | 5.24e-04 | *** |
| `const` | 4.515 | — | — | 0.0130 | ** |
| `frac_facts` | 4.332 | 76.110 | [0.556, 10412.618] | 0.0843 | * |
| `depth` | 0.442 | 1.556 | [0.923, 2.624] | 0.0971 | * |
| `linear_chain` | -3.638 | 0.026 | [0.000, 2.501] | 0.1175 |  |
| `linear_chain_x_nsteps` | 0.415 | 1.515 | [0.882, 2.602] | 0.1321 |  |
| `failed_verification` | 0.386 | 1.471 | [0.421, 5.146] | 0.5455 |  |

---

## Discussion

`has_offschema_node` captures traces with steps whose `op` label falls outside the five-category schema. This flag merges two conceptually different phenomena: steps labeled `other` (the segmenter could not classify the step) and steps labeled `substitute` (a schema-legal but infrequent operation). If the flag is significant, it may reflect the segmenter's uncertainty rather than the reasoning model's behavior. However, if `substitute` steps appear disproportionately in incorrect traces, that would suggest a different failure mode.

`orphan_fact` fires when a fact is extracted but never referenced downstream. With only ~3 occurrences per model in the probe data, this motif is expected to hit the min-count guard and logistic regression will not be run. Fisher's exact test will be reported but should be interpreted with extreme caution given the rarity.

`linear_chain` flags traces with `n_steps >= 7`, `mean_out_degree <= 1.1`, and `max_out_degree <= 2`. These traces proceed in an almost purely sequential manner. The probe showed this fires in ~43% of traces with roughly equal rates in correct and incorrect traces, suggesting linear chains are not individually diagnostic of failure. The `linear_chain_x_nsteps` interaction in the extended regression tests whether long linear chains specifically (not just linear chains generally) predict failure.

`high_fanin_conclude` requires a conclude node with in-degree >= 3. The threshold was set at 3 (rather than 4) because the probe showed the maximum observed conclude in-degree across both datasets is 3 — a threshold of 4 would produce zero positive cases. With in-degree >= 3, the motif fires in approximately 5 traces in gpt-oss and 1 in gemma4, so the min-count guard will likely apply for gemma4.

`failed_verification` is the diagnostic motif. The probe showed verify steps appear in approximately 39% of incorrect gpt-oss traces vs. 23% of correct traces, and 21% of incorrect gemma4 traces vs. 15% of correct traces. This reverses the expected direction: if verification were a positive reasoning strategy, we would expect it to appear more in correct traces. The observed pattern instead suggests that models invoke verification as a compensatory behavior on problems they are already struggling with. This is consistent with the interpretation that explicit self-checking in chain-of-thought is triggered by difficulty, not by rigor.

---

## Limitations

The `has_offschema_node` flag mixes `other` and `substitute` steps. If `substitute` steps are schema-legal and mechanistically distinct from `other` steps, combining them into one binary flag may dilute the signal from truly off-schema behavior. Separate flags for `other` vs. `substitute` would require additional schema redesign.

`orphan_fact` and `high_fanin_conclude` (for gemma4) will trigger the min-count guard and logistic regression results will not be computed. Fisher's exact test will still be reported but its power is severely limited at n < 5 positive cases.

The `failed_verification` result should not be interpreted as 'verification causes failure.' The causality likely runs the other way: hard problems elicit verification attempts. Disentangling this would require controlling for problem difficulty, which the current dataset does not support at scale (see the within-problem analysis, which has only 14 discordant pairs).
