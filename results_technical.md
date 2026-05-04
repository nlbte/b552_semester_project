# B552 Semester Project — Technical Results Report


## Overall Accuracy

| Model | Correct | Incorrect | Total | Accuracy |
|---|---|---|---|---|
| `gpt-oss:120b` | 159 | 41 | 200 | **79.5%** |
| `gemma4:31b` | 161 | 39 | 200 | **80.5%** |

Both models perform nearly identically on GSM-Hard at temperature 0. The ~1% gap is not meaningful given the sample size. This near-parity makes structural comparison more interesting — the models fail on different problems (and for potentially different structural reasons).

---

## Graph Feature Comparison: Correct vs. Incorrect

14 structural features were extracted per graph. Mann-Whitney U tests compared correct vs. incorrect traces within each model. Results sorted by p-value.

### gpt-oss:120b (n=159 correct, 41 incorrect)

| Feature | Correct Mean | Incorrect Mean | Correct Median | Incorrect Median | p-value | Sig |
|---|---|---|---|---|---|---|
| `n_steps` | 7.49 | 9.54 | 7.0 | 9.0 | 5.3e-07 | *** |
| `depth` | 4.24 | 5.37 | 4.0 | 5.0 | 5.3e-06 | *** |
| `mean_out_degree` | 0.990 | 1.099 | 1.0 | 1.08 | 1.7e-04 | *** |
| `mean_in_degree` | 0.990 | 1.099 | 1.0 | 1.08 | 1.7e-04 | *** |
| `frac_facts` | 0.359 | 0.287 | 0.375 | 0.273 | 2.9e-04 | *** |
| `max_out_degree` | 1.711 | 2.146 | 2.0 | 2.0 | 1.2e-03 | ** |
| `n_arithmetic` | 3.377 | 4.585 | 3.0 | 4.0 | 2.7e-03 | ** |
| `orphan_nodes` | 0.088 | 0.244 | 0.0 | 0.0 | 0.043 | * |
| `conclude_in_degree` | 1.170 | 1.317 | 1.0 | 1.0 | 0.156 | — |
| `unsupported_arithmetic` | 0.050 | 0.000 | 0.0 | 0.0 | 0.174 | — |
| `frac_arithmetic` | 0.446 | 0.457 | 0.429 | 0.500 | 0.386 | — |
| `n_facts` | 2.673 | 2.683 | 3.0 | 3.0 | 0.660 | — |
| `n_conclude` | 1.000 | 1.000 | 1.0 | 1.0 | 1.000 | — |
| `unsupported_conclude` | 0.000 | 0.000 | 0.0 | 0.0 | 1.000 | — |

### gemma4:31b (n=161 correct, 39 incorrect)

| Feature | Correct Mean | Incorrect Mean | Correct Median | Incorrect Median | p-value | Sig |
|---|---|---|---|---|---|---|
| `n_steps` | 7.32 | 9.03 | 7.0 | 9.0 | 2.6e-05 | *** |
| `depth` | 4.21 | 4.92 | 4.0 | 5.0 | 1.6e-03 | ** |
| `n_arithmetic` | 3.51 | 4.26 | 3.0 | 4.0 | 0.023 | * |
| `mean_out_degree` | 1.001 | 1.060 | 1.0 | 1.0 | 0.033 | * |
| `mean_in_degree` | 1.001 | 1.060 | 1.0 | 1.0 | 0.033 | * |
| `frac_facts` | 0.341 | 0.308 | 0.364 | 0.286 | 0.037 | * |
| `n_conclude` | 1.000 | 1.026 | 1.0 | 1.0 | 0.043 | * |
| `max_out_degree` | 1.758 | 2.179 | 2.0 | 2.0 | 0.058 | ~ |
| `orphan_nodes` | 0.062 | 0.128 | 0.0 | 0.0 | 0.162 | — |
| `n_facts` | 2.497 | 2.744 | 3.0 | 3.0 | 0.323 | — |
| `unsupported_arithmetic` | 0.043 | 0.103 | 0.0 | 0.0 | 0.384 | — |
| `frac_arithmetic` | 0.478 | 0.462 | 0.444 | 0.444 | 0.658 | — |
| `conclude_in_degree` | 1.130 | 1.179 | 1.0 | 1.0 | 0.664 | — |
| `unsupported_conclude` | 0.000 | 0.000 | 0.0 | 0.0 | 1.000 | — |

### Cross-Model Feature Comparison

| Feature | gpt-oss correct | gpt-oss incorrect | gemma4 correct | gemma4 incorrect |
|---|---|---|---|---|
| `n_steps` (mean) | 7.49 | 9.54 | 7.32 | 9.03 |
| `depth` (mean) | 4.24 | 5.37 | 4.21 | 4.92 |
| `frac_facts` (mean) | 0.359 | 0.287 | 0.341 | 0.308 |
| `orphan_nodes` (mean) | 0.088 | 0.244 | 0.062 | 0.128 |
| `max_out_degree` (mean) | 1.711 | 2.146 | 1.758 | 2.179 |

The patterns are **strikingly consistent across models**. Both models show the same direction and rough magnitude of difference on every significant feature. This suggests the structural signals are real properties of correct vs. incorrect mathematical reasoning, not artifacts of any specific model's style.

---

## Logistic Regression (Standardized Features)

Predicting `is_correct` from graph features. Redundant features (highly correlated pairs: `n_steps`/`depth`, `mean_in_degree`/`mean_out_degree`) were dropped before fitting.

### gpt-oss:120b

| Feature | Coefficient | p-value | Sig |
|---|---|---|---|
| `const` | +1.634 | 2.8e-14 | *** |
| `depth` | **-0.639** | **0.007** | *** |
| `frac_arithmetic` | +0.786 | 0.012 | ** |
| `frac_facts` | +0.776 | 0.016 | ** |
| `mean_out_degree` | -0.432 | 0.190 | — |
| `orphan_nodes` | -0.240 | 0.226 | — |
| `conclude_in_degree` | +0.172 | 0.445 | — |
| `max_out_degree` | +0.130 | 0.699 | — |

### gemma4:31b

| Feature | Coefficient | p-value | Sig |
|---|---|---|---|
| `const` | +1.558 | 3.7e-15 | *** |
| `frac_arithmetic` | +0.670 | 0.014 | ** |
| `frac_facts` | +0.535 | 0.069 | * |
| `depth` | **-0.346** | **0.093** | * |
| `max_out_degree` | -0.685 | 0.096 | * |
| `mean_out_degree` | +0.408 | 0.337 | — |
| `conclude_in_degree` | +0.114 | 0.611 | — |
| `orphan_nodes` | -0.018 | 0.923 | — |

Both models agree on the top predictors: `depth` is the strongest negative predictor (more depth = worse), while `frac_arithmetic` and `frac_facts` are positive predictors (more focused, factual reasoning = better). The effect is stronger and more significant for gpt-oss, likely because gpt-oss produces more verbose traces, making structural differences more legible.

---

## Logistic Regression with Odds Ratios

Raw (unstandardized) features using Fisher's exact + logistic regression with odds ratios.

### gpt-oss:120b

| Feature | Odds Ratio | p-value | Sig |
|---|---|---|---|
| `depth` | **0.673** | 0.015 | ** |
| `const` | 10.908 | 0.074 | * |
| `orphan_nodes` | **0.457** | 0.097 | * |
| `frac_facts` | 30.603 | 0.113 | — |
| `max_out_degree` | 0.894 | 0.684 | — |

`depth` OR = 0.673: each additional step of reasoning depth reduces odds of being correct by ~33%. `orphan_nodes` OR = 0.457: each orphaned (disconnected) reasoning node cuts the odds of correctness roughly in half.

### gemma4:31b

| Feature | Odds Ratio | p-value | Sig |
|---|---|---|---|
| `const` | **61.43** | 0.003 | *** |
| `depth` | **0.664** | 0.013 | ** |
| `max_out_degree` | 0.762 | 0.208 | — |
| `unsupported_arithmetic` | 0.524 | 0.214 | — |
| `orphan_nodes` | 0.611 | 0.436 | — |
| `frac_facts` | 0.499 | 0.756 | — |

`depth` OR = 0.664 for gemma4 — almost identical to gpt-oss (0.673). This cross-model replication of the depth effect is the strongest finding in the study.

---

## Motif Flags

Five binary "bad pattern" flags were defined and summed into a composite `motif_count`:

| Flag | Definition |
|---|---|
| `has_unsupported_arith` | Any arithmetic node with no incoming edges (unsupported computation) |
| `has_orphan` | Any non-conclude node with out-degree = 0 (dead-end reasoning step) |
| `has_unsupported_conclude` | Conclude node with no incoming edges (conclusion from nowhere) |
| `low_fact_ratio` | `frac_facts` < 0.25 (model spends less than 25% of steps extracting given facts) |
| `long_chain` | `depth` > 5 (reasoning chain is long) |

Higher motif counts predict failure on both models. `has_unsupported_conclude` is essentially zero across all 200 traces for both models — the models always chain their conclusions to prior steps, even when those steps are wrong.

---

## Targeted Motifs

Three hypothesis-driven binary flags were tested, each operationalizing a specific failure mechanism observed in the error analysis.

### Definitions

| Flag | Definition |
|---|---|
| `late_arithmetic` | Any arithmetic node at depth >= 4 whose full ancestor set contains no `extract_fact` node — arithmetic that has drifted from the problem's given information |
| `verbose_ungrounded` | `n_steps >= 8` AND `frac_facts < 0.3` — a long trace that still devotes fewer than 30% of steps to extracting stated quantities |
| `early_branching` | Any node in the first half of the trace (steps S1–S_floor(n/2)) with out-degree >= 3 — premature fan-out before reasoning has converged |

### Per-Model Results

#### gpt-oss:120b (n=200, correct=159, incorrect=41)

| motif | n present | % of traces | correct | incorrect | Fisher OR | Fisher p | Logit OR | 95% CI | Logit p |
|---|---|---|---|---|---|---|---|---|---|
| `late_arithmetic` | 1 | 0.5% | 1 | 0 | 0.000 | 1.0000 | — | — | — |
| `verbose_ungrounded` | 33 | 16.5% | 15 | 18 | 7.513 | 1.55e-06 | 0.133 | [0.059, 0.301] | 1.21e-06 |
| `early_branching` | 27 | 13.5% | 18 | 9 | 2.203 | 0.0689 | 0.454 | [0.187, 1.102] | 0.0811 |

#### gemma4:31b (n=200, correct=161, incorrect=39)

| motif | n present | % of traces | correct | incorrect | Fisher OR | Fisher p | Logit OR | 95% CI | Logit p |
|---|---|---|---|---|---|---|---|---|---|
| `late_arithmetic` | 0 | 0.0% | 0 | 0 | — | 1.0000 | — | — | — |
| `verbose_ungrounded` | 34 | 17.0% | 21 | 13 | 3.333 | 0.0041 | 0.300 | [0.134, 0.673] | 0.0035 |
| `early_branching` | 30 | 15.0% | 23 | 7 | 1.312 | 0.3608 | 0.762 | [0.301, 1.930] | 0.5663 |

### Cross-Model Replication

| motif | gpt-oss Fisher p | gpt-oss Logit OR | gemma4 Fisher p | gemma4 Logit OR | replicates |
|---|---|---|---|---|---|
| `late_arithmetic` | 1.0000 | — | 1.0000 | — | no |
| `verbose_ungrounded` | 1.55e-06 | 0.133 | 0.0041 | 0.300 | **yes** |
| `early_branching` | 0.0689 | 0.454 | 0.3608 | 0.762 | no |

`verbose_ungrounded` is the only targeted motif that replicates. Note that it is a conjunction of the two features already significant in the Mann-Whitney tests (`n_steps` and `frac_facts`), so it is best understood as a convenient threshold rule rather than a new mechanistic signal. In the 5-predictor regressions (motifs + depth + frac_facts), `verbose_ungrounded` retains significance for gpt-oss (p=0.040) but not for gemma4 (p=0.181), consistent with some collinearity with `frac_facts`.

---

## Additional Motifs

Five more binary flags derived from visual graph inspection.

### Definitions

| Flag | Definition |
|---|---|
| `has_offschema_node` | Any step whose `op` label is not in `{extract_fact, arithmetic, substitute, conclude, verify}` |
| `orphan_fact` | Any `extract_fact` node with out-degree 0 — a fact was extracted but never used |
| `linear_chain` | `n_steps >= 7` AND `mean_out_degree <= 1.1` AND `max_out_degree <= 2` — almost purely sequential trace |
| `high_fanin_conclude` | Any `conclude` node with in-degree >= 3 |
| `failed_verification` | Trace contains at least one `verify` step (diagnostic: measures whether verification was attempted, not whether it succeeded) |

### Per-Model Results

#### gpt-oss:120b (n=200, correct=159, incorrect=41)

| motif | n present | % of traces | correct | incorrect | Fisher OR | Fisher p | Logit OR | 95% CI | Logit p |
|---|---|---|---|---|---|---|---|---|---|
| `has_offschema_node` | 34 | 17.0% | 13 | 21 | 11.792 | 3.97e-09 | 0.085 | [0.037, 0.195] | 6.89e-09 |
| `orphan_fact` | 3 | 1.5% | 3 | 0 | 0.000 | 1.0000 | — | — | — |
| `linear_chain` | 86 | 43.0% | 70 | 16 | 0.814 | 0.7737 | 1.229 | [0.609, 2.478] | 0.5645 |
| `high_fanin_conclude` | 5 | 2.5% | 2 | 3 | 6.197 | 0.0595 | 0.161 | [0.026, 1.000] | 0.0500 |
| `failed_verification` | 52 | 26.0% | 36 | 16 | 2.187 | 0.0291 | 0.457 | [0.221, 0.948] | 0.0355 |

#### gemma4:31b (n=200, correct=161, incorrect=39)

| motif | n present | % of traces | correct | incorrect | Fisher OR | Fisher p | Logit OR | 95% CI | Logit p |
|---|---|---|---|---|---|---|---|---|---|
| `has_offschema_node` | 28 | 14.0% | 10 | 18 | 12.943 | 1.34e-08 | 0.077 | [0.031, 0.190] | 2.27e-08 |
| `orphan_fact` | 3 | 1.5% | 3 | 0 | 0.000 | 1.0000 | — | — | — |
| `linear_chain` | 89 | 44.5% | 70 | 19 | 1.235 | 0.3394 | 0.810 | [0.402, 1.632] | 0.5551 |
| `high_fanin_conclude` | 1 | 0.5% | 0 | 1 | inf | 0.1950 | — | — | — |
| `failed_verification` | 32 | 16.0% | 24 | 8 | 1.473 | 0.2631 | 0.679 | [0.279, 1.653] | 0.3936 |

### Cross-Model Replication

| motif | gpt-oss Fisher p | gpt-oss Logit OR | gemma4 Fisher p | gemma4 Logit OR | replicates |
|---|---|---|---|---|---|
| `has_offschema_node` | 3.97e-09 | 0.085 | 1.34e-08 | 0.077 | **yes** |
| `orphan_fact` | 1.0000 | — | 1.0000 | — | no |
| `linear_chain` | 0.7737 | 1.229 | 0.3394 | 0.810 | no |
| `high_fanin_conclude` | 0.0595 | 0.161 | 0.1950 | — | no |
| `failed_verification` | 0.0291 | 0.457 | 0.2631 | 0.679 | diagnostic |

`has_offschema_node` is the strongest individual predictor found in the project. In the extended regression controlling for depth, n_steps, and frac_facts, it remains the dominant term (gpt-oss: OR=0.082, p=1.18e-06; gemma4: OR=0.082, p=3.64e-06). The 5-motif model pseudo-R² is 0.204 (gpt-oss) and 0.173 (gemma4); the extended model reaches 0.321 and 0.278 respectively.

`failed_verification` is significant only in gpt-oss (p=0.029). The pattern — verify steps appearing more in incorrect traces (39% vs. 23% for gpt-oss, 21% vs. 15% for gemma4) — suggests models use verification reactively on problems they are already struggling with, not as a routine quality-control step.

---

## Within-Problem Comparison

### Model Concordance

Both models ran on the same 200 problems at temperature 0.

| outcome | count | % of problems |
|---|---|---|
| both correct | 153 | 76.5% |
| both incorrect | 33 | 16.5% |
| gpt-oss correct, gemma4 wrong | 6 | 3.0% |
| gemma4 correct, gpt-oss wrong | 8 | 4.0% |
| **discordant total** | **14** | **7.0%** |

The 14 discordant problems are the only ones usable for within-problem comparison (problem difficulty is held constant by construction).

### Feature Comparison on Discordant Pairs

For each pair, delta = incorrect model's feature − correct model's feature. Wilcoxon signed-rank test (one-sided, H1: delta > 0).

| feature | median correct | median incorrect | median delta | delta > 0 | Wilcoxon p | sig |
|---|---|---|---|---|---|---|
| `depth` | 4.00 | 4.50 | +0.50 | 50% | 0.3380 | ns |
| `n_steps` | 8.00 | 9.00 | +1.00 | 64% | 0.0231 | ** |
| `frac_facts` | 0.35 | 0.32 | −0.04 | 21% | 0.9817 | ns |
| `frac_arithmetic` | 0.43 | 0.40 | −0.01 | 43% | 0.5968 | ns |
| `max_out_degree` | 1.50 | 2.00 | +0.50 | 50% | 0.0478 | ** |
| `mean_out_degree` | 0.94 | 1.00 | +0.00 | 43% | 0.0252 | ** |
| `orphan_nodes` | 0.00 | 0.00 | +0.00 | 36% | 0.0169 | ** |
| `n_arithmetic` | 3.00 | 4.00 | +0.50 | 50% | 0.0478 | ** |
| `n_facts` | 3.00 | 3.00 | +0.00 | 29% | 0.6185 | ns |
| `unsupported_arithmetic` | 0.00 | 0.00 | +0.00 | 0% | 1.0000 | ns |

**Key result:** `n_steps` survives the within-problem control (p=0.023); `depth` does not (p=0.338). This means depth is partly a proxy for problem difficulty (harder problems require deeper chains and are more often wrong), while step count reflects something about reasoning quality even when the problem is held constant.

### Per-Pair Detail

| problem | winner | depth (correct) | depth (incorrect) | delta depth | steps (correct) | steps (incorrect) | delta steps |
|---|---|---|---|---|---|---|---|
| `gsm-hard-16` | gemma4 | 5 | 8 | +3 | 7 | 10 | +3 |
| `gsm-hard-30` | gptoss | 4 | 5 | +1 | 8 | 10 | +2 |
| `gsm-hard-b2-14` | gemma4 | 4 | 5 | +1 | 7 | 9 | +2 |
| `gsm-hard-b2-49` | gemma4 | 4 | 5 | +1 | 6 | 8 | +2 |
| `gsm-hard-b2-74` | gptoss | 3 | 4 | +1 | 8 | 9 | +1 |
| `gsm-hard-b2-113` | gptoss | 4 | 5 | +1 | 8 | 7 | −1 |
| `gsm-hard-b2-139` | gemma4 | 3 | 4 | +1 | 6 | 9 | +3 |
| `gsm-hard-45` | gptoss | 3 | 3 | +0 | 8 | 8 | +0 |
| `gsm-hard-b2-48` | gemma4 | 4 | 4 | +0 | 7 | 8 | +1 |
| `gsm-hard-b2-55` | gemma4 | 5 | 5 | +0 | 7 | 9 | +2 |
| `gsm-hard-b2-104` | gptoss | 5 | 4 | −1 | 9 | 10 | +1 |
| `gsm-hard-b2-126` | gemma4 | 8 | 7 | −1 | 10 | 9 | −1 |
| `gsm-hard-b2-3` | gemma4 | 6 | 4 | −2 | 8 | 6 | −2 |
| `gsm-hard-b2-68` | gptoss | 6 | 3 | −3 | 9 | 9 | +0 |

---

## Error Magnitude Analysis

Among the incorrect traces, errors split into two qualitatively different categories:

**Catastrophic errors (log10 relative error ≥ 3 — off by more than 1,000x):**
These are fundamental misunderstandings. The model either applies the wrong operation, loses track of units, or confuses a scaled number with an unscaled one. The predicted answer is in a completely different ballpark.

Examples:
- `gsm-hard-20` (gemma4): gold = 3.5e-07, predicted = 827,012 — off by ~2.4 billion
- `gsm-hard-b2-81` (gemma4): gold ≈ 9.7M, predicted ≈ 5e18 — 19-step trace, runaway exponentiation
- `gsm-hard-b2-1` (gpt-oss): gold = 10.0, model spent ~3,000 tokens trying to reconcile a non-integer answer, ultimately guessed 584,536

**Near misses (log10 relative error < 0 — off by a small factor):**
The model gets the right structure but makes a single arithmetic error — a wrong denominator, a missed halving, or a sign flip.

Examples:
- `gsm-hard-39` (gemma4): gold = 2,518,759.68, predicted = 1,574,224.8 (37.5% off — missed a factor of 5/8 vs. 1/2)
- `gsm-hard-b2-85` (gemma4): gold = 52,414,335, predicted = 78,621,480 (50% off — multiplied instead of divided by 1.5)
- `gsm-hard-b2-94` (gemma4): gold = 1,316,597, predicted = -1,316,590 (sign flip — perfect magnitude, wrong sign)
- `gsm-hard-b2-1` (gpt-oss): gold = 10.0, predicted = 584,536 (off by 58,000x — catastrophic, model noticed no integer solution existed but guessed anyway)

A notable qualitative difference between models: **gpt-oss produces much longer traces when uncertain**. On ambiguous problems, gpt-oss will spend 2,000-3,000 tokens exploring multiple interpretations before committing to an answer. Gemma4 tends to be more decisive. This verbosity doesn't improve accuracy (both models are at ~80%), but it does produce structurally larger graphs for the incorrect cases.

