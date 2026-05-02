# Targeted Motifs — gpt-oss

## Per-Motif Results

| motif | n_present | n_correct | n_incorrect | fisher OR | fisher p | logit OR | 95% CI | logit p |
|---|---|---|---|---|---|---|---|---|
| `late_arithmetic` | 1 | 1 | 0 | 0.000 | 1.0000 | — | — | — |
| `verbose_ungrounded` | 33 | 15 | 18 | 7.513 | 1.55e-06 | 0.133 | [0.059, 0.301] | 1.21e-06 |
| `early_branching` | 27 | 18 | 9 | 2.203 | 0.0689 | 0.454 | [0.187, 1.102] | 0.0811 |

## Combined Regression: 3 New Motifs

| feature | coef | OR | 95% CI | p-value | sig |
|---|---|---|---|---|---|
| `const` | 1.834 | — | — | 1.16e-15 | *** |
| `verbose_ungrounded` | -2.018 | 0.133 | [0.055, 0.320] | 6.53e-06 | *** |
| `early_branching` | 0.005 | 1.005 | [0.351, 2.879] | 0.9918 |  |

## Combined Regression: 3 New Motifs + depth + frac_facts

| feature | coef | OR | 95% CI | p-value | sig |
|---|---|---|---|---|---|
| `const` | 2.540 | — | — | 0.0386 | ** |
| `verbose_ungrounded` | -1.262 | 0.283 | [0.085, 0.945] | 0.0401 | ** |
| `depth` | -0.279 | 0.757 | [0.540, 1.060] | 0.1055 |  |
| `frac_facts` | 1.232 | 3.429 | [0.041, 286.903] | 0.5853 |  |
| `early_branching` | 0.125 | 1.133 | [0.388, 3.303] | 0.8194 |  |
