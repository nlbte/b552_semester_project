# Targeted Motifs — gemma4

## Per-Motif Results

| motif | n_present | n_correct | n_incorrect | fisher OR | fisher p | logit OR | 95% CI | logit p |
|---|---|---|---|---|---|---|---|---|
| `late_arithmetic` | 0 | 0 | 0 | — | 1.0000 | — | — | — |
| `verbose_ungrounded` | 34 | 21 | 13 | 3.333 | 0.0041 | 0.300 | [0.134, 0.673] | 0.0035 |
| `early_branching` | 30 | 23 | 7 | 1.312 | 0.3608 | 0.762 | [0.301, 1.930] | 0.5663 |

## Combined Regression: 3 New Motifs

| feature | coef | OR | 95% CI | p-value | sig |
|---|---|---|---|---|---|
| `const` | 1.673 | — | — | 2.75e-14 | *** |
| `verbose_ungrounded` | -1.230 | 0.292 | [0.126, 0.681] | 0.0044 | *** |
| `early_branching` | 0.105 | 1.111 | [0.407, 3.028] | 0.8376 |  |

## Combined Regression: 3 New Motifs + depth + frac_facts

| feature | coef | OR | 95% CI | p-value | sig |
|---|---|---|---|---|---|
| `const` | 3.278 | — | — | 0.0066 | *** |
| `depth` | -0.358 | 0.699 | [0.500, 0.977] | 0.0358 | ** |
| `verbose_ungrounded` | -0.758 | 0.469 | [0.155, 1.422] | 0.1810 |  |
| `early_branching` | 0.251 | 1.285 | [0.462, 3.576] | 0.6307 |  |
| `frac_facts` | -0.344 | 0.709 | [0.008, 61.504] | 0.8800 |  |
