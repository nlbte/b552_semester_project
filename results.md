# B552 Project Results


## 1. How to read this repo

**Data lives in two directories:**
- `gsm_hard_data/` contains everything for gpt-oss:120b. Raw traces, DAG graphs, CSVs of features and test results, and figures.
- `experiments/gemma4-31b-cloud/` contains the same for gemma4:31b.

**Cross-model report files live in the repo root:** `targeted_motifs_report.md`, `additional_motifs_report.md`, `within_problem_report.md`.

**The analysis pipeline runs in this order:**

| Script | What it does |
|---|---|
| `gsm_hard_ollama.py` | Samples problems from GSM-Hard and runs them through an Ollama Cloud model, saving raw thinking traces and answers |
| `traces_to_graphs.py` | Asks an LLM to read each trace and convert it into a DAG: a list of reasoning steps with `op` labels and dependency edges |
| `motif_analysis.py` | Computes 14 structural features per graph (step count, depth, fact ratio, etc.) and runs Mann-Whitney U tests comparing correct vs. incorrect traces |
| `extended_analysis.py` | Correlation matrix, standardized logistic regression, original 5 motif flags, out-degree breakdown by step type, error magnitude analysis |
| `stats_tests.py` | Fisher's exact test and logistic regression with odds ratios for each feature |
| `targeted_motifs.py` | Tests three hypothesis-driven binary flags: `late_arithmetic`, `verbose_ungrounded`, `early_branching` |
| `additional_motifs.py` | Tests five more binary flags derived from visual graph inspection: `has_offschema_node`, `orphan_fact`, `linear_chain`, `high_fanin_conclude`, `failed_verification` |
| `within_problem_comparison.py` | Paired analysis of the 14 problems where the two models disagreed, eliminating problem difficulty as a confound |
| `targeted_motifs_report.py` / `additional_motifs_report.py` | Load per-model CSVs from the motif scripts and generate the cross-model markdown reports |
| `visualize_results.py`, `visualize_graphs.py` | Generate all figures |

---

## 3. The main findings, in plain English

### Finding 1: Wrong reasoning is bigger

Incorrect traces are structurally larger than correct ones, across both models, with nearly identical effect sizes.

For gpt-oss, incorrect traces average 9.5 steps vs. 7.5 for correct ones (p=5.3e-07). For gemma4, it's 9.0 vs. 7.3 (p=2.6e-05). Depth (longest chain from input to output in the reasoning graph) shows the same pattern: medians of 5 vs. 4 in both models, significant in both. Odds ratios for depth are 0.673 (gpt-oss, p=0.015) and 0.664 (gemma4, p=0.013), meaning each additional reasoning step of depth cuts the odds of a correct answer by roughly a third. The effect sizes are nearly identical across two very different models.

One nuance: in the within-problem analysis (see Finding 6 and Section 5), step count holds up as a real reasoning-quality signal, but depth is partly a proxy for problem difficulty. See `feature_comparison.csv` in either model directory for the full feature table.

### Finding 2: Anchoring to the problem matters

Correct traces spend a higher fraction of their steps pulling facts directly from the problem statement. This fraction (`frac_facts`) averages 0.359 in correct gpt-oss traces and 0.287 in incorrect ones (p=2.9e-04). For gemma4 the split is 0.341 vs. 0.308 (p=0.037). The gap is smaller in gemma4 but directionally identical.

The interpretation: models that fail spend proportionally less time grounding their reasoning in the given numbers, and more time in computation built on earlier steps. Whether this reflects a cause (rushing into arithmetic before fully reading the problem) or a consequence (more computation steps dilute the fact fraction) is hard to separate, but the signal is real and replicated. See `feature_comparison.csv` and `logistic_regression.csv` in either model directory.

### Finding 3: Off-schema steps are the strongest single predictor

This is the most striking finding in the project. When the graph extractor produces a step labeled `other` or `substitute` (types that don't fit the standard five-step schema of `extract_fact`, `arithmetic`, `substitute`, `conclude`, `verify`), the trace is about 12 times more likely to be wrong.

For gpt-oss: 21 of 41 incorrect traces (51%) contain an off-schema step, vs. 13 of 159 correct ones (8%). Fisher p=3.97e-09, logit OR=0.085. For gemma4: 18 of 39 incorrect traces (46%) vs. 10 of 161 correct ones (6%). Fisher p=1.34e-08, logit OR=0.077. Both results survive the extended regression that controls for step count, fact ratio, and depth. This is the highest-significance finding in the project by a wide margin.

The interpretation is ambiguous: off-schema steps could reflect the reasoning model doing something structurally weird, or they could reflect the graph extractor struggling to classify an unusual step. Both explanations predict failure, just via different mechanisms. We plan to split the flag into `has_other_node` and `has_substitute_node` to see which is driving the result. See `additional_motifs_report.md` for full numbers.

### Finding 4: Verification is a distress signal

Traces that contain explicit `verify` steps are more likely to be wrong in gpt-oss (p=0.029, OR=0.457 on correctness), and directionally the same in gemma4 (p=0.26, not significant). The base rates: gpt-oss has verify steps in 39% of its incorrect traces vs. 23% of correct ones; gemma4 has them in 21% of incorrect traces vs. 15% of correct ones.

This is the opposite of what you'd expect if verification were a positive reasoning habit. The likely explanation is that models invoke explicit verification reactively, on problems they're already uncertain about, rather than as a routine step. The gpt-oss signal is significant; gemma4 shows the same direction but weaker. This result is gpt-oss specific in terms of statistical significance. See `additional_motifs_report.md`.

### Finding 5: Errors split into two qualitatively different types

Among the 40 or so incorrect traces per model, errors fall into two distinct buckets. Catastrophic errors are off by a factor of 1000 or more: the model applies the right procedure to the wrong problem, loses track of units, or confuses a scaled number with a raw number. Near misses are off by a small factor: the model gets the structure right but makes a single arithmetic error, such as multiplying where it should divide, or computing the wrong denominator. These two failure modes probably require different interventions. A model that makes catastrophic errors needs better problem comprehension; a model that makes near misses needs better arithmetic precision.

gpt-oss produces notably longer traces when uncertain. On ambiguous problems, it will spend 2,000-3,000 tokens exploring multiple interpretations. This verbosity explains some of the step-count signal: failing traces are bigger partly because gpt-oss tries harder before giving up. See `error_split.csv` and `error_distribution.png` in the gemma4 directory.

### Finding 6: The two models fail in similar ways

The same structural features predict failure in both gpt-oss and gemma4, at nearly identical effect sizes. The depth odds ratios of 0.673 and 0.664 are as close as you could plausibly expect from two independently run models. The step-count significance values are 5.3e-07 and 2.6e-05. The fact-ratio signal appears in both. The `has_offschema_node` logit ORs are 0.085 and 0.077.

This replication across models is the main reason to take these findings seriously. If it were just one model, the signals could reflect a quirk of that model's generation style. The fact that two models with different architectures and different verbosity levels show the same structural patterns in their failures is evidence that we're measuring something real about math reasoning, not a model artifact.

---

## 4. What did and didn't work in the targeted motif analyses

**`verbose_ungrounded`** (n_steps >= 8 AND frac_facts < 0.3) is the one targeted motif that replicates: Fisher p=1.55e-06 / logit OR=0.133 for gpt-oss and p=0.0041 / OR=0.300 for gemma4. It is best understood as a convenient threshold version of the two features already significant in the main analysis (step count and fact ratio) rather than a new independent signal, but it is a practically useful single-flag summary of both.

**`late_arithmetic`** fires when an arithmetic node is deep in the graph and has no `extract_fact` ancestor. This is a reasonable hypothesis about ungrounded computation, but the flag fires only once in gpt-oss (on a correct trace) and zero times in gemma4. It's too rare to test statistically.

**`early_branching`** fires when a node in the first half of a trace has out-degree 3 or more, indicating premature fan-out before the reasoning has converged. This showed marginal significance in gpt-oss (Fisher p=0.069) but not in gemma4 (p=0.36). It did not meet the replication criterion.

**`linear_chain`** flags long traces that are almost purely sequential (low mean and max out-degree). About 43-44% of traces in both models qualify, with nearly identical rates in correct and incorrect traces. The main effect is null, and a `linear_chain x n_steps` interaction term is also null in both models.

**`high_fanin_conclude`** requires a conclude node with 3 or more incoming edges. This fires in 5 gpt-oss traces and 1 gemma4 trace. Too rare to draw conclusions, and the min-count guard prevents logistic regression from running for gemma4.

**`orphan_fact`** fires when an `extract_fact` node has no outgoing edges (a fact was extracted but never used). Fires 3 times per model, and in all 6 cases the trace is correct. Essentially a null result by prevalence.

**`unsupported_conclude`** is zero across all 200 traces in both models. Both models always trace their final conclusion back to some prior step, even when that step is wrong. We tested it anyway; there is nothing to find.

**Individual original motif flags** (`has_unsupported_arith`, `has_orphan`, `has_unsupported_conclude`, `low_fact_ratio`, `long_chain`) were not individually significant. The composite `motif_count` (sum of all five) did predict failure, because bad patterns co-occur rather than because each flag is independently diagnostic.

---

## 5. Robustness check: within-problem comparison

A potential objection to the main findings is that hard problems both require more steps to solve and are more often wrong. If that's the explanation, the structural differences between correct and incorrect traces reflect problem difficulty, not reasoning quality.

We addressed this directly by looking only at the 14 problems where the two models disagreed. On these problems, one model got the right answer and the other didn't, while the problem itself was held constant. Any structural difference within a disagreeing pair reflects how the models reasoned, not what problem they were given.

Of the 200 shared problems, 153 were correct by both models and 33 were wrong by both. That leaves 14 discordant cases: 6 where gpt-oss was right and gemma4 was wrong, and 8 where gemma4 was right and gpt-oss was wrong. These are the only pairs usable for the within-problem test.

The result: step count (`n_steps`) survives the control. In 64% of disagreeing pairs, the incorrect trace has more steps than the correct trace, and the Wilcoxon signed-rank test (a non-parametric test for whether a consistent direction exists across paired observations) gives p=0.023. Depth does not survive the control: only 50% of pairs have the incorrect trace deeper, p=0.338. This suggests depth is partly driven by problem difficulty (harder problems are both deeper and more often wrong), while step count reflects something about reasoning quality even within the same problem.

The sample is small (n=14) and the results should be treated as directional. See `within_problem_report.md` for the full per-pair table and all 10 features.

---

## 6. What each result file holds

| File | Description |
|---|---|
| `gsm_hard_data/gsm_hard_traces.jsonl` | Raw gpt-oss outputs: thinking trace, predicted answer, correctness, token counts |
| `gsm_hard_data/gsm_hard_graphs_all.jsonl` | DAG graphs for all 200 gpt-oss problems (steps with op labels and dependency edges) |
| `gsm_hard_data/graph_features.csv` | 14 structural features per graph for gpt-oss |
| `gsm_hard_data/feature_comparison.csv` | Mann-Whitney U results: correct vs. incorrect means, medians, p-values for gpt-oss |
| `gsm_hard_data/logistic_regression.csv` | Standardized logistic regression coefficients for gpt-oss |
| `gsm_hard_data/logistic_odds_ratios.csv` | Odds-ratio logistic regression for gpt-oss |
| `gsm_hard_data/motif_scores.csv` | Five original binary motif flags plus composite count, per trace, for gpt-oss |
| `gsm_hard_data/targeted_motifs_gptoss.csv` | Three targeted motif flags (`late_arithmetic`, `verbose_ungrounded`, `early_branching`) for gpt-oss |
| `gsm_hard_data/additional_motifs_gptoss.csv` | Five additional motif flags (`has_offschema_node`, `orphan_fact`, `linear_chain`, `high_fanin_conclude`, `failed_verification`) for gpt-oss |
| `gsm_hard_data/error_split.csv` | Per-trace gold answer, predicted answer, relative error, log10 error for gpt-oss |
| `gsm_hard_data/outdegree_by_op.csv` | Node-level out-degree broken down by step type for gpt-oss |
| `gsm_hard_data/graph_grids/ollama_correct_20.png` | Grid of 20 correct gpt-oss reasoning graphs |
| `gsm_hard_data/graph_grids/ollama_incorrect_20.png` | Grid of 20 incorrect gpt-oss reasoning graphs |
| `experiments/gemma4-31b-cloud/gsm_hard_traces.jsonl` | Raw gemma4 outputs |
| `experiments/gemma4-31b-cloud/gsm_hard_graphs.jsonl` | DAG graphs for all 200 gemma4 problems |
| `experiments/gemma4-31b-cloud/graph_features.csv` | 14 structural features per graph for gemma4 |
| `experiments/gemma4-31b-cloud/feature_comparison.csv` | Mann-Whitney U results for gemma4 |
| `experiments/gemma4-31b-cloud/logistic_regression.csv` | Standardized logistic regression for gemma4 |
| `experiments/gemma4-31b-cloud/logistic_odds_ratios.csv` | Odds-ratio regression for gemma4 |
| `experiments/gemma4-31b-cloud/motif_scores.csv` | Original 5 motif flags for gemma4 |
| `experiments/gemma4-31b-cloud/additional_motifs_gemma4.csv` | Five additional motif flags for gemma4 |
| `experiments/gemma4-31b-cloud/error_split.csv` | Per-trace error data for gemma4 |
| `experiments/gemma4-31b-cloud/correlation_matrix.png` | Spearman correlation heatmap for gemma4 features |
| `experiments/gemma4-31b-cloud/feature_comparison.png` | Boxplots of correct vs. incorrect feature distributions for gemma4 |
| `experiments/gemma4-31b-cloud/error_distribution.png` | Histogram of log10 relative error for gemma4 |
| `experiments/gemma4-31b-cloud/case_studies.png` | DAG visualizations: 2 correct, 2 incorrect gemma4 traces |
| `experiments/gemma4-31b-cloud/graph_grids/` | Grids of 20 correct and 20 incorrect gemma4 graphs |
| `targeted_motifs_report.md` | Cross-model results for the three targeted motifs |
| `additional_motifs_report.md` | Cross-model results for the five additional motifs |
| `within_problem_report.md` | Within-problem paired analysis: 14 discordant pairs, Wilcoxon results, per-pair table |
| `results_technical.md` | Full technical results report with all tables and raw numbers (preserved from earlier version) |

---

## 7. How to reproduce

```bash
# --- stage 1: collect traces ---
# run gpt-oss on 200 GSM-Hard problems (creates gsm_hard_data/gsm_hard_traces.jsonl)
python gsm_hard_ollama.py --model gpt-oss:120b-cloud --n 200 --out gsm_hard_data

# run gemma4 on the same 200 problems (--manifest reuses the same problem set)
python gsm_hard_ollama.py --model gemma4:31b-cloud --n 200 --manifest gsm_hard_data/manifest.json --out experiments/gemma4-31b-cloud

# --- stage 2: extract graphs ---
# convert traces to DAGs (creates gsm_hard_graphs_all.jsonl)
python traces_to_graphs.py --traces gsm_hard_data/gsm_hard_traces.jsonl --out gsm_hard_data/gsm_hard_graphs_all.jsonl

python traces_to_graphs.py --traces experiments/gemma4-31b-cloud/gsm_hard_traces.jsonl --out experiments/gemma4-31b-cloud/gsm_hard_graphs.jsonl

# --- stage 3: feature analysis ---
python motif_analysis.py --graphs gsm_hard_data/gsm_hard_graphs_all.jsonl --traces gsm_hard_data/gsm_hard_traces.jsonl --out gsm_hard_data

python motif_analysis.py --graphs experiments/gemma4-31b-cloud/gsm_hard_graphs.jsonl --traces experiments/gemma4-31b-cloud/gsm_hard_traces.jsonl --out experiments/gemma4-31b-cloud

python extended_analysis.py --graphs gsm_hard_data/gsm_hard_graphs_all.jsonl --traces gsm_hard_data/gsm_hard_traces.jsonl --out gsm_hard_data

python extended_analysis.py --graphs experiments/gemma4-31b-cloud/gsm_hard_graphs.jsonl --traces experiments/gemma4-31b-cloud/gsm_hard_traces.jsonl --out experiments/gemma4-31b-cloud

python stats_tests.py --graphs gsm_hard_data/gsm_hard_graphs_all.jsonl --traces gsm_hard_data/gsm_hard_traces.jsonl --out gsm_hard_data

python stats_tests.py --graphs experiments/gemma4-31b-cloud/gsm_hard_graphs.jsonl --traces experiments/gemma4-31b-cloud/gsm_hard_traces.jsonl --out experiments/gemma4-31b-cloud

# --- stage 4: motif analyses ---
python targeted_motifs.py \
    --graphs gsm_hard_data/gsm_hard_graphs_all.jsonl \
    --traces gsm_hard_data/gsm_hard_traces.jsonl \
    --label gpt-oss --out-csv gsm_hard_data/targeted_motifs_gptoss.csv

python targeted_motifs.py \
    --graphs experiments/gemma4-31b-cloud/gsm_hard_graphs.jsonl \
    --traces experiments/gemma4-31b-cloud/gsm_hard_traces.jsonl \
    --label gemma4 --out-csv experiments/gemma4-31b-cloud/targeted_motifs_gemma4.csv

python targeted_motifs_report.py \
    --gptoss gsm_hard_data/targeted_motifs_gptoss.csv \
    --gemma4 experiments/gemma4-31b-cloud/targeted_motifs_gemma4.csv \
    --out targeted_motifs_report.md

python additional_motifs.py \
    --graphs gsm_hard_data/gsm_hard_graphs_all.jsonl \
    --traces gsm_hard_data/gsm_hard_traces.jsonl \
    --label gpt-oss --out-csv gsm_hard_data/additional_motifs_gptoss.csv

python additional_motifs.py \
    --graphs experiments/gemma4-31b-cloud/gsm_hard_graphs.jsonl \
    --traces experiments/gemma4-31b-cloud/gsm_hard_traces.jsonl \
    --label gemma4 --out-csv experiments/gemma4-31b-cloud/additional_motifs_gemma4.csv

python additional_motifs_report.py \
    --gptoss gsm_hard_data/additional_motifs_gptoss.csv \
    --gemma4 experiments/gemma4-31b-cloud/additional_motifs_gemma4.csv \
    --out additional_motifs_report.md

# --- stage 5: within-problem comparison ---
python within_problem_comparison.py \
    --gptoss gsm_hard_data/graph_features.csv \
    --gemma4 experiments/gemma4-31b-cloud/graph_features.csv \
    --out within_problem_report.md

# --- stage 6: figures ---
python visualize_results.py --traces gsm_hard_data/gsm_hard_traces.jsonl --out gsm_hard_data
python visualize_graphs.py --graphs gsm_hard_data/gsm_hard_graphs_all.jsonl --traces gsm_hard_data/gsm_hard_traces.jsonl --out gsm_hard_data/graph_grids

python visualize_graphs.py --graphs experiments/gemma4-31b-cloud/gsm_hard_graphs.jsonl --traces experiments/gemma4-31b-cloud/gsm_hard_traces.jsonl --out experiments/gemma4-31b-cloud/graph_grids
```

---

## 8. Honest limitations

**The graph extractor is an LLM.** All features are derived from graphs built by asking a language model to segment each trace into steps and draw dependency edges. The extractor decides what counts as `extract_fact` vs. `arithmetic` vs. `other`, and how edges are drawn. If the extractor is inconsistent, or if it produces different segmentations for correct vs. incorrect traces for reasons unrelated to the reasoning structure, every downstream feature is affected. There is no ground truth for graph structure. The `has_offschema_node` finding in particular could partially reflect the extractor struggling with unusual trace formats rather than the reasoning model producing unusual steps.

**Two models is not a generalization.** Both models are Ollama Cloud deployments with similar training paradigms. The cross-model replication is encouraging, but it doesn't establish that these structural signals hold for models trained very differently (e.g., smaller models, models trained without chain-of-thought, models in non-English languages).

**The within-problem analysis has 14 pairs.** With n=14, the Wilcoxon test requires consistent directional effects to reach significance. Findings that are significant at n=14 are robust to the available data, but the analysis is underpowered for detecting weak effects, and the 14 discordant pairs may not be representative of typical disagreements between models.

**GSM-Hard is grade-school math with big numbers.** The benchmark is designed to stress-test arithmetic by replacing small numbers with large ones, not to test general mathematical reasoning. The structural signals we find may be specific to this type of problem and may not transfer to problems requiring deeper mathematical understanding, multi-step proofs, or symbolic manipulation.
