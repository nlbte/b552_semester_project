# Reasoning Graphs for Interpretable Chain-of-Thought Analysis

This repository contains the code and artifacts for a B552 semester project on
post-hoc analysis of chain-of-thought reasoning traces. The core idea is to run
large language models on GSM-Hard math word problems, convert each free-form
reasoning trace into a directed reasoning graph, and then test whether graph
structure differs between correct and incorrect answers.

The final paper is intentionally short. This README is the more detailed
technical record: what each script does, how to reproduce the pipeline, what
design decisions were made, and how those decisions affected the results.

## Project Summary

We tested two Ollama Cloud models on the same fixed 200-problem subset of
GSM-Hard:

| Model | Correct | Incorrect | Accuracy |
| --- | ---: | ---: | ---: |
| `gpt-oss:120b-cloud` | 159 | 41 | 79.5% |
| `gemma4:31b-cloud` | 161 | 39 | 80.5% |

The pipeline found repeatable structural differences between successful and
failed traces:

- Incorrect traces had more graph nodes and longer dependency chains.
- Correct traces devoted a larger fraction of their steps to extracting facts
  from the problem statement.
- Off-schema graph steps were the strongest single failure signal, but this is
  also the most extractor-dependent result.
- In the 14 problems where the two models disagreed, step count remained a
  clearer signal of failure than graph depth after partially controlling for
  problem difficulty.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `gsm_hard_ollama.py` | Samples or loads GSM-Hard problems, sends them to Ollama Cloud, saves prompts, traces, checkpoints, errors, and run metadata. |
| `gsm_hard_claude.py` | Converts raw trace JSONL files into reasoning graph JSONL files using Claude Opus 4.6. |
| `motif_analysis.py` | Computes 14 graph features and runs Mann-Whitney U tests for correct vs. incorrect traces. |
| `extended_analysis.py` | Produces correlation matrices, logistic regressions, motif scores, out-degree analysis, error splits, and case-study figures. |
| `stats_tests.py` | Runs Fisher's exact test and odds-ratio logistic regression on saved motif scores. |
| `targeted_motifs.py` | Tests three hypothesis-driven motifs: `late_arithmetic`, `verbose_ungrounded`, and `early_branching`. |
| `additional_motifs.py` | Tests five additional motifs: `has_offschema_node`, `orphan_fact`, `linear_chain`, `high_fanin_conclude`, and `failed_verification`. |
| `within_problem_comparison.py` | Merges the two models' feature tables and runs paired tests on problems where only one model is correct. |
| `visualize_results.py` | Plots trace-level accuracy, token counts, and relative error summaries. |
| `visualize_graphs.py` | Draws grids of correct and incorrect reasoning graphs. |
| `results_gptoss/` | Committed gpt-oss analysis artifacts and figures. |
| `results_gemma4/` | Committed gemma4 analysis artifacts and figures. |
| `_reports/` | Longer generated result reports used while writing the final paper. |
| `appendix/figures/` | Figures copied into appendix-ready names. |
| `paper/` | AAAI-style paper source and template files. |

## Setup

Use Python 3.10 or newer. The project was run from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and add the API keys needed for the stages you plan to run:

```text
OLLAMA_API_KEY=your_ollama_key
ANTHROPIC_API_KEY=your_anthropic_key
```

`gsm_hard_ollama.py` loads `OLLAMA_API_KEY` from the environment first, then
from `.env`, and prompts once with `getpass` if no key is found. The graph
extraction script uses the Anthropic client, which reads `ANTHROPIC_API_KEY`.

## Graph Schema

Each graph is a JSON object with one list of ordered steps:

```json
{
  "question_id": "gsm-hard-0",
  "steps": [
    {
      "id": "S1",
      "text": "extract the number of items",
      "depends_on": [],
      "op": "extract_fact"
    },
    {
      "id": "S2",
      "text": "multiply the two quantities",
      "depends_on": ["S1"],
      "op": "arithmetic"
    }
  ],
  "final_answer": "123",
  "gold_answer": 123.0,
  "is_correct": true
}
```

The allowed operation labels are:

- `extract_fact`
- `arithmetic`
- `substitute`
- `conclude`
- `verify`
- `other`

Edges are stored in each step's `depends_on` list. The downstream scripts build
a `networkx.DiGraph` from those dependencies and compute feature values from
the graph.

## Reproducing the Pipeline

The committed artifacts are already sufficient to rerun the analysis stages.
Full trace collection and graph extraction require paid cloud APIs.

### 1. Collect traces from Ollama Cloud

To create a new 200-problem gpt-oss run:

```powershell
python gsm_hard_ollama.py `
  --model gpt-oss:120b `
  --sample-size 200 `
  --output-dir runs `
  --temperature 0 `
  --run-label gptoss_t0
```

The script automatically sends `gpt-oss:120b-cloud` to Ollama if the model name
does not already end in `-cloud`. By default, outputs are grouped under a
model-specific folder such as:

```text
runs/gpt-oss-120b-cloud/
```

To run a second model on exactly the same problems, use the first run's sample
file as a manifest:

```powershell
python gsm_hard_ollama.py `
  --model gemma4:31b `
  --from-sample-jsonl runs/gpt-oss-120b-cloud/gsm_hard_sample.jsonl `
  --output-dir runs `
  --temperature 0 `
  --run-label gemma4_t0
```

The current committed full-manifest file is:

```text
results_gptoss/manifest_aligned_graphs_all.jsonl
```

It contains the fixed 200 problem ids, inputs, and targets used for aligned
reruns.

### 2. Retry failed API calls without breaking alignment

If an Ollama run leaves a `gsm_hard_errors.jsonl` file, rerun only those ids and
merge the successful retry rows back into the existing trace file:

```powershell
python gsm_hard_ollama.py `
  --model gemma4:31b `
  --from-sample-jsonl results_gptoss/manifest_aligned_graphs_all.jsonl `
  --output-dir runs `
  --only-ids-from-errors runs/gemma4-31b-cloud/gsm_hard_errors.jsonl `
  --merge-traces-into runs/gemma4-31b-cloud/gsm_hard_traces.jsonl
```

Manifest mode writes one row per manifest id by default. If a trace is still
missing, the script inserts a `_stub` row so the trace file length and ordering
stay aligned with the manifest. This matters because later cross-model analysis
depends on `question_id` matching the same problem in both model outputs.

### 3. Convert traces to reasoning graphs

```powershell
python gsm_hard_claude.py `
  --traces runs/gpt-oss-120b-cloud/gsm_hard_traces.jsonl `
  --out runs/gpt-oss-120b-cloud/gsm_hard_graphs.jsonl
```

For the committed artifacts, the final graph files used by the analysis are:

```text
results_gptoss/gsm_hard_graphs_all.jsonl
results_gemma4/gsm_hard_graphs.jsonl
```

The gpt-oss run was built historically from an initial 50-problem run plus a
second 150-problem batch. The full 200-problem graph-level file is
`results_gptoss/gsm_hard_graphs_all.jsonl`.

### 4. Compute structural features

```powershell
python motif_analysis.py `
  --graphs results_gptoss/gsm_hard_graphs_all.jsonl `
  --out-dir results_gptoss

python motif_analysis.py `
  --graphs results_gemma4/gsm_hard_graphs.jsonl `
  --out-dir results_gemma4
```

This writes:

- `graph_features.csv`
- `feature_comparison.csv`
- `feature_comparison.png`

### 5. Run extended analysis

```powershell
python extended_analysis.py `
  --graphs results_gptoss/gsm_hard_graphs_all.jsonl `
  --features results_gptoss/graph_features.csv `
  --out-dir results_gptoss

python extended_analysis.py `
  --graphs results_gemma4/gsm_hard_graphs.jsonl `
  --features results_gemma4/graph_features.csv `
  --out-dir results_gemma4
```

This writes correlation plots, logistic regression tables, motif scores,
out-degree summaries, error magnitude splits, and case-study figures.

### 6. Run odds-ratio tests

```powershell
python stats_tests.py --data-dir results_gptoss
python stats_tests.py --data-dir results_gemma4
```

### 7. Run targeted motif tests

```powershell
python targeted_motifs.py `
  --graphs results_gptoss/gsm_hard_graphs_all.jsonl `
  --traces results_gptoss/gsm_hard_traces.jsonl `
  --label gpt-oss `
  --out-csv results_gptoss/targeted_motifs_gptoss.csv

python targeted_motifs.py `
  --graphs results_gemma4/gsm_hard_graphs.jsonl `
  --traces results_gemma4/gsm_hard_traces.jsonl `
  --label gemma4 `
  --out-csv results_gemma4/targeted_motifs_gemma4.csv
```

### 8. Run additional motif tests

```powershell
python additional_motifs.py `
  --graphs results_gptoss/gsm_hard_graphs_all.jsonl `
  --traces results_gptoss/gsm_hard_traces.jsonl `
  --label gpt-oss `
  --out-csv results_gptoss/additional_motifs_gptoss.csv

python additional_motifs.py `
  --graphs results_gemma4/gsm_hard_graphs.jsonl `
  --traces results_gemma4/gsm_hard_traces.jsonl `
  --label gemma4 `
  --out-csv results_gemma4/additional_motifs_gemma4.csv
```

### 9. Run within-problem comparison

```powershell
python within_problem_comparison.py `
  --gptoss-features results_gptoss/graph_features.csv `
  --gemma4-features results_gemma4/graph_features.csv
```

This analysis keeps only the 14 problems where exactly one model was correct.
For each discordant pair, it compares the feature value from the correct trace
against the feature value from the incorrect trace on the same problem.

### 10. Generate figures

```powershell
python visualize_results.py `
  --traces results_gemma4/gsm_hard_traces.jsonl results_gptoss/gsm_hard_traces.jsonl `
  --out results_gptoss/summary.png

python visualize_graphs.py `
  --ollama-graphs results_gptoss/gsm_hard_graphs_all.jsonl `
  --gemma4-graphs results_gemma4/gsm_hard_graphs.jsonl `
  --out-dir results_gptoss/graph_grids
```

## Output Files

Important committed analysis artifacts:

| File | Meaning |
| --- | --- |
| `results_gptoss/gsm_hard_graphs_all.jsonl` | Full 200-problem gpt-oss reasoning graphs. |
| `results_gemma4/gsm_hard_graphs.jsonl` | Full 200-problem gemma4 reasoning graphs. |
| `results_gptoss/graph_features.csv` | One row per gpt-oss graph with structural features. |
| `results_gemma4/graph_features.csv` | One row per gemma4 graph with structural features. |
| `results_*/feature_comparison.csv` | Mann-Whitney U comparisons of correct vs. incorrect traces. |
| `results_*/logistic_regression.csv` | Standardized logistic regression coefficients and p-values. |
| `results_*/logistic_odds_ratios.csv` | Odds ratios from unstandardized logistic regression. |
| `results_*/motif_scores.csv` | Original five binary motif flags plus total motif count. |
| `results_*/targeted_motifs_*.csv` | Three targeted motif flags per trace. |
| `results_*/additional_motifs_*.csv` | Five additional motif flags per trace. |
| `results_*/error_split.csv` | Gold answer, predicted answer, relative error, and log10 error. |
| `_reports/results.md` | Plain-English findings report. |
| `_reports/results_technical.md` | Full technical result tables and interpretation. |

## Feature Definitions

`motif_analysis.py` computes 14 graph-level features:

| Feature | Definition |
| --- | --- |
| `n_steps` | Number of graph nodes. |
| `depth` | Longest dependency path length in the DAG. |
| `n_arithmetic` | Count of arithmetic-like nodes. |
| `n_facts` | Count of fact-extraction-like nodes. |
| `n_conclude` | Count of conclusion-like nodes. |
| `frac_arithmetic` | `n_arithmetic / n_steps`. |
| `frac_facts` | `n_facts / n_steps`. |
| `unsupported_arithmetic` | Arithmetic nodes with no dependencies. |
| `unsupported_conclude` | Conclusion nodes with no incoming edges. |
| `conclude_in_degree` | Maximum in-degree among conclusion nodes. |
| `max_out_degree` | Maximum node out-degree. |
| `mean_out_degree` | Mean node out-degree. |
| `mean_in_degree` | Mean node in-degree. |
| `orphan_nodes` | Non-conclusion nodes with out-degree 0. |

The first analysis pass uses Mann-Whitney U tests because most features are
non-normal counts, ratios, or sparse flags. The extended analysis then uses
logistic regression to test whether features retain signal after accounting for
correlation.

## Design Decisions and Performance Effects

### 1. Use GSM-Hard rather than a broad benchmark

GSM-Hard preserves grade-school word-problem structure while replacing small
numbers with larger ones. We chose it because it stresses arithmetic reasoning
without requiring domain-specific outside knowledge. This made it more likely
that failures would appear in the reasoning trace rather than in missing facts.

The tradeoff is scope. The conclusions are strongest for arithmetic word
problems and should not be assumed to transfer directly to law, science, proof,
or open-domain reasoning.

### 2. Run both models on the same fixed 200-problem manifest

The project started as a smaller sampled run, then expanded to a 200-problem
comparison. Once we wanted to compare two models, random sampling became a
problem: different problem sets would confound model behavior with problem
difficulty. Manifest mode in `gsm_hard_ollama.py` solves this by loading fixed
`id`, `input`, and `target` rows from JSONL instead of sampling again from
HuggingFace.

This decision enabled the within-problem analysis. The two models agreed on 186
of 200 problems and disagreed on 14. Those 14 discordant examples let us compare
a correct and incorrect trace for the same question, which changed our
interpretation: depth looked highly predictive in the pooled analysis, but step
count was the cleaner signal after the problem was held constant.

### 3. Use deterministic decoding

Both model runs used temperature 0. This reduces randomness in the traces, makes
reruns easier to compare, and makes structural differences easier to attribute
to the model and problem rather than sampling noise.

The cost is that we did not measure self-consistency or variation across
multiple sampled traces. A future version could run multiple temperatures or
seeds and test whether the same graph features predict answer stability.

### 4. Require a final answer marker and parse answers automatically

The Ollama system prompt asks the model to end with:

```text
FINAL ANSWER: <number>
```

`extract_final_answer()` still includes fallback regexes for boxed answers,
"final answer is", "answer:", and the last numeric token in the last non-empty
line. This design made it possible to compute correctness automatically across
hundreds of runs.

The parser improved throughput, but it also creates a small risk: a strangely
formatted response can be marked wrong even if the prose contains the correct
answer. The fallback rules reduce but do not eliminate that risk.

### 5. Save prompts, samples, traces, checkpoints, errors, and metadata

Each trace run writes several files rather than only the final answers:

- `gsm_hard_sample.jsonl` records the problem manifest.
- `gsm_hard_prompts.jsonl` records the exact prompts sent to the model.
- `gsm_hard_traces.jsonl` records model traces, predictions, token counts, and correctness.
- `gsm_hard_traces_checkpoint.jsonl` protects long API runs from interruption.
- `gsm_hard_errors.jsonl` records API failures by id.
- `run_meta.json` records model, temperature, manifest paths, output path, and run label.

This made the pipeline more verbose on disk, but it improved reproducibility.
When later analysis showed we needed exact model alignment, those files made it
possible to recover the problem ids and rerun missing rows.

### 6. Use model-specific output directories

Earlier versions wrote directly into one output directory. That was simple for a
single model, but unsafe once we started swapping model names and comparing
runs. The current script converts a model such as `gemma4:31b-cloud` into a
safe folder name such as `gemma4-31b-cloud`.

This prevented accidental overwrites and made side-by-side analysis possible.
It also explains the cleaned final layout:

```text
results_gptoss/
results_gemma4/
```

### 7. Preserve manifest order during retries and merges

The commit history shows that retry handling became a real design requirement.
API failures can leave holes in a trace file. A naive retry can append repaired
rows to the end, which breaks line-by-line correspondence with the manifest and
can silently corrupt cross-model comparisons.

The current `merge_traces_aligned_to_manifest()` function loads the manifest,
indexes existing and newly rerun traces by id, overwrites repaired ids, drops
stray ids, and emits rows in manifest order. Missing traces can become explicit
`_stub` rows. This improved correctness of the data pipeline more than raw model
accuracy, because it prevented analysis errors caused by misaligned files.

### 8. Convert free-form traces into dependency graphs

Raw chain-of-thought text is hard to compare because each response has different
wording, length, and formatting. The graph extractor turns each trace into a
common representation: short steps, operation labels, and dependency edges.

This design made the rest of the project possible. It let us compute graph
depth, fact ratio, degree statistics, orphan nodes, and motif flags. The cost is
that the graph is not ground truth. It is an LLM-produced interpretation of
another model's trace. The strongest result, `has_offschema_node`, must be read
with this caveat because an `other` node can mean actual unusual reasoning, a
trace style the extractor struggled with, or a gap in our schema.

### 9. Start with simple interpretable graph features

The feature set favors simple quantities over complex graph embeddings. Counts,
ratios, degree summaries, and unsupported-node flags are easy to inspect and
easy to explain in the paper.

This helped interpretation. For example, incorrect gpt-oss traces averaged 9.54
steps versus 7.49 for correct traces, while incorrect gemma4 traces averaged
9.03 steps versus 7.32. However, simple features can be correlated. Depth and
step count carry overlapping information, so `extended_analysis.py` drops some
redundant predictors before logistic regression.

### 10. Add targeted motifs after inspecting early results

The first motif set was generic: unsupported arithmetic, orphan nodes,
unsupported conclusions, low fact ratio, and long chains. Later we added motifs
based on concrete failure hypotheses:

- `late_arithmetic`: arithmetic deep in the graph without fact ancestors.
- `verbose_ungrounded`: at least 8 steps and less than 30% fact extraction.
- `early_branching`: a high-out-degree node in the first half of the trace.

Only `verbose_ungrounded` replicated clearly across both models. That result was
useful but not fully independent: it combines two already-important signals,
trace length and fact ratio.

### 11. Add additional motifs from visual graph inspection

After drawing graph grids and case studies, we added five more flags:

- `has_offschema_node`
- `orphan_fact`
- `linear_chain`
- `high_fanin_conclude`
- `failed_verification`

This was the most productive motif iteration. `has_offschema_node` became the
strongest predictor in the project: it appeared in about half of incorrect
traces but only a small fraction of correct traces in both models. `failed_verification`
was also informative for interpretation: verification steps appeared more often
in wrong gpt-oss traces, suggesting verification was used reactively on hard
problems rather than as routine quality control.

### 12. Use within-problem pairing as a robustness check

The pooled correct-vs-incorrect analysis cannot fully separate reasoning quality
from problem difficulty. Harder problems can naturally require longer traces and
also produce more wrong answers. The paired script directly addresses that by
keeping only model-disagreement cases.

This changed the technical interpretation of the results. In the pooled data,
both depth and step count looked strong. In the paired analysis, step count had
the clearer signal, while depth looked more confounded by problem difficulty.
That is why the final paper treats step count as a more reliable failure signal
than depth.

## Main Findings From the Committed Results

The detailed tables live in `_reports/results_technical.md`. The most important
numbers are:

- gpt-oss accuracy: 159/200, or 79.5%.
- gemma4 accuracy: 161/200, or 80.5%.
- Incorrect traces have more steps in both models.
- Incorrect traces have greater depth in pooled correct-vs-incorrect analysis.
- Correct traces have higher `frac_facts` in both models.
- `verbose_ungrounded` replicates as a failure-associated motif.
- `has_offschema_node` is the strongest single motif in both models.
- The two models agree on 186/200 problems and disagree on 14.
- In the 14 discordant pairs, `n_steps` is more robust than `depth`.

## Current Artifact Caveats

The cleaned repository stores the final graph-level and CSV-level analysis
artifacts for both models. The gpt-oss raw trace history is split across earlier
and later batches: the full 200-problem gpt-oss analysis should be read from
`results_gptoss/gsm_hard_graphs_all.jsonl` and `results_gptoss/graph_features.csv`.

Some older report text and examples still mention the pre-cleanup `experiments/`
or `gsm_hard_data/` layout. The final committed layout is:

```text
results_gptoss/
results_gemma4/
_reports/
appendix/figures/
paper/
```

## Limitations

The graph extractor is an LLM. We validate JSON structure and dependency ids,
but we do not have a human-labeled gold graph for each trace. A human calibration
set would be the strongest next step.

The analysis uses two models. Cross-model replication is useful, but it is not
enough to claim universality across model families, training methods, or
domains.

The dataset is GSM-Hard only. It isolates arithmetic reasoning well, but the
same graph features may behave differently on law, science, proof, or open-ended
planning tasks.

The paired analysis has only 14 discordant problems. It is valuable because it
controls for problem identity, but it is underpowered and should be interpreted
as robustness evidence rather than a final causal test.

## Code Provenance

The project-specific runtime scripts in the repository were written by the team.
The code depends on external open-source Python packages listed in
`requirements.txt`, the HuggingFace `reasoning-machines/gsm-hard` dataset,
Ollama Cloud models, and the Anthropic API for graph extraction.

The files under `paper/` include AAAI style/template assets (`aaai25.sty`,
`aaai25.bst`, and sample bibliography content). Those are formatting assets for
the paper and are not part of the project runtime pipeline.
