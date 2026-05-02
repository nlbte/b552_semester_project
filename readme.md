# this workspace

this folder is a small pipeline around **gsm-hard** (huggingface `reasoning-machines/gsm-hard`): sample math word problems, run them through **ollama cloud**, save reasoning traces, optionally turn traces into step graphs, summarize runs with plots.

## setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

edit `.env` and set the api key variable shown in `.env.example` (or export the same name in your shell). get a key at https://ollama.com/settings/keys. if `python-dotenv` is installed, `.env` is loaded automatically; otherwise a small built-in parser reads `.env`. if the key is still missing, scripts may prompt once via `getpass`.

## scripts (project root)

| file | what it does |
| --- | --- |
| `gsm_hard_ollama.py` | loads gsm-hard, draws a random sample, sends each question to ollama cloud, writes `gsm_hard_data/gsm_hard_traces.jsonl` (plus prompts, sample slice, checkpoints, errors as needed). |
| `traces_to_graphs.py` | reads `gsm_hard_traces.jsonl`, asks ollama cloud to segment each trace into json `steps` with `depends_on` and `op`, writes `gsm_hard_graphs.jsonl`. supports checkpoint/resume. |
| `motif_analysis.py` | computes 14 structural features per graph (step count, depth, fact ratio, etc.) and runs mann-whitney u tests comparing correct vs. incorrect traces. |
| `extended_analysis.py` | correlation matrix, standardized logistic regression, original 5 motif flags, out-degree breakdown by step type, error magnitude analysis. |
| `stats_tests.py` | fisher's exact test and logistic regression with odds ratios for each feature. |
| `targeted_motifs.py` | tests three hypothesis-driven binary flags: `late_arithmetic`, `verbose_ungrounded`, `early_branching`. |
| `additional_motifs.py` | tests five more binary flags: `has_offschema_node`, `orphan_fact`, `linear_chain`, `high_fanin_conclude`, `failed_verification`. |
| `targeted_motifs_report.py` | loads per-model targeted motif csvs and generates a cross-model markdown report. |
| `additional_motifs_report.py` | loads per-model additional motif csvs and generates a cross-model markdown report. |
| `within_problem_comparison.py` | paired analysis of discordant problems (same problem, different model outcomes); wilcoxon signed-rank test on all 10 features. |
| `visualize_results.py` | reads one or more `*traces*.jsonl` files under `gsm_hard_data` (or paths you pass), prints summary stats and saves a figure (default `gsm_hard_data/summary.png`): accuracy, token histograms, etc. |
| `visualize_graphs.py` | generates 5-column grid visualizations of 20 correct and 20 incorrect reasoning graphs per model. |
| `pretty_jsonl.py` | converts a `.jsonl` file into a single indented `.json` array (default: same basename with `.json`) so nested data is easier to read. |

## data directories

**`gsm_hard_data/`** — gpt-oss:120b results:

- `gsm_hard_traces.jsonl` — one json object per question: model output, extracted answer, correctness, token counts
- `gsm_hard_graphs_all.jsonl` — dag graphs for all 200 problems
- `graph_features.csv` — 14 structural features per graph
- `feature_comparison.csv` — mann-whitney u results: correct vs. incorrect means, medians, p-values
- `logistic_regression.csv` / `logistic_odds_ratios.csv` — regression results
- `motif_scores.csv` — original 5 motif flags per trace
- `targeted_motifs_gptoss.csv` — targeted motif flags per trace
- `additional_motifs_gptoss.csv` — additional motif flags per trace
- `error_split.csv` — per-trace gold/predicted/relative error
- `graph_grids/` — grid figures of correct and incorrect graphs

**`experiments/gemma4-31b-cloud/`** — gemma4:31b results, same structure as above.

**repo root report files:**

- `targeted_motifs_report.md` — cross-model results for the three targeted motifs
- `additional_motifs_report.md` — cross-model results for the five additional motifs
- `within_problem_report.md` — paired analysis of 14 discordant problems
- `results.md` — main findings in plain english
- `results_technical.md` — full technical results with all tables and raw numbers

## example flow

```bash
python gsm_hard_ollama.py --sample-size 50
python traces_to_graphs.py
python motif_analysis.py --graphs gsm_hard_data/gsm_hard_graphs_all.jsonl --traces gsm_hard_data/gsm_hard_traces.jsonl --out gsm_hard_data
python visualize_results.py
python pretty_jsonl.py gsm_hard_data/gsm_hard_graphs_all.jsonl
```