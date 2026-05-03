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
| `gsm_hard_ollama.py` | loads gsm-hard, draws a random sample, sends each question to ollama cloud, writes `results_gptoss/gsm_hard_traces.jsonl` (plus prompts, sample slice, checkpoints, errors as needed). |
| `gsm_hard_claude.py` | drop-in replacement for graph extraction using the claude api (claude opus 4.6); same input/output format as the ollama extractor. |
| `motif_analysis.py` | computes 14 structural features per graph (step count, depth, fact ratio, etc.) and runs mann-whitney u tests comparing correct vs. incorrect traces. |
| `extended_analysis.py` | correlation matrix, standardized logistic regression, original 5 motif flags, out-degree breakdown by step type, error magnitude analysis. |
| `stats_tests.py` | fisher's exact test and logistic regression with odds ratios for each feature. |
| `targeted_motifs.py` | tests three hypothesis-driven binary flags: `late_arithmetic`, `verbose_ungrounded`, `early_branching`. |
| `additional_motifs.py` | tests five more binary flags: `has_offschema_node`, `orphan_fact`, `linear_chain`, `high_fanin_conclude`, `failed_verification`. |
| `within_problem_comparison.py` | paired analysis of discordant problems (same problem, different model outcomes); wilcoxon signed-rank test on all 10 features. |
| `visualize_results.py` | reads one or more `*traces*.jsonl` files (or paths you pass), prints summary stats and saves a figure (default `results_gptoss/summary.png`): accuracy, token histograms, etc. |
| `visualize_graphs.py` | generates 5-column grid visualizations of 20 correct and 20 incorrect reasoning graphs per model. |

## data directories

**`results_gptoss/`** — gpt-oss:120b results:

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

**`results_gemma4/`** — gemma4:31b results, same structure as above.

**repo root report files:**

- `results.md` — main findings in plain english
- `results_technical.md` — full technical results with all tables and raw numbers

## example flow

```bash
python gsm_hard_ollama.py --sample-size 50
python gsm_hard_claude.py --traces results_gptoss/gsm_hard_traces.jsonl --out results_gptoss/gsm_hard_graphs_all.jsonl
python motif_analysis.py --graphs results_gptoss/gsm_hard_graphs_all.jsonl --traces results_gptoss/gsm_hard_traces.jsonl --out results_gptoss
python visualize_results.py
```