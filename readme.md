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
| `visualize_results.py` | reads one or more `*traces*.jsonl` files under `gsm_hard_data` (or paths you pass), prints summary stats and saves a figure (default `gsm_hard_data/summary.png`): accuracy, token histograms, etc. |
| `pretty_jsonl.py` | converts a `.jsonl` file into a single indented `.json` array (default: same basename with `.json`) so nested data is easier to read. |

## data directory: `gsm_hard_data/`

typical artifacts (exact names may vary after runs):

- `gsm_hard_sample.jsonl` — sampled rows from gsm-hard (id, input, target)
- `gsm_hard_prompts.jsonl` — built prompts + gold answers for the run
- `gsm_hard_traces.jsonl` — one json object per question: model output, extracted answer, correctness, token counts
- `gsm_hard_traces_checkpoint.jsonl` — may appear during long `gsm_hard_ollama.py` runs
- `gsm_hard_graphs.jsonl` / `gsm_hard_graphs.json` — structured graphs from `traces_to_graphs.py`; run `pretty_jsonl.py` on the jsonl for a readable `gsm_hard_graphs.json`
- `summary.png` — from `visualize_results.py`

## example flow

```bash
python gsm_hard_ollama.py --sample-size 50
python traces_to_graphs.py
python visualize_results.py
python pretty_jsonl.py gsm_hard_data/gsm_hard_graphs.jsonl
```