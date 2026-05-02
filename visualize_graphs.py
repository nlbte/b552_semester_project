"""Grid visualizations of reasoning graphs — 20 correct + 20 incorrect per model.

Outputs (in --out-dir):
    ollama_correct_20.png
    ollama_incorrect_20.png
    gemma4_correct_20.png
    gemma4_incorrect_20.png

Usage:
    python visualize_graphs.py
    python visualize_graphs.py --n 20 --out-dir gsm_hard_data
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

OP_COLORS = {
    "extract_fact": "#4e9af1",
    "arithmetic":   "#f4a261",
    "substitute":   "#a8b5a0",
    "conclude":     "#e76f51",
    "verify":       "#e5c07b",
    "other":        "#aaaaaa",
    "unknown":      "#cccccc",
}


def load_graphs(path: str) -> list[dict]:
    graphs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                graphs.append(json.loads(line))
    return graphs


def build_nx(steps: list[dict]) -> nx.DiGraph:
    g = nx.DiGraph()
    for s in steps:
        g.add_node(s["id"], op=s.get("op", "unknown"), text=s.get("text", ""))
    for s in steps:
        for dep in s.get("depends_on", []) or []:
            if dep in g.nodes and dep != s["id"]:
                g.add_edge(dep, s["id"])
    return g


def draw_graph(g_data: dict, ax, title: str) -> None:
    steps = g_data.get("steps", [])
    g = build_nx(steps)
    if len(g) == 0:
        ax.text(0.5, 0.5, "empty", ha="center", va="center", fontsize=7)
        ax.set_title(title, fontsize=7)
        ax.axis("off")
        return

    try:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    except Exception:
        try:
            pos = nx.planar_layout(g)
        except Exception:
            pos = nx.spring_layout(g, seed=42)

    node_colors = [OP_COLORS.get(g.nodes[n].get("op", "unknown"), "#cccccc") for n in g.nodes]
    labels = {n: f"{n}\n{g.nodes[n].get('op', '?')[:4]}" for n in g.nodes}

    nx.draw(g, pos, ax=ax, labels=labels, node_color=node_colors,
            node_size=350, font_size=5, arrows=True,
            arrowstyle="-|>", arrowsize=10, edge_color="#555555",
            width=1.0, with_labels=True)
    ax.set_title(title, fontsize=7, pad=3)


def make_grid(graphs: list[dict], n: int, label: str, out_path: Path, seed: int = 42) -> None:
    random.seed(seed)
    sample = random.sample(graphs, min(n, len(graphs)))

    ncols = 5
    nrows = (len(sample) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
    axes = np.array(axes).flatten()

    for i, g_data in enumerate(sample):
        qid = g_data.get("question_id", f"?{i}")
        steps = len(g_data.get("steps", []))
        title = f"{qid}\n({steps} steps)"
        draw_graph(g_data, axes[i], title)

    for j in range(len(sample), len(axes)):
        axes[j].axis("off")

    legend_patches = [mpatches.Patch(color=c, label=op)
                      for op, c in OP_COLORS.items() if op != "unknown"]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(legend_patches),
               fontsize=7, title="op type")
    fig.suptitle(label, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}  ({len(sample)} graphs)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ollama-graphs",  default="gsm_hard_data/gsm_hard_graphs_all.jsonl")
    ap.add_argument("--gemma4-graphs",  default="experiments/gemma4-31b-cloud/gsm_hard_graphs.jsonl")
    ap.add_argument("--out-dir",        default="gsm_hard_data/graph_grids")
    ap.add_argument("--n",              type=int, default=20)
    ap.add_argument("--seed",           type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading ollama graphs...")
    ollama = load_graphs(args.ollama_graphs)
    ollama_correct   = [g for g in ollama if g.get("is_correct")]
    ollama_incorrect = [g for g in ollama if not g.get("is_correct")]
    print(f"  ollama: {len(ollama_correct)} correct, {len(ollama_incorrect)} incorrect")

    print("loading gemma4 graphs...")
    gemma4 = load_graphs(args.gemma4_graphs)
    gemma4_correct   = [g for g in gemma4 if g.get("is_correct")]
    gemma4_incorrect = [g for g in gemma4 if not g.get("is_correct")]
    print(f"  gemma4: {len(gemma4_correct)} correct, {len(gemma4_incorrect)} incorrect")

    n = args.n
    make_grid(ollama_correct,   n, f"Ollama — correct ({len(ollama_correct)} total, showing {min(n,len(ollama_correct))})",
              out_dir / "ollama_correct_20.png", args.seed)
    make_grid(ollama_incorrect, n, f"Ollama — incorrect ({len(ollama_incorrect)} total, showing {min(n,len(ollama_incorrect))})",
              out_dir / "ollama_incorrect_20.png", args.seed)
    make_grid(gemma4_correct,   n, f"Gemma4 — correct ({len(gemma4_correct)} total, showing {min(n,len(gemma4_correct))})",
              out_dir / "gemma4_correct_20.png", args.seed)
    make_grid(gemma4_incorrect, n, f"Gemma4 — incorrect ({len(gemma4_incorrect)} total, showing {min(n,len(gemma4_incorrect))})",
              out_dir / "gemma4_incorrect_20.png", args.seed)

    print(f"\ndone — all grids in {out_dir}/")


if __name__ == "__main__":
    main()
