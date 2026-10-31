"""Regenerate comparison plots from verified Kaggle score JSON."""
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    root = Path("results")
    rows = json.loads((root / "scores.json").read_text(encoding="utf-8"))
    names = [r["model"].replace("basic_transformer", "Basic Transformer").replace("llama", "TinyLlama") for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in zip(axes, ("bleu", "chrf"), ("Corpus BLEU", "Corpus chrF")):
        bars = ax.bar(names, [r[key] for r in rows], color="#3b688a")
        ax.bar_label(bars, fmt="%.2f", padding=2)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylim(0, max(r[key] for r in rows) * 1.18)
    fig.tight_layout()
    fig.savefig(root / "quality_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, [r["inference_seconds"] for r in rows], color="#7b6153")
    axes[0].set_ylabel("Inference time for 200 examples (s)")
    axes[1].bar(names, [r["parameters"] / 1e6 for r in rows], color="#7b6153")
    axes[1].set_ylabel("Model parameters (millions)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(root / "compute_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
