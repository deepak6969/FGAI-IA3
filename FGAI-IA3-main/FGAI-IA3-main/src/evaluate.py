"""Evaluate aligned predictions from successful Kaggle experiments."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import sacrebleu


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def evaluate(prediction_paths, metric_paths, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    loaded = {name: load_jsonl(path) for name, path in prediction_paths.items()}
    if not loaded:
        raise ValueError("No predictions supplied")
    baseline = next(iter(loaded.values()))
    ids = [row["id"] for row in baseline]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate test IDs")
    rows = []
    for name, items in loaded.items():
        if [r["id"] for r in items] != ids:
            raise ValueError(f"Test order or IDs differ for {name}")
        if [r["fr"] for r in items] != [r["fr"] for r in baseline]:
            raise ValueError(f"References differ for {name}")
        metadata = json.loads(Path(metric_paths[name]).read_text(encoding="utf-8"))
        pred = [r["prediction"] for r in items]
        refs = [r["fr"] for r in items]
        rows.append({"model": name, "bleu": sacrebleu.corpus_bleu(pred, [refs]).score,
                     "chrf": sacrebleu.corpus_chrf(pred, [refs]).score,
                     "parameters": metadata["parameters"],
                     "training_seconds": metadata["training_seconds"],
                     "inference_seconds": metadata["inference_seconds"],
                     "test_count": len(items),
                     "architecture": metadata["architecture"]})
    (output / "scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    samples = []
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if i >= len(baseline):
            break
        samples.append({"id": baseline[i]["id"], "en": baseline[i]["en"],
                        "reference": baseline[i]["fr"],
                        "translations": {name: items[i]["prediction"] for name, items in loaded.items()}})
    (output / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = [r["model"] for r in rows]
    axes[0].bar(names, [r["bleu"] for r in rows])
    axes[0].set_ylabel("Corpus BLEU")
    axes[1].bar(names, [r["chrf"] for r in rows])
    axes[1].set_ylabel("Corpus chrF")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output / "quality_comparison.png", dpi=180)
    plt.close(fig)
    return rows


def evaluate_temperature(greedy_paths, sampled_paths, output):
    """Compare decoding on the same first 30 test items for each decoder."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    samples = []
    for name, sampled_path in sampled_paths.items():
        sampled = load_jsonl(sampled_path)
        greedy = load_jsonl(greedy_paths[name])[:len(sampled)]
        if [r["id"] for r in sampled] != [r["id"] for r in greedy]:
            raise ValueError(f"Temperature sample IDs differ for {name}")
        refs = [r["fr"] for r in sampled]
        for label, items in (("greedy", greedy), ("temperature_0.7", sampled)):
            pred = [r["prediction"] for r in items]
            rows.append({"model": name, "decoding": label, "n": len(items),
                         "bleu": sacrebleu.corpus_bleu(pred, [refs]).score,
                         "chrf": sacrebleu.corpus_chrf(pred, [refs]).score})
        for base, alt in zip(greedy[:5], sampled[:5]):
            samples.append({"model": name, "id": base["id"], "source": base["en"],
                            "reference": base["fr"], "greedy": base["prediction"],
                            "temperature_0.7": alt["prediction"]})
    (output / "temperature_scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output / "temperature_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prediction", action="append", required=True,
                   help="model=path; repeat for every successful model")
    p.add_argument("--metric", action="append", required=True,
                   help="model=path; repeat for every successful model")
    p.add_argument("--output", default="results/evaluation")
    a = p.parse_args()
    parse = lambda values: dict(v.split("=", 1) for v in values)
    print(json.dumps(evaluate(parse(a.prediction), parse(a.metric), a.output), indent=2))
