"""Deterministic English-to-French OPUS Books preprocessing."""
import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

DATASET = "Helsinki-NLP/opus_books"
CONFIG = "en-fr"
REVISION = "main"


def clean(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value or "")).strip()


def prepare(rows, train_size=4000, val_size=200, test_size=200, seed=4403):
    """Deduplicate on source, then shuffle; source strings cannot cross splits."""
    unique = {}
    rejected = {"empty_or_length": 0, "length_ratio": 0, "duplicate_source": 0}
    for row in rows:
        pair = row["translation"]
        en, fr = clean(pair["en"]), clean(pair["fr"])
        if not (10 <= len(en) <= 300 and 10 <= len(fr) <= 350):
            rejected["empty_or_length"] += 1
            continue
        ratio = len(fr) / len(en)
        if not 0.45 <= ratio <= 2.2:
            rejected["length_ratio"] += 1
            continue
        key = en.casefold()
        if key in unique:
            rejected["duplicate_source"] += 1
            continue
        uid = hashlib.sha256((en + "\0" + fr).encode()).hexdigest()[:16]
        unique[key] = {"id": uid, "en": en, "fr": fr}
    pairs = list(unique.values())
    random.Random(seed).shuffle(pairs)
    n = train_size + val_size + test_size
    if len(pairs) < n:
        raise ValueError(f"Need {n} clean unique pairs; found {len(pairs)}")
    picked = pairs[:n]
    splits = {
        "train": picked[:train_size],
        "validation": picked[train_size:train_size + val_size],
        "test": picked[train_size + val_size:],
    }
    return splits, {"raw_rows": len(rows), "clean_unique_rows": len(pairs),
                    "rejected": rejected, "seed": seed,
                    "sizes": {k: len(v) for k, v in splits.items()},
                    "source": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}/{CONFIG}",
                    "language_pair": "en-fr", "normalization": "Unicode NFKC and whitespace collapse",
                    "filters": "English 10-300 chars, French 10-350 chars, length ratio 0.45-2.2",
                    "deduplication": "casefolded English source; first occurrence retained"}


def write_splits(splits, manifest, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/processed")
    p.add_argument("--train-size", type=int, default=4000)
    p.add_argument("--val-size", type=int, default=200)
    p.add_argument("--test-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=4403)
    a = p.parse_args()
    from datasets import load_dataset
    rows = load_dataset(DATASET, CONFIG, split="train", revision=REVISION)
    splits, manifest = prepare(rows, a.train_size, a.val_size, a.test_size, a.seed)
    write_splits(splits, manifest, a.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
