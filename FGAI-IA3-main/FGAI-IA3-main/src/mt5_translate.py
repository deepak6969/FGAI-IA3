"""Fine-tune mT5-small and translate the fixed English-French test split."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "google/mt5-small"
PROMPT = "translate English to French: "


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def run(data_dir, output, epochs=1, batch_size=8, seed=4403):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this experiment")
    torch.manual_seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train, val, test = (read(Path(data_dir) / (name + ".jsonl"))
                        for name in ("train", "validation", "test"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).cuda()
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    def collate(rows):
        enc = tokenizer([PROMPT + r["en"] for r in rows], truncation=True,
                        max_length=128, padding=True, return_tensors="pt")
        labels = tokenizer(text_target=[r["fr"] for r in rows], truncation=True,
                           max_length=128, padding=True, return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        return {**{k: v.cuda() for k, v in enc.items()}, "labels": labels.cuda()}

    history = []
    start = time.perf_counter()
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        losses = []
        loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            # mT5 can overflow in fp16 on T4; float32 keeps the loss finite.
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite mT5 loss at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        vals = []
        with torch.inference_mode():
            for batch in DataLoader(val, batch_size=batch_size, collate_fn=collate):
                vals.append(model(**batch).loss.item())
        row = {"epoch": epoch + 1, "train_loss": sum(losses) / len(losses),
               "validation_loss": sum(vals) / len(vals)}
        print(row, flush=True)
        history.append(row)
        if row["validation_loss"] < best:
            best = row["validation_loss"]
            model.save_pretrained(output / "checkpoint", safe_serialization=True)
            tokenizer.save_pretrained(output / "checkpoint")
    training_seconds = time.perf_counter() - start
    model = AutoModelForSeq2SeqLM.from_pretrained(output / "checkpoint", torch_dtype=torch.float32).cuda().eval()
    torch.cuda.synchronize()
    start = time.perf_counter()
    predictions = []
    with torch.inference_mode():
        for i in range(0, len(test), batch_size):
            enc = tokenizer([PROMPT + r["en"] for r in test[i:i + batch_size]],
                            truncation=True, max_length=128, padding=True,
                            return_tensors="pt").to("cuda")
            ids = model.generate(**enc, max_new_tokens=96, do_sample=False,
                                 num_beams=1)
            predictions += tokenizer.batch_decode(ids, skip_special_tokens=True)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row, prediction in zip(test, predictions):
            f.write(json.dumps({**row, "prediction": prediction}, ensure_ascii=False) + "\n")
    metrics = {"model": "mt5", "model_id": MODEL_ID, "architecture": "encoder-decoder",
               "parameters": sum(p.numel() for p in model.parameters()),
               "training_seconds": training_seconds, "inference_seconds": inference_seconds,
               "test_count": len(test), "history": history, "epochs": epochs,
               "decoding": "greedy, max 96 new tokens", "device": torch.cuda.get_device_name(0)}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--output", default="outputs/mt5")
    p.add_argument("--epochs", type=int, default=1)
    a = p.parse_args()
    print(json.dumps(run(a.data_dir, a.output, a.epochs), indent=2))
