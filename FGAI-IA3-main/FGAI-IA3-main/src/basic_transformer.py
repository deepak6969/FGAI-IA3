"""Small Transformer trained from scratch on the shared OPUS Books split."""
import argparse
import json
import math
import time
from pathlib import Path

import sentencepiece as spm
import torch
from torch import nn
from torch.utils.data import DataLoader


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


class BasicTransformer(nn.Module):
    def __init__(self, vocab, d_model=256, nhead=4, layers=2, ff=512, max_len=160):
        super().__init__()
        self.token = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.net = nn.Transformer(d_model=d_model, nhead=nhead,
                                  num_encoder_layers=layers, num_decoder_layers=layers,
                                  dim_feedforward=ff, dropout=0.1, batch_first=True)
        self.out = nn.Linear(d_model, vocab)
        self.scale = math.sqrt(d_model)

    def embed(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.token(x) * self.scale + self.pos(pos)

    def forward(self, source, target):
        mask = nn.Transformer.generate_square_subsequent_mask(target.size(1), device=target.device)
        hidden = self.net(self.embed(source), self.embed(target), tgt_mask=mask,
                          src_key_padding_mask=source.eq(0),
                          tgt_key_padding_mask=target.eq(0),
                          memory_key_padding_mask=source.eq(0))
        return self.out(hidden)


def batchify(items, sp, device, max_len=96):
    def encode(text):
        return [sp.bos_id()] + sp.encode(text)[:max_len - 2] + [sp.eos_id()]
    en = [encode(r["en"]) for r in items]
    fr = [encode(r["fr"]) for r in items]
    def pad(seqs):
        width = max(map(len, seqs))
        return torch.tensor([s + [0] * (width - len(s)) for s in seqs], device=device)
    return pad(en), pad(fr)


@torch.inference_mode()
def translate(model, sp, sources, device, max_len=96):
    src, _ = batchify([{"en": x, "fr": x} for x in sources], sp, device, max_len)
    target = torch.full((len(sources), 1), sp.bos_id(), dtype=torch.long, device=device)
    done = torch.zeros(len(sources), dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        next_token = model(src, target)[:, -1].argmax(-1)
        next_token = torch.where(done, sp.eos_id(), next_token)
        target = torch.cat([target, next_token[:, None]], dim=1)
        done |= next_token.eq(sp.eos_id())
        if bool(done.all()):
            break
    return [sp.decode([t for t in row[1:].tolist() if t not in (0, sp.eos_id())])
            for row in target]


def run(data_dir, output, epochs=5, batch_size=64, seed=4403):
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This full experiment requires a CUDA GPU")
    device = torch.device("cuda")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train = read_jsonl(Path(data_dir) / "train.jsonl")
    val = read_jsonl(Path(data_dir) / "validation.jsonl")
    test = read_jsonl(Path(data_dir) / "test.jsonl")
    corpus = output / "tokenizer_corpus.txt"
    corpus.write_text("\n".join(x[k] for x in train for k in ("en", "fr")), encoding="utf-8")
    prefix = str(output / "shared_bpe")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=prefix,
        vocab_size=4000, model_type="bpe", character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3, hard_vocab_limit=False)
    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    model = BasicTransformer(sp.vocab_size()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    start = time.perf_counter()
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        losses = []
        loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=lambda z: z)
        for items in loader:
            src, tgt = batchify(items, sp, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(src, tgt[:, :-1])
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                                    tgt[:, 1:].reshape(-1), ignore_index=0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.inference_mode():
            for i in range(0, len(val), batch_size):
                src, tgt = batchify(val[i:i + batch_size], sp, device)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(src, tgt[:, :-1])
                    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                                        tgt[:, 1:].reshape(-1), ignore_index=0)
                val_losses.append(loss.item())
        row = {"epoch": epoch + 1, "train_loss": sum(losses) / len(losses),
               "validation_loss": sum(val_losses) / len(val_losses)}
        history.append(row)
        print(row, flush=True)
        if row["validation_loss"] < best_loss:
            best_loss = row["validation_loss"]
            torch.save(model.state_dict(), output / "model.pt")
    training_seconds = time.perf_counter() - start
    model.load_state_dict(torch.load(output / "model.pt", map_location=device, weights_only=True))
    model.eval()
    torch.cuda.synchronize()
    start = time.perf_counter()
    predictions = []
    for i in range(0, len(test), 16):
        predictions += translate(model, sp, [r["en"] for r in test[i:i + 16]], device)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row, prediction in zip(test, predictions):
            f.write(json.dumps({**row, "prediction": prediction}, ensure_ascii=False) + "\n")
    metrics = {"model": "basic_transformer", "architecture": "encoder-decoder",
               "parameters": sum(p.numel() for p in model.parameters()),
               "training_seconds": training_seconds, "inference_seconds": inference_seconds,
               "test_count": len(test), "epochs": epochs, "history": history,
               "decoding": "greedy, max 95 generated tokens", "device": torch.cuda.get_device_name(0)}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--output", default="outputs/basic_transformer")
    p.add_argument("--epochs", type=int, default=5)
    a = p.parse_args()
    print(json.dumps(run(a.data_dir, a.output, a.epochs), indent=2))
