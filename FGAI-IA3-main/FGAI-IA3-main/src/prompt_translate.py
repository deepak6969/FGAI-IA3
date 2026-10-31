"""Few-shot, no-update LLaMA or Gemma translation on the fixed test split."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {"llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
          "gemma": "unsloth/gemma-3-1b-it"}


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def prompt(source, examples):
    shots = "\n".join(f"English: {r['en']}\nFrench: {r['fr']}" for r in examples)
    return ("Translate the final English text into French. Preserve names, numbers, "
            "and meaning. Return only the French translation.\n\n"
            f"Examples:\n{shots}\n\nEnglish: {source}\nFrench:")


def run(model_name, data_dir, output, temperature=0.0, sample_count=None):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this experiment")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train = read(Path(data_dir) / "train.jsonl")
    test = read(Path(data_dir) / "test.jsonl")
    if sample_count is not None:
        test = test[:sample_count]
    # Fixed examples come only from the training split and are shared by both models.
    examples = train[:3]
    model_id = MODELS[model_name]
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float32 if model_name == "gemma" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                                device_map="auto", low_cpu_mem_usage=True).eval()
    load_seconds = time.perf_counter() - load_start
    predictions = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for row in test:
            messages = [{"role": "user", "content": prompt(row["en"], examples)}]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
            args = {"max_new_tokens": 96, "do_sample": temperature > 0,
                    "pad_token_id": tokenizer.eos_token_id}
            if temperature > 0:
                args.update(temperature=temperature, top_p=0.9,
                            remove_invalid_values=True, renormalize_logits=True)
            ids = model.generate(**inputs, **args)
            new_tokens = ids[0, inputs["input_ids"].shape[-1]:]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            predictions.append(prediction)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    suffix = "greedy" if temperature == 0 else f"temp_{temperature:g}"
    with (output / f"predictions_{suffix}.jsonl").open("w", encoding="utf-8") as f:
        for row, prediction in zip(test, predictions):
            f.write(json.dumps({**row, "prediction": prediction}, ensure_ascii=False) + "\n")
    metrics = {"model": model_name, "model_id": model_id, "architecture": "decoder-only",
               "adaptation": "three-shot prompting; no parameter updates",
               "parameters": sum(p.numel() for p in model.parameters()),
               "training_seconds": 0, "load_seconds": load_seconds,
               "inference_seconds": inference_seconds, "test_count": len(test),
               "temperature": temperature, "max_new_tokens": 96,
               "device": torch.cuda.get_device_name(0),
               "few_shot_ids": [r["id"] for r in examples]}
    (output / f"metrics_{suffix}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=MODELS)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--output", default=None)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--sample-count", type=int)
    a = p.parse_args()
    print(json.dumps(run(a.model, a.data_dir, a.output or f"outputs/{a.model}",
                         a.temperature, a.sample_count), indent=2))
