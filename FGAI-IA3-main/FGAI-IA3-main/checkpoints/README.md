# Published checkpoints

The two trained models and their tokenizers are public Kaggle model variations. These are the weights produced by the reported Kaggle training runs. Downloading them does not require retraining. The repository keeps a [file manifest](manifest.json) with sizes and SHA-256 hashes recorded during the publication runs.

| Model | Public checkpoint | KaggleHub handle |
|---|---|---|
| Basic Transformer | [Version 1](https://www.kaggle.com/models/eltonlewis/ict4403-en-fr-translation-checkpoints/PyTorch/basic-transformer/1) | `eltonlewis/ict4403-en-fr-translation-checkpoints/pyTorch/basic-transformer` |
| mT5 small | [Version 1](https://www.kaggle.com/models/eltonlewis/ict4403-en-fr-translation-checkpoints/PyTorch/mt5-small-finetuned/1) | `eltonlewis/ict4403-en-fr-translation-checkpoints/pyTorch/mt5-small-finetuned` |

```python
import kagglehub

basic_dir = kagglehub.model_download(
    "eltonlewis/ict4403-en-fr-translation-checkpoints/pyTorch/basic-transformer/1"
)
mt5_dir = kagglehub.model_download(
    "eltonlewis/ict4403-en-fr-translation-checkpoints/pyTorch/mt5-small-finetuned/1"
)
print(basic_dir, mt5_dir)
```

The basic checkpoint contains `model.pt`, `shared_bpe.model`, and `shared_bpe.vocab`. Load it with `BasicTransformer` and `translate` in `src/basic_transformer.py`. The mT5 directory contains `model.safetensors` and tokenizer/configuration files; load it with `AutoModelForSeq2SeqLM.from_pretrained(mt5_dir)` and `AutoTokenizer.from_pretrained(mt5_dir)`. The exact processed test split is in `data/processed/test.jsonl`, and saved predictions are in `results/predictions/`.

GitHub's ordinary Git file limit is 100 MB. The 1.20 GB mT5 weight file therefore lives in the public Kaggle model artifact rather than in the repository. Both public checkpoint variations were verified through the Kaggle MCP model listing after upload. The original training notebooks may remain private, so their source and execution logs are also preserved in `notebooks/`.
