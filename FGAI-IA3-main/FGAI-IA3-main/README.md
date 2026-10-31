# English to French machine translation comparison

## Problem statement and objective

ICT4403 Foundations of Generative AI, Assignment 3 asks for a comparison of encoder decoder and decoder-only machine translation systems. This project translates English literary sentences into French with a small Transformer trained from scratch, fine tuned mT5, and prompted LLaMA architecture and Gemma decoders. Its objective is to evaluate translation quality and computational cost on one held-out test split, inspect generated text, and identify responsible-use risks.

## Dataset and preparation

The source is [OPUS Books English French](https://huggingface.co/datasets/Helsinki-NLP/opus_books/tree/main/en-fr), an aligned literary corpus hosted by Helsinki NLP. The dataset card permits personal, educational, and research use and restricts commercial redistribution. The original corpus is not copied into Git. The exact 4,400-pair processed subset is included in `data/processed/` with attribution in `data/README.md`. The [Kaggle preprocessing run](https://www.kaggle.com/code/eltonlewis/ict4403-english-french-data-preparation) saved the same split files in its version 2 output.

The script normalizes Unicode to NFKC, collapses whitespace, removes very short or long pairs and extreme length ratios, and keeps the first translation for each casefolded English source. A seeded shuffle (4403) then selects 4,400 pairs and partitions them into 4,000 training, 200 validation, and 200 test examples. This prevents identical English source strings from crossing the splits. The Kaggle run reported 127,085 raw rows, 117,720 clean unique rows, and the stated split sizes. See `src/prepare_data.py` for the exact filters.

## Models implemented

| System            | Architecture                    | Adaptation                                   | Checkpoint                                                                       |
| ----------------- | ------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------- |
| Basic Transformer | encoder decoder                 | trained from scratch, shared 4,000 piece BPE | local 2 layer model                                                              |
| mT5               | encoder decoder                 | full fine tuning, one epoch                  | [google/mt5-small](https://huggingface.co/google/mt5-small)                      |
| TinyLlama         | decoder only LLaMA architecture | three shot prompting, no weight updates      | [TinyLlama 1.1B Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| Gemma             | decoder only                    | three shot prompting, no weight updates      | [Unsloth Gemma 3 1B IT](https://huggingface.co/unsloth/gemma-3-1b-it)            |

Meta LLaMA 3.2 1B Instruct and Google's original Gemma 3 270M IT both returned HTTP 401 in Kaggle because their weights require access and authentication. TinyLlama shares LLaMA 2 architecture and tokenizer; the Unsloth Gemma checkpoint is a hosted copy of Gemma 3 1B IT. These exact substitutions are named in the measured results. An initial Gemma 270M run produced empty decoded translations; its diagnostic scores and samples are retained separately in `results/`.

## Dependencies and execution steps

Use Python 3.10 or newer with a CUDA GPU. `requirements.txt` lists PyTorch, Transformers, Datasets, SentencePiece, SacreBLEU, Accelerate, pandas, and Matplotlib. Install it, then run the following commands in Bash. The preparation command regenerates the included deterministic split and can be skipped if you use `data/processed/` directly:

```bash
python -m pip install -r requirements.txt
python -m src.prepare_data --output data/processed
python -m src.basic_transformer --data-dir data/processed --output outputs/basic_transformer
python -m src.mt5_translate --data-dir data/processed --output outputs/mt5
python -m src.prompt_translate llama --data-dir data/processed --output outputs/llama
python -m src.prompt_translate gemma --data-dir data/processed --output outputs/gemma
python -m src.evaluate \
  --prediction basic_transformer=outputs/basic_transformer/predictions.jsonl \
  --prediction mt5=outputs/mt5/predictions.jsonl \
  --prediction llama=outputs/llama/predictions_greedy.jsonl \
  --prediction gemma=outputs/gemma/predictions_greedy.jsonl \
  --metric basic_transformer=outputs/basic_transformer/metrics.json \
  --metric mt5=outputs/mt5/metrics.json \
  --metric llama=outputs/llama/metrics_greedy.json \
  --metric gemma=outputs/gemma/metrics_greedy.json
```

The Kaggle MCP batch notebook API did not attach a prior notebook's output to `/kaggle/input`. Each training notebook therefore prepares the deterministic split from the same source and seed. The evaluation code checks exact row IDs and references before comparing predictions. Kaggle model run links are recorded in the results section.

## Executed notebooks and checkpoints

`notebooks/01_preprocess.ipynb` through `notebooks/05_gemma_1b.ipynb` contain the saved Kaggle script source and execution logs for the five completed phases. They were assembled from the Kaggle MCP source and run log because the Kaggle runs were saved as Python scripts. `notebooks/06_evaluation.ipynb` displays the verified local result files and plots with executed outputs. These copies let an evaluator inspect the run evidence even if the original Kaggle notebooks remain private.

The trained weights and tokenizers are published as public Kaggle model variations: [basic Transformer](https://www.kaggle.com/models/eltonlewis/ict4403-en-fr-translation-checkpoints/PyTorch/basic-transformer) and [fine tuned mT5](https://www.kaggle.com/models/eltonlewis/ict4403-en-fr-translation-checkpoints/PyTorch/mt5-small-finetuned). The mT5 weights are 1.20 GB, above GitHub's ordinary 100 MB per-file limit, so the repository includes download instructions and SHA-256 hashes in `checkpoints/README.md` and `checkpoints/manifest.json`. Downloading these published checkpoints avoids repeating training. The pretrained TinyLlama and Gemma weights are linked in the model table above; no adapted weights were created for the prompted decoders.

## Results obtained

The following values are from completed Kaggle Tesla T4 runs and the [final evaluation notebook, version 2](https://www.kaggle.com/code/eltonlewis/ict4403-final-translation-evaluation). The notebook checked the IDs and references of all 200 examples before computing corpus scores. Machine readable results and ten paired examples are in `results/scores.json` and `results/samples.json`. Full predictions retrieved from the Kaggle outputs are in `results/predictions/`, and the 30-example sampled outputs are in `results/temperature_predictions/`. The plots in `results/` were regenerated locally from the verified scores with `python -m src.plot_results`.

| Model               |    Parameters | Train time | Inference time for 200 examples |      BLEU |      chrF |
| ------------------- | ------------: | ---------: | ------------------------------: | --------: | --------: |
| Basic Transformer   |     4,729,760 |    12.89 s |                          5.79 s |      0.42 |      9.50 |
| mT5 small           |   300,176,768 |   185.56 s |                         33.47 s |      4.63 |     23.24 |
| TinyLlama 1.1B Chat | 1,100,048,384 |       none |                        674.44 s |      1.83 |     23.55 |
| Gemma 3 1B IT       |   999,885,952 |       none |                        417.71 s | **14.11** | **42.83** |

- [Basic Transformer Kaggle run](https://www.kaggle.com/code/eltonlewis/ict4403-basic-transformer-english-french), version 3. Five epochs; validation loss fell from 6.27 to 5.38. The output contains `model.pt`, the BPE tokenizer, predictions, and metrics.
- [mT5 Kaggle run](https://www.kaggle.com/code/eltonlewis/ict4403-mt5-english-french), version 3. One epoch; training loss 5.44 and validation loss 2.78. The output contains the fine tuned checkpoint, predictions, and metrics.
- [TinyLlama Kaggle run](https://www.kaggle.com/code/eltonlewis/ict4403-llama-english-french), version 3.
- [Gemma 1B Kaggle run](https://www.kaggle.com/code/eltonlewis/ict4403-gemma-1b-english-french), version 1. The [270M diagnostic run](https://www.kaggle.com/code/eltonlewis/ict4403-gemma-translation-trial) saved 200 empty greedy predictions; its temperature sampling required a separate float32 run. The diagnostic score files remain in `results/` and are not part of the four-model final ranking.

## Model comparison and evaluation

All quality scores use corpus BLEU and corpus chrF from SacreBLEU, with the same 200 references, greedy decoding, and up to 96 generated tokens. Training time excludes preprocessing and download. Inference time excludes model loading. Parameters count all model parameters, including frozen pretrained parameters. Basic Transformer and mT5 inference used batches of 16 and 8, while decoder inference used one example at a time, so the runtime column describes these implementations rather than an architecture-only speed difference. Basic Transformer and TinyLlama used fp16 on T4. mT5 used float32 after fp16 produced nonfinite loss, and Gemma 1B used float32 for stable sampling. The 270M output failure cannot be attributed to precision alone from these runs.

Gemma 1B is best on both automatic quality metrics. Its prompted approach required no task-specific weight update, but used almost one billion pretrained parameters and 418 seconds of sequential inference. mT5 was the strongest encoder decoder model at BLEU 4.63, although one epoch of fine tuning on 4,000 literary pairs left obvious semantic errors. The 4.73 million parameter scratch model was cheap to train but frequently repeated tokens. TinyLlama also repeated prompt examples and produced irrelevant text. These results reflect differences in pretraining, model size, tokenizer, prompt adherence, and adaptation method as well as encoder decoder versus decoder-only design; they do not isolate the effect of architecture alone.

## Sampling temperature analysis

The decoder notebooks ran the same first 30 test examples at temperature 0.7 with top p 0.9 after the greedy pass. Temperature 0 means greedy decoding. On those 30 examples, TinyLlama fell from BLEU 1.81 / chrF 24.70 to BLEU 0.98 / chrF 22.93 under sampling. Gemma changed from BLEU 13.08 / chrF 42.62 to BLEU 12.36 / chrF 42.64. See `results/temperature_scores.json` and `results/temperature_samples.json`. For one source, “At low water the way will be open,” Gemma's sampled wording “En basse mer, le chemin sera ouvert” was closer to the reference than its greedy wording “À faible tide, le chemin sera ouvert.” This one improvement does not overturn the aggregate BLEU decline. Sampling also changed Holmes's “drily” to “sarcastique,” which may overstate tone. The temperature run is a single stochastic sample; repeat runs would be needed to estimate variance.

## Responsible use

The saved examples show concrete fidelity failures. TinyLlama inserted unrelated training examples after translating a Holmes sentence, an instance of hallucinated material and prompt leakage. On “I don't want to die!”, mT5 returned “Je ne suis pas peur!”, changing a statement about dying into a different statement about fear; this is unsafe for sensitive or urgent text. The scratch Transformer often repeated pronouns instead of conveying meaning. Gemma translated “low water” as “faible tide” in one greedy example, mixing English and French rather than using the cultural and nautical term “mer basse.” These are observations from the saved samples, not a measured bias rate. Literary training data can overrepresent older writing, social roles, and European cultural references; a dedicated audit is needed to test whether gender, formality, names, or dialects are translated differently across groups. BLEU and chrF compare with one reference but do not detect all such harms.

Before deployment, evaluate stratified examples for gender, dialect, named entities, idioms, and sensitive language with bilingual human reviewers. Use constrained terminology and checks for names, numbers, and omitted or added content where accuracy matters. Flag unusually long outputs and prompt echoes for review, and abstain or escalate when the text is medical, legal, or safety critical. Protect private inputs, document known limitations, and retain a human review path.

## Conclusion

Gemma 3 1B IT is the best-performing model in this experiment because it has the highest BLEU and chrF scores on the shared 200-example test set. It requires far more parameters and inference time than the small Transformer and was evaluated with prompting rather than task-specific fine tuning, so the result should be interpreted as a comparison of these complete systems.
