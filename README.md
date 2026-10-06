# MLLM-VisionEncoder-Eval

## Introduction

Given a set of vision encoders and a target LLM, how to select the encoder best suited to that LLM for MLLM training? Training and evaluating an MLLM for every candidate is expensive.

MLLM-VisionEncoder-Eval provides a unified framework for this selection problem, bringing together encoder evaluation metrics, baseline comparisons, and downstream MLLM training and evaluation.

## Our Method: RAVEL

<h3 align="center">A Strong Baseline for Evaluating Vision Encoders<br>in Multimodal Large Language Models</h3>

<p align="center">
  Yilin Yang<sup>*</sup>, Jun-Tao Tang<sup>*</sup>, Kengyi Wang, Siyuan Su, Gaoyong Luo, Mingda Chen<sup>&dagger;</sup>
</p>

<p align="center">
  <sub>Shanghai Jiao Tong University &middot; Nanjing University &middot; Fudan University &middot; Independent Researcher</sub><br>
  <sub><sup>*</sup> Equal contribution. <sup>&dagger;</sup> Corresponding author.</sub>
</p>

<div align="center">
  <a href="https://arxiv.org/abs/2610.05413"><img src="https://img.shields.io/badge/arXiv-2610.05413-B31B1B?style=flat-square" alt="Paper on arXiv"></a>
  <a href="https://huggingface.co/336labs/VisionEncoder-to-MLLM-ModelZoo"><img src="https://img.shields.io/badge/Hugging%20Face-Checkpoints-FFD21E?style=flat-square" alt="Model checkpoints on Hugging Face"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/BibTeX-Citation-3776AB?style=flat-square" alt="BibTeX citation"></a>
</div>

RAVEL ranks vision encoders by comparing nearest-neighbor structures in visual and textual representation spaces, enabling training-free assessment that predicts downstream MLLM performance using only paired image-text representations, with no downstream labels or MLLM training required.


<p align="center">
  <strong>70 vision encoders &middot; 3 language models &middot; 210 MLLMs &middot; 9 evaluation metrics</strong>
</p>

## Requirements

**Environment.** Python 3.10+. From the repository root:

```bash
bash scripts/setup.sh
.venv/bin/python -m pip install -e '.[paper]'
bash scripts/reproduce.sh --init-local --data-root /path/to/vision_encoder_eval_data
```

Edit `configs/local.yaml` ([template](configs/local.example.yaml)) to set your dataset, weight, feature, and worker Python paths. Initialization creates the path configuration; it does not download datasets or generate feature caches. Baseline and MLLM workers have separate dependencies; use [scripts/setup_worker.sh](scripts/setup_worker.sh) for the worker you need.

### Datasets

Download only the data needed for your experiment:

| Dataset | Used By | Download |
| --- | --- | --- |
| LLaVA-LCS-558K | RAVEL, MLLM pretraining, Law A-score | [Annotations and images](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main) |
| LLaVA-1.5 mix665K | MLLM finetuning | [Annotation JSON](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), [image download and layout instructions](https://github.com/haotian-liu/LLaVA/blob/main/docs/Train.md) |
| ImageNet-1K | kNN, linear probing, zero-shot | [Official download](https://image-net.org/download.php) |
| COCO / Karpathy split | Alignment probing, caption evaluation | [COCO images](https://cocodataset.org/#download), [Karpathy annotations](https://cs.stanford.edu/people/karpathy/deepimagesent/) |
| CC3M | Alignment probing | [Official image URLs and captions](https://ai.google.com/research/ConceptualCaptions/) |
| SPair-71k | Law C-score | [Images and pair annotations](https://cvlab.postech.ac.kr/research/SPair-71k/) |
| TokBench | Tokenizer reconstruction evaluation | [Images and annotations](https://huggingface.co/datasets/Junfeng5/TokBench/tree/main), [preparation instructions](https://github.com/wjf5203/TokBench) |

For MLLM training, set `paths.datasets` to a directory with this layout:

```text
datasets/
  instructions/pretrain/blip_laion_cc_sbu_558k.json
  instructions/finetune/llava_v1_5_mix665k_drop_ge8kchars.json
  images/pretrain/                 # extracted LCS images, preserving subdirectories
  images/finetune/                 # coco/, gqa/, ocr_vqa/, textvqa/, vg/
```

The default [data configuration](configs/mllm/data.yaml) uses a filtered mix665K manifest that excludes conversations with at least 8,000 text characters. Prepare this manifest from the downloaded JSON before training and retain its relative image paths. CC3M provides image URLs; the images must be downloaded separately. ImageNet requires `train/` and `val/`; linear probing also requires DINOv2's `extra/` metadata ([instructions](https://github.com/facebookresearch/dinov2#data-preparation)).

MLLM evaluation uses the following splits from [configs/mllm/eval_datasets.yaml](configs/mllm/eval_datasets.yaml):

| Benchmark | Download |
| --- | --- |
| MMMU_TEST | [VLMEvalKit TSV](https://opencompass.openxlab.space/utils/VLMEval/MMMU_TEST.tsv) |
| MMBench_TEST_EN_V11 | [VLMEvalKit TSV](https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_EN_V11.tsv) |
| VQAv2_VAL | [Official images, questions, and annotations](https://visualqa.org/download.html) |
| ScienceQA_VAL | [VLMEvalKit TSV](https://opencompass.openxlab.space/utils/benchmarks/ScienceQA/ScienceQA_VAL.tsv) |
| ChartQA_TEST | [VLMEvalKit TSV](https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv) |
| DocVQA_VAL | [VLMEvalKit TSV](https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv) |
| TextVQA_VAL | [VLMEvalKit TSV](https://opencompass.openxlab.space/utils/VLMEval/TextVQA_VAL.tsv) |
| POPE | [VLMEvalKit TSV](https://opencompass.openxlab.space/utils/VLMEval/POPE.tsv) |
| GQA_TestDev_Balanced | [VLMEvalKit TSV](https://opencompass.openxlab.space/utils/VLMEval/GQA_TestDev_Balanced.tsv) |
| MSCOCO_KARPATHY_TEST | [Karpathy annotations](https://cs.stanford.edu/people/karpathy/deepimagesent/), [COCO images](https://cocodataset.org/#download) |
| FLICKR30K_KARPATHY_TEST | [Karpathy annotations and image source](https://cs.stanford.edu/people/karpathy/deepimagesent/) |

Set `paths.lmudata` to your prepared LMUData directory, containing `<benchmark>.tsv` and `images/<benchmark>/`. Downloaded TSVs may embed images; materialize those images using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) before running this project's evaluation. For VQAv2 and the two caption datasets, convert the source annotations to the same TSV layout, including sample IDs, image paths, questions, and reference answers/captions. The evaluation runner expects these local files to be prepared already.

### Feature Caches

Feature caches are computed locally by running each pretrained vision encoder over the selected dataset and saving its output features. Prepare the dataset and encoder weights first, extract the features once, then reuse them across evaluation runs. Feature extraction settings follow the supplied default configurations.

For RAVEL, extract visual patch features from 1,000 LCS-558K image-text pairs sampled with seed 42, and encode the corresponding captions with the target LLM. Save both feature arrays with the same sample order and retain the sample manifest. `ravel` evaluates these saved visual and text features; `ravel_paper` uses the prepared visual features for all 70 encoders and automatically extracts and caches caption features for the three LLMs in `runs/ravel_paper_features/`.

Set the generated cache paths in `configs/local.yaml`:

```yaml
paths:
  ravel_patches_dir: /path/to/ravel/patches
  ravel_text_dir: /path/to/ravel/text
  ravel_sample_manifest: /path/to/alignment_sample_n1000_seed42.json
```

Keep each export's accompanying JSON metadata and the ordered sample manifest. The runner checks that visual and textual features refer to the same samples. To use your own extracted arrays directly, see the RAVEL command below.

For kNN, run each vision encoder over the shared 200,000-image ImageNet training subset and save its features together with labels, source indices, and per-model metadata. Set `paths.knn_exports` to the resulting export directory ([layout](src/workers/knn/feature_exports/README.md)).

### Weights

RAVEL text backbones: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), and [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct). `ravel_paper` downloads its pinned versions automatically. Encoder and MLLM weight paths are configured through `configs/local.yaml` and the [MLLM recipes](configs/mllm/). For evaluation without training, download the [released MLLM checkpoints](https://huggingface.co/336labs/VisionEncoder-to-MLLM-ModelZoo) and set `paths.trained`.

## Running

List available experiments, run a baseline, or train and evaluate an MLLM:

```bash
bash scripts/reproduce.sh --list
bash scripts/reproduce.sh knn
bash scripts/reproduce.sh mllm_train
bash scripts/reproduce.sh mllm_eval
```

After preparing the caches above, run RAVEL (defaults to CLIP-L/14 + Qwen2.5), or evaluate all 70 encoders with all 3 LLMs:

```bash
bash scripts/reproduce.sh ravel --check
bash scripts/reproduce.sh ravel
bash scripts/reproduce.sh ravel_paper --check
bash scripts/reproduce.sh ravel_paper
```

For your own exported features, pass a visual patch array `[N,T,D]`, a text array `[N,D]`, and a JSON list of sample IDs in their shared row order:

```bash
bash scripts/reproduce.sh ravel --patches /path/to/patches.npy \
  --text /path/to/text.npy --sample-ids /path/to/sample_ids.json
```

Add `--check` to inspect required paths and dependencies. Results are saved in `runs/`. The paper's downstream MLLM scores are bundled in [src/resources/ground_truth.json](src/resources/ground_truth.json), so computing correlations with those scores does not require retraining the MLLMs.

## Evaluate Your Own Metric

Use your metric to score encoder compatibility with each target LLM, then compare its predictions with our **210 trained MLLMs**. The repository bundles the [ground truth](src/resources/ground_truth.json): scores on all 11 downstream tasks and the original paper average for every encoder/LLM pair. No MLLM retraining or checkpoint download is needed to calculate correlations.

### 1. Get the Ground Truth and Prediction Template

Install the CPU-only evaluation dependencies in your environment:

```bash
python -m pip install -e '.[benchmark]'
vision-encoder-eval data ground-truth
vision-encoder-eval benchmark export-ground-truth --output ground_truth.csv
vision-encoder-eval benchmark template --output predictions.csv
```

The template contains all 210 canonical pairs with an empty `score` column. Fill that column with your metric's predictions, preserving `encoder_id` and `llm`:

```csv
encoder_id,llm,score
clip_openai__l14,qwen3,0.73
clip_openai__l14,qwen25,0.81
clip_openai__l14,smollm2,0.65
```

These three values illustrate the format; the full evaluation requires all 210 rows. LLM IDs are `qwen3` (Qwen3-1.7B), `qwen25` (Qwen2.5-1.5B-Instruct), and `smollm2` (SmolLM2-1.7B-Instruct). Scores are matched by IDs, so row order does not matter. An encoder-only metric can repeat its score across the three LLM rows.

### 2. Calculate Correlations

```bash
vision-encoder-eval benchmark evaluate \
  --predictions predictions.csv --metric my_metric --output runs/my_metric
```

This writes `correlations.json` and `correlations.csv`, reporting **Spearman and Pearson separately for each LLM over its 70 encoders**, against both the original 11-task average and each individual task. Reports include prediction and ground-truth hashes. Missing or duplicate pairs, unknown IDs, nonfinite scores, and constant predictions are rejected.

Higher scores indicate better predicted performance by default. For a distance or error metric, add `--direction lower`. To reuse an existing CSV column, such as RAVEL's `ravel_score`, add `--score-column ravel_score`.

To evaluate only one LLM, select it when generating the template and evaluating the resulting 70 predictions:

```bash
vision-encoder-eval benchmark template --llm qwen25 --output predictions_qwen25.csv
vision-encoder-eval benchmark evaluate \
  --predictions predictions_qwen25.csv --llm qwen25 \
  --metric my_metric --output runs/my_metric_qwen25
```

### 3. Connect a Python Metric

Implement an importable function `predict(*, encoder_id, llm)` returning one finite numeric score. The runner calls it for each canonical pair and saves `predictions.csv` before evaluation:

```bash
PYTHONPATH=. vision-encoder-eval benchmark run \
  --metric my_metric:predict --output runs/my_metric
```

The function receives the two IDs; you control feature extraction, model loading, and metric computation. See [the runnable paired-feature example](src/examples/custom_metric.py), which verifies image/text sample alignment and uses RSA as an example scoring function:

```bash
VEE_METRIC_FEATURES=/path/to/paired_features \
  vision-encoder-eval benchmark run \
  --metric vision_encoder_eval.examples.custom_metric:predict --output runs/custom_rsa
```

You can also evaluate predictions directly from Python:

```python
from vision_encoder_eval.benchmark import evaluate_predictions, prediction_pairs

rows = [{**pair, "score": my_metric(**pair)} for pair in prediction_pairs()]
report = evaluate_predictions(rows, metric="my_metric")
print(report["per_llm"]["qwen25"]["spearman"])
```

For training-free comparisons, compute predictions without downstream ground-truth labels. For methods fitted to labels, report held-out results and disclose the training split and label budget. See [the ground-truth format, provenance, and evaluation protocol](src/resources/BENCHMARK.md).

## Checkpoints

MLLM checkpoints are available at [336labs/VisionEncoder-to-MLLM-ModelZoo](https://huggingface.co/336labs/VisionEncoder-to-MLLM-ModelZoo).

## Citation

```bibtex
@misc{yang2026strong,
  title={A Strong Baseline for Evaluating Vision Encoders in Multimodal Large Language Models},
  author={Yang, Yilin and Tang, Jun-Tao and Wang, Kengyi and Su, Siyuan and Luo, Gaoyong and Chen, Mingda},
  year={2026},
  eprint={2610.05413},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## Acknowledgement

We thank [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), [CLIP](https://github.com/openai/CLIP), and [DINOv2](https://github.com/facebookresearch/dinov2) for their open-source code.
