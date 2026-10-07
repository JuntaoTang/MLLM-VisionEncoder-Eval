# MLLM-VisionEncoder-Eval

## Introduction

**Given a set of vision encoders and a target LLM, how to select the encoder best suited to that LLM for MLLM training?**

Training and evaluating an MLLM for every candidate in order to choose the best one is expensive.

MLLM-VisionEncoder-Eval provides a unified framework for this selection problem, bringing together encoder evaluation metrics, baseline comparisons, and downstream MLLM training and evaluation.

## Our Evaluation Method: RAVEL

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

RAVEL predicts downstream MLLM performance by comparing nearest-neighbor structures of paired image and text features, without downstream labels or MLLM training.

<p align="center">
  <strong>70 vision encoders &middot; 3 language models &middot; 210 MLLMs &middot; 9 evaluation metrics</strong>
</p>

## Setup

Python 3.10+. From the repository root:

```bash
bash scripts/setup.sh
.venv/bin/python -m pip install -e '.[paper]'
bash scripts/reproduce.sh --init-local --data-root /path/to/data
```

Set dataset, weight, feature, and worker Python paths in `configs/local.yaml` ([template](configs/local.example.yaml)). Worker environments use [scripts/setup_worker.sh](scripts/setup_worker.sh).

## Data

[336labs/VisionEncoder-Eval-ReproData](https://huggingface.co/datasets/336labs/VisionEncoder-Eval-ReproData) provides datasets for downstream MLLM evaluation, and reproducing all vision encoder evaluation methods (we released precomputed image features from all vision encoders). Download the required data and set the paths in `configs/local.yaml` ([template](configs/local.example.yaml)).

[Ground-truth scores](src/resources/ground_truth.json) for all **210 trained MLLMs** across 11 downstream tasks are included in this repository.

## Checkpoints

Our trained MLLM checkpoints are available at [336labs/VisionEncoder-to-MLLM-ModelZoo](https://huggingface.co/336labs/VisionEncoder-to-MLLM-ModelZoo). 

To run downstream MLLM evaluation, download the desired checkpoints and set `paths.trained` in `configs/local.yaml` to your checkpoint root. 

## Running

```bash
bash scripts/reproduce.sh --list
bash scripts/reproduce.sh ravel          # CLIP-L/14 + Qwen2.5
bash scripts/reproduce.sh ravel_paper    # 70 encoders x 3 LLMs
bash scripts/reproduce.sh knn
bash scripts/reproduce.sh mllm_train
bash scripts/reproduce.sh mllm_eval
```

Add `--check` to inspect required inputs. Results are saved in `runs/`.

## Try Your Own Metric

### Option 1: Submit a Prediction CSV

Fill the template's `score` column for each encoder/LLM pair, then compare with the bundled 210 MLLM scores:

```bash
.venv/bin/python -m pip install -e '.[benchmark]'
.venv/bin/vision-encoder-eval benchmark template --output predictions.csv
.venv/bin/vision-encoder-eval benchmark evaluate \
  --predictions predictions.csv --metric my_metric --output runs/my_metric
```

### Option 2: Connect a Python Scoring Function

Install `.[benchmark]` as above, then create `my_metric.py` with a function `predict(*, encoder_id, llm)` returning one finite numeric score. Wrap your own implementation, replacing `your_method` below:

```python
# my_metric.py
from your_method import compute_score


def predict(*, encoder_id, llm):
    return float(compute_score(encoder_id=encoder_id, llm=llm))
```

Run from the directory containing `my_metric.py` (adjust the virtual environment path if needed):

```bash
PYTHONPATH=. .venv/bin/vision-encoder-eval benchmark run \
  --metric my_metric:predict --output runs/my_metric
```

The runner calls your function for all 210 encoder/LLM pairs, saves `predictions.csv`, and writes `correlations.json` and `correlations.csv`. Your function handles model loading, feature extraction or cache loading, and scoring; it receives only the two IDs. No registry changes are required. See [the paired-feature Python example](src/examples/custom_metric.py) for a complete adapter using RSA.

Both options report Spearman and Pearson correlations per LLM. Add `--direction lower` for distance/error metrics, or `--llm qwen25` to evaluate only that LLM's 70 encoders (also use it when generating a CSV template). See [the evaluation protocol](src/resources/BENCHMARK.md).

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

We thank the authors and contributors of [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for their MLLM training and evaluation infrastructure, and [CLIP](https://github.com/openai/CLIP) and [DINOv2](https://github.com/facebookresearch/dinov2) for their vision encoder implementations.
