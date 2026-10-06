# MLLM-VisionEncoder-Eval

Evaluate vision encoders for a target LLM with RAVEL, baseline metrics, and downstream MLLM training and evaluation.

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

| Dataset | Use | Download Source |
| --- | --- | --- |
| LLaVA-LCS-558K | RAVEL, MLLM pretraining, AC Policy | [Hugging Face](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main) |
| LLaVA-1.5 mix665K | MLLM finetuning | [Hugging Face](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main) |
| ImageNet-1K | kNN, linear probing, zero-shot | [ImageNet](https://image-net.org/download.php) |
| COCO / Karpathy | Alignment probing, caption evaluation | [COCO](https://cocodataset.org/#download), [Karpathy](https://cs.stanford.edu/people/karpathy/deepimagesent/) |
| CC3M | Alignment probing | [Official repository](https://github.com/google-research-datasets/conceptual-captions) |
| SPair-71k | AC Policy | [Project page](https://cvlab.postech.ac.kr/research/SPair-71k/) |
| TokBench | Reconstruction evaluation | [Hugging Face](https://huggingface.co/datasets/Junfeng5/TokBench/tree/main) |

MLLM training paths follow [configs/mllm/data.yaml](configs/mllm/data.yaml). Prepare the 11 evaluation splits listed in [configs/mllm/eval_datasets.yaml](configs/mllm/eval_datasets.yaml) as local LMUData TSVs and images using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit); [VQAv2](https://visualqa.org/download.html) and Karpathy annotations supply the VQA and caption splits.

**Feature caches.** Run each pretrained vision encoder over the dataset once and save its features for reuse. RAVEL also needs caption features from the target LLM, in the same sample order. Set cache paths in `configs/local.yaml`; `ravel_paper` uses prepared visual caches and automatically extracts text features. Extraction settings use the supplied defaults.

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

## Your Own Metric

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

## Checkpoints

Download trained MLLMs from [336labs/VisionEncoder-to-MLLM-ModelZoo](https://huggingface.co/336labs/VisionEncoder-to-MLLM-ModelZoo) and set `paths.trained` for evaluation.

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

We thank [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), [CLIP](https://github.com/openai/CLIP), and [DINOv2](https://github.com/facebookresearch/dinov2).
