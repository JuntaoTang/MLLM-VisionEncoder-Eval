# MLLM-VisionEncoder-Eval

## Paper

**A Strong Baseline for Evaluating Vision Encoders in Multimodal Large Language Models**

Yilin Yang<sup>*</sup>, Jun-Tao Tang<sup>*</sup>, Kengyi Wang, Siyuan Su, Gaoyong Luo, Mingda Chen<sup>&dagger;</sup>

<sup>*</sup> Equal contribution. <sup>&dagger;</sup> Corresponding author.

[Paper](https://arxiv.org/pdf/2610.05413) | [Checkpoints](https://huggingface.co/336labs/VisionEncoder-to-MLLM-ModelZoo)

## Introduction

Given a set of vision encoders and a target LLM, we aim to select the encoder best suited to that LLM for MLLM training. Training and evaluating an MLLM for every candidate is expensive.

RAVEL (Retrieval-based Assessment of Vision Encoders for Language Models) ranks candidate encoders for the target LLM through cross-modal nearest-neighbor retrieval, without MLLM training. This repository includes RAVEL, evaluation baselines, and MLLM training and evaluation for 70 vision encoders and 3 language models.

## Requirements

**Environment.** Python 3.10+. From the repository root:

```bash
bash scripts/setup.sh
.venv/bin/python -m pip install -e '.[paper]'
bash scripts/reproduce.sh --init-local --data-root /path/to/vision_encoder_eval_data
```

**Data and weights.** Prepare aligned visual/text feature caches for RAVEL. MLLM training uses LLaVA-LCS-558K and LLaVA-665K; evaluation uses 11 benchmarks. Set dataset, weight, feature, and worker Python paths in `configs/local.yaml` ([template](configs/local.example.yaml)). See the [data inventory](docs/DATASET_AUDIT.md) and [feature guide](docs/FEATURE_PANEL_REPRODUCTION.md) for details.

## Running

Run RAVEL with prepared caches (defaults to CLIP-L/14 + Qwen2.5):

```bash
bash scripts/reproduce.sh ravel --check
bash scripts/reproduce.sh ravel
```

Run the paper's 70-encoder x 3-LLM RAVEL evaluation, or other experiments:

```bash
bash scripts/reproduce.sh ravel_paper
bash scripts/reproduce.sh knn
bash scripts/reproduce.sh mllm_train
bash scripts/reproduce.sh mllm_eval
bash scripts/reproduce.sh --list
```

Paper runs require verified final-layer visual patches. For existing caches with unverified layer provenance, add `--use-existing-visual-cache`; the output records this limitation. Add `--check` to inspect required paths and dependencies. Results are saved in `runs/`.

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
