# MLLM-VisionEncoder-Eval

## Introduction

Given a set of vision encoders and a target LLM, we aim to select the encoder best suited to that LLM for MLLM training. Training and evaluating an MLLM for every candidate is expensive.

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

<p align="center">
  <strong>RAVEL</strong><br>
  Retrieval-based Assessment of Vision Encoders for Language Models
</p>

<div align="center">
  <a href="https://arxiv.org/abs/2610.05413"><img src="https://img.shields.io/badge/arXiv-2610.05413-B31B1B?style=flat-square" alt="Paper on arXiv"></a>
  <a href="https://huggingface.co/336labs/VisionEncoder-to-MLLM-ModelZoo"><img src="https://img.shields.io/badge/Hugging%20Face-Checkpoints-FFD21E?style=flat-square" alt="Model checkpoints on Hugging Face"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/BibTeX-Citation-3776AB?style=flat-square" alt="BibTeX citation"></a>
</div>

<p align="center">
  <strong>A training-free metric for selecting vision encoders for a target language model.</strong>
</p>

**RAVEL** ranks vision encoders by comparing nearest-neighbor structures in visual and textual representation spaces. It combines **stabilized PCA whitening** with **fine-grained patch-level similarity scoring** to predict downstream MLLM performance.

- **Training-free assessment.** Evaluate encoder compatibility with a target LLM using paired image-text representations, without downstream labels or MLLM training.
- **Stabilized representations.** PCA whitening reduces geometric bias while limiting the amplification of low-variance directions.
- **Fine-grained visual similarity.** Patch-level scoring preserves information that global pooling can discard.

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

**Data and weights.** Prepare aligned visual/text feature caches for RAVEL. MLLM training uses LLaVA-LCS-558K and LLaVA-665K; evaluation uses 11 benchmarks. Set dataset, weight, feature, and worker Python paths in `configs/local.yaml` ([template](configs/local.example.yaml)).

## Running

List available experiments, run a baseline, or train and evaluate an MLLM:

```bash
bash scripts/reproduce.sh --list
bash scripts/reproduce.sh knn
bash scripts/reproduce.sh mllm_train
bash scripts/reproduce.sh mllm_eval
```

Run RAVEL with prepared caches (defaults to CLIP-L/14 + Qwen2.5), or evaluate all 70 encoders with all 3 LLMs:

```bash
bash scripts/reproduce.sh ravel --check
bash scripts/reproduce.sh ravel
bash scripts/reproduce.sh ravel_paper
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
