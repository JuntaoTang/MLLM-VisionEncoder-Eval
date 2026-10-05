# 5-shot BN linear probing

This is the final protocol behind Table 1, not the older selective-BN or
multi-epoch experiments elsewhere in the research repository.

## Fixed protocol

- 70 encoders in `tokenizers.tsv`, including the five discrete tokenizers.
- Frozen encoder; deterministic ImageNet-1K support set with exactly five
  training images per class (5,000 total), seed 0, no augmentation.
- `BatchNorm1d(affine=False) -> Linear(1000)` for every encoder.
- FP32 cached features, global batch size 1,000, one epoch / five updates.
- SGD, momentum 0.9, weight decay 0; 13 independent learning-rate heads and a
  cosine-to-zero schedule.
- Evaluation on 5,000 fixed ImageNet validation images sampled with seed 42.

The exact sampled validation indices are archived at
`../output/validation_indices_n5000_seed42.npy`. Every run also emits a
protocol JSON with seeds, hashes, head definition, sample counts, and learning
rates.

## Data and external models

Expected ImageNet layout:

```text
mini/data/imagenet1k/
  train/
  val/
  extra/
```

Set `IMAGENET_ROOT` and `IMAGENET_EXTRA_ROOT` to use another location. The
loader requires the standard DINOv2 ImageNet metadata in `extra/`.

The minimal CLIP and DINOv2 runtime sources are vendored under `vendor/`.
Other encoder implementations and all weights are deliberately omitted. By
default, `probe_core.py` expects official source checkouts under
`../TokBench/tokenzier_vae_scripts/image_scripts/` and weights under
`../TokBench/tokenizer_modelzoo/`. Its command-line options can point to other
locations. Run `bash ../TokBench/setup_sources.sh` to clone the six required
upstream repositories at the exact commits in `../TokBench/source_versions.tsv`.
`feature_extractors.py` is the single registry showing the exact
model constructor, checkpoint, preprocessing, and feature readout for all 70
encoders.

## Run

```bash
python -m pip install -r requirements.txt

# One model: manifest rank, result id, or internal model id are accepted.
IMAGENET_ROOT=/data/imagenet1k bash run_model.sh toklip_s_256

# All 70; comma-separated GPU ids run independent workers.
IMAGENET_ROOT=/data/imagenet1k PROBE_GPUS=0,1,2,3 bash run_all.sh

# A selected subset is also accepted.
bash run_all.sh toklip_s_256 toklip_l_384 unitok_attn vilau_256 uniar_bsq

python summarize_results.py
```

`PROBE_OUTPUT_ROOT`, `PROBE_CACHE_ROOT`, `PROBE_NUM_WORKERS`, and
`PROBE_PYTHON` are optional overrides. Interrupted feature extraction resumes
from partial memmaps. Fresh results go to `../output/linear_probing_runs/`.

To audit the archived paper results without ImageNet or checkpoints:

```bash
python summarize_results.py --results-root ../output/linear_probing_raw
```
