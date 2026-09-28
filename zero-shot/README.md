## Zero-shot evaluation

### Official model sources

| Model family | Official code, documentation and checkpoints |
|---|---|
| SigLIP 2 | [`google-research/big_vision` — SigLIP 2 README](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md) |
| MetaCLIP 1 (MC1) | [`facebookresearch/metaclip`](https://github.com/facebookresearch/metaclip) |
| MetaCLIP 2 (MC2) | [`facebookresearch/metaclip`](https://github.com/facebookresearch/metaclip) |

These links point to the upstream projects used to identify model names,
preprocessing settings, tokenizers and released checkpoints. This repository
contains evaluation code only and does not redistribute model weights.

### `clip_benchmark`

`zero_shot/run_clip_benchmark.sh` clones the upstream
[`LAION-AI/CLIP_benchmark`](https://github.com/LAION-AI/CLIP_benchmark) and runs
ImageNet-1K zero-shot classification for explicitly listed MetaCLIP/SigLIP
model-pretrained pairs. Before producing final numbers, set
`CLIP_BENCHMARK_REF` to the exact reviewed upstream commit rather than a moving
branch:

```bash
export IMAGENET_ROOT=/path/to/imagenet/val
export CLIP_BENCHMARK_REF=<commit-sha>
bash zero_shot/run_clip_benchmark.sh
```

`zero_shot/models.txt` contains the tested MetaCLIP and SigLIP
`model,pretrained` pairs. Set `MODEL_FILE` to a smaller compatible list to run
only a subset.

### Native MetaCLIP 2 batch evaluator

`zero_shot/zero_shot_metaclip2_imagenet.py` is the corresponding native
MetaCLIP 2 evaluation path. By default it evaluates the 12 worldwide
checkpoints other than the previously evaluated B/16 224px baseline. Pass
`--include-baseline` to evaluate all 13 checkpoints shown by `--list-models`.

```bash
python zero_shot/zero_shot_metaclip2_imagenet.py \
  --root /path/to/metaclip2/checkpoints \
  --val /path/to/imagenet/val \
  --xlm-tokenizer facebook/metaclip-2-worldwide-b16 \
  --mt5-spm google/mt5-base
```

The script builds the ImageNet classifier from the 80 OpenAI prompt templates,
uses exact checkpoint filenames rather than a filesystem glob, writes JSON and
CSV after every completed model, and contains no machine-specific paths.
