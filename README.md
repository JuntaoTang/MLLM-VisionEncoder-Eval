# VisualTokenizerBench / RAVEL

Evaluate visual encoders with RAVEL and baseline methods using cached features.

## Run RAVEL

Python 3.10+ is required. From the repository root:

```bash
bash scripts/setup.sh
bash scripts/reproduce.sh --init-local --data-root /path/to/vision_encoder_eval_data
.venv/bin/python -m pip install -e '.[paper]'
bash scripts/reproduce.sh ravel_paper --check
bash scripts/reproduce.sh ravel_paper
```

Table 1 uses 70 encoders x 3 LLMs, 1,000 aligned LCS pairs, final-layer visual
patches without special tokens, penultimate text states averaged over valid
tokens, k=100 and full-rank PCA whitening with epsilon=1e-4. The runner extracts
text from the original frozen backbones and writes `scores.csv` and
`correlations.json` under `runs/ravel_paper/`.

Visual audits must verify `feature_layer: final` (or `-1`) and the ordered sample
manifest. Check the extraction source before adding this field; legacy audits
without layer provenance must be verified or the features re-extracted.
To attempt reproduction with existing patches whose layer is unverified, use
`ravel_paper --use-existing-visual-cache`; reports retain that limitation.

On this server, the three LLMs and aligned caches are already prepared.
Run the complete 210-pair evaluation with the existing visual features:

```bash
HF_HUB_OFFLINE=1 bash scripts/reproduce.sh ravel_paper --use-existing-visual-cache
```

The data root is `/cache/vision_encoder_eval_data`; `configs/local.yaml` selects
an audited copy that restores one corrupt UniAR scalar from its official encoder.
The original cache is preserved. Initialization is needed only once. CUDA is
used when available; add `--device cpu` for CPU execution.

With verified final-layer visual and penultimate mean-pooled text caches:

```bash
bash scripts/reproduce.sh ravel --encoder dino_vits16 --text-encoder qwen3
bash scripts/reproduce.sh ravel --encoder all --text-encoder all
```

The full panel contains 70 visual encoders x 3 text encoders and can take time.
The launcher verifies cache audits and sample order; it never falls back to
last-token text. The cached `ravel` command defaults to CLIP-L/14 + Qwen2.5.

For your own features, supply patch `[N,T,D]` and text `[N,D]` arrays,
plus a JSON list of unique sample IDs in their shared row order:

```bash
bash scripts/reproduce.sh ravel --patches /path/patches.npy \
  --text /path/text.npy --sample-ids /path/sample_ids.json
```

Results and logs are under `runs/`; suites include CSV/JSON reports.
Repeated unchanged runs reuse successful results. Local paths are configured
in `configs/local.yaml` and are not committed.

The paper's 210 downstream labels (70 encoders x 3 LLMs, 11 benchmarks)
are bundled in `src/resources/ground_truth.json`; validate with
`.venv/bin/vision-encoder-eval data ground-truth`.

## Other Experiments

```bash
bash scripts/reproduce.sh --list
bash scripts/reproduce.sh knn                 # 70 cached models, six shot counts
bash scripts/reproduce.sh linear_probe --check
bash scripts/reproduce.sh mllm_train
bash scripts/reproduce.sh mllm_eval
```

Add `--check` to inspect required inputs/dependencies, or `--dry-run` to
inspect the plan. Other baselines and MLLM pipelines require their datasets,
checkpoints and worker environments; configure them in `configs/local.yaml`.
RAVEL cached-score evaluation does not retrain MLLMs or reproduce every paper
experiment. Source modules live directly in `src/`; recipes are in `configs/`.
