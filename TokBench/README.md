# TokBench discrete-tokenizer subset

This directory retains only the image reconstruction and TokBench evaluation
needed for the five discrete configurations reported in Table 6.

| Vision encoder | Reconstruction path | TokBench |
| --- | --- | --- |
| TokLIP-S | shared TokLIP VQ tokenizer | evaluated at 256 |
| TokLIP-L | shared TokLIP VQ tokenizer | evaluated at 256 |
| UniTok | UniTok attention tokenizer | evaluated at 256 |
| VILA-U | VILA-U 7B tokenizer | evaluated at 256 |
| UniAR-BSQ | no released decoder | not applicable |

TokLIP-S and TokLIP-L intentionally produce the same reconstruction and score:
the two vision-encoder sizes share one discrete VQ tokenizer.

## Setup

```bash
python -m pip install -r requirements.txt
bash setup_sources.sh
bash download_data.sh
```

The benchmark downloads to `tokbench_data/` by default. Set `DATA_ROOT` to
place it elsewhere. The local, patched docTR evaluator is included because it
is part of the metric implementation.

Model repositories and checkpoints are intentionally excluded. Put the
official TokLIP, UniTok, and VILA-U source checkouts under
`tokenzier_vae_scripts/image_scripts/`, as referenced by the small `*_rec.py`
adapters, and their weights under `tokenizer_modelzoo/`. `setup_sources.sh`
does the source part at the exact commits recorded in `source_versions.tsv`;
it also fetches the three source trees needed only by the full 70-encoder
linear-probing panel. The adapter CLIs also
accept explicit source/checkpoint paths; run them with `--help` for details.

## Reconstruct and evaluate

```bash
# Everything that has a released reconstruction decoder.
bash run_all.sh

# Or one stage/model at a time.
bash run_reconstruction.sh toklip_s
bash run_reconstruction.sh toklip_l
bash run_reconstruction.sh unitok
bash run_reconstruction.sh vilau_256
bash run_eval_all.sh
```

Useful overrides are `DATA_ROOT`, `RECON_ROOT`, `MODEL_ZOO`, `OUT_DIR`, and
`SUMMARY_DIR`. Reconstruction is fixed to the native 256 setting used in the
paper.

The final summary gives equal weight to three difficulty buckets, matching the
paper:

- text area ratios `[0.02,0.03)`, `[0.03,0.04)`, `[0.04,1)` for T-ACC/T-NED;
- face area ratios `[0.1,0.2)`, `[0.2,0.3)`, `[0.3,1)` for F-Sim.

`summarize_paper_results.py` writes the final table and the bucket-level audit
CSV files to `../output/`.
