# CKA-X

**CKA-X** predicts how well a vision encoder will perform inside a multimodal
LLM, **without training that LLM**.  Each candidate encoder is described by two
coordinate-free signals — how its feature geometry relates to other encoders,
and how compatible it is with the language model's text space — and a
gradient-boosted regressor maps them to the downstream benchmark score.

Scoring a new encoder therefore needs only its frozen features and the scores
of a handful of already-evaluated encoders; no MLLM is trained for it.

This folder holds the method, the frozen inputs for the reported numbers, and
the upstream stages that rebuild those inputs from raw features.  It is
Linux-only and self-contained.  Nothing in it fixes how many encoders or images
are used: both are inputs (§8.2), and any pool of encoders with collected scores
works.

---

## 1. Method in one paragraph

Raw encoder features are mutually incomparable (dimensions 384 → 4608,
unrelated coordinate systems), so CKA-X replaces them with two
**coordinate-free** signals and maps them to the MLLM score with a
gradient-boosted tree ensemble:

| signal | symbol | what it is | dimension |
|---|---|---|---|
| **C** | $\mathbf{k}_m = (\mathrm{CKA}(m,m'))_{m'\in\mathcal{R}}$ | linear centered-kernel-alignment similarity to the **labeled reference pool** $\mathcal{R}$ | $\lvert\mathcal{R}\rvert$ |
| **X** | $x_m = [R^2_m,\ \mathrm{CKA}(Z_m,Z^t)]$ | cross-modal compatibility with the frozen target language model | 2 |

descriptor $d_m = [\mathbf{k}_m, x_m]$; learner = depth-2 trees, $T=50$,
$\eta=0.05$, subsample $0.8$, hyperparameters fixed across folds. Scoring a
held-out encoder needs only its frozen features plus the reference pool's known
scores — no MLLM is ever trained for it.

---

## 2. Method → code → artifact map

| step | implemented by | input | reference output |
|---|---|---|---|
| **problem setup** — a pool of candidate encoders, label $y_m$ = unweighted mean over 11 benchmarks | `config.BENCHMARKS`, `scripts/ckax_core.py`, `scripts/ckax_folds.py`; the pool is derived | the labels | — |
| **pairwise CKA (C)** — $G_m = H Z_m Z_m^\top H$, $\mathrm{CKA}(m,m')$ | `scripts/compute_cka_kernel.py` (`--full`, every calibration image) | not included — build it (§5.1) | — |
| **cross-modal alignment (X)** — ridge $R^2$ (half split, seed 42, LOO-$\alpha$) + $\mathrm{CKA}(Z_m,Z^t)$ | `scripts/compute_crossmodal_stats.py` → `scripts/make_cm_final.py` | `artifacts/crossmodal_stats_final_qwen25_full.csv` | — |
| **gradient-boosted prediction** + learner comparison | `scripts/run_ckax.py` (`--preset main`, `--preset learners`) | C (§5.1) + X | `results/reference/fullinfo_gbdt.json`, `fullinfo_krr.json`; `results/derived/learner_comparison.json` |
| **labeling-budget study** — accuracy vs. number of labeled encoders | `scripts/run_budget_sweep.py` + `scripts/plot_budget_sweep.py` | C (§5.1) + X | `results/reference/budget_sweep.{json,csv,md}` (the figure itself is generated, §5.2) |
| upstream chain (rebuild the inputs) | `scripts/prepare_images.py`, `extract_features.py`, `tokenizer_loaders.py`, `extract_discrete_features.py`, `extract_qwen_text_features.py`, `extract_text_features.py` | tokenizer weights, LLM weights, LMUData, OCR-VQA | `images_diverse/`, `features_diverse/`, `sample_data/text_features/` |

**X** and the reference numbers are included; **C** (the pairwise CKA kernel)
is not — §5.1 gives the recipe that builds it, and the reference numbers then
show whether the build is correct.

Every script documents the equations, the fixed hyperparameters and the
protocol it implements in its module docstring.

---

## 3. Layout

```
CKA-X/
├── README.md                      this file
├── requirements.txt
├── .gitignore                     keeps the CKA kernel and rebuild products out
├── run.sh                         the Linux entry point
├── config.py                      paths (env-overridable) + the 11 benchmarks + the label loader
├── artifacts/
│   └── crossmodal_stats_final_qwen25_full.csv   X — cm_cka / cm_r2 per encoder
│                                  (the CKA kernel C is not included; §5.1 builds it)
├── ground_truth/
│   └── ground_truth.example.json  the label format (§7)
├── scripts/
│   ├── ckax_common.py             paths, data-dir lookup, backbone names, spearman
│   ├── ckax_core.py               descriptor loaders (C, X), kernel-ridge predictor, metrics
│   ├── ckax_folds.py              the shared k-fold split + ridge / kNN / GBDT / MLP fold fits
│   ├── verify_ckax.py             self-check: recompute and diff against results/reference/
│   ├── run_ckax.py          ★ method driver: main / combos / xvariants / learners / krr
│   ├── run_budget_sweep.py ★ labeling-budget study (k' = 8…64, 100 splits, seed 0)
│   ├── plot_budget_sweep.py   ★ budget-study figure and tables
│   ├── compute_cka_kernel.py       C from visual features
│   ├── compute_crossmodal_stats.py X from visual + text features
│   ├── make_cm_final.py            raw cross-modal columns → the descriptor's two scalars
│   ├── prepare_images.py           build the calibration image set
│   ├── extract_features.py, tokenizer_loaders.py                visual features, continuous encoders
│   ├── extract_discrete_features.py                             visual features, discrete encoders
│   └── extract_text_features.py, extract_qwen_text_features.py  text features (frozen LLM / CLIP)
├── results/
│   ├── reference/                 reference outputs (tracked; the diff targets)
│   │   ├── budget_sweep.{json,csv,md}      the labeling-budget study
│   │   └── fullinfo_gbdt.json, fullinfo_krr.json    full-information 5-fold
│   └── derived/                   what a run computes (tracked)
│       └── learner_comparison.json      the learners × backbones table
└── logs/                          per-stage logs (created by run.sh)
```

Three paths appear only after a run: `results/rebuild/` (the regenerated
descriptors, §5.3), `figures/` (`bash run.sh figure`) and `logs/`.  Of these,
`results/rebuild/` and `logs/` are git-ignored.

---

## 4. Quick start (Linux)

```bash
# 1. environment
conda create -n ckax python=3.10 -y && conda activate ckax
pip install -r requirements.txt
#    GPU box: pip install torch==2.6.0 torchvision \
#       --index-url https://download.pytorch.org/whl/cu124

# 2. supply the labels (§7)
export CKA_X_GT=/path/to/ground_truth/ground_truth.json

# 3. check the layout
bash run.sh preflight

# 4. build the CKA kernel C (§5.1; it is not included in the package)
export FEAT_DIR=/path/to/features_diverse
bash run.sh cka               # -> results/rebuild/cka_diverse.pt

# 5. reproduce the reference numbers (~2 min, CPU is fine)
bash run.sh pipeline          # main -> learners -> sweep -> figure -> verify
#    (`bash run.sh figure` alone needs no kernel: it plots the archived sweep)

# 6. rebuild the descriptor X (and C) from feature matrices, then evaluate (GPU)
bash run.sh pipeline-descriptor

# 7. rebuild everything, from the calibration images upward (GPU, hours)
bash run.sh pipeline-all
```

`bash run.sh help` lists every stage. Extra arguments after a command are
appended to the underlying command, e.g. `bash run.sh sweep --n_splits 20`.

Every stage writes to `results/derived/` (or `results/rebuild/` for descriptor
rebuilds), never over the frozen `artifacts/` or the `results/reference/` diff
targets — and both descriptor pipelines end by verifying the rebuilt descriptor
against that reference.

---

## 5. Reproducing step by step

### 5.1 Step 1 — build the CKA kernel (C)

**C is not included in the package.**  It is built from the feature matrices
with the command below; the reference numbers (§6) then show whether the build
is correct.

What `compute_cka_kernel.py` expects:

* a feature directory with **one sub-directory per encoder**, each holding
  * `visual_features.pt` — an `(n, d)` float tensor, one row per
    calibration image, rows **L2-normalised** and **row-aligned across
    encoders** (the alignment is recorded in `image_paths.txt`),
  * `image_paths.txt` — the image order that produced those rows;
* `--full`, i.e. every calibration image (no subsampling), and `--device cuda`.

```bash
export FEAT_DIR=/path/to/features_diverse      # the pool's encoders
bash run.sh cka                                # = compute_cka_kernel.py --full
```

The command writes **`results/rebuild/cka_diverse.pt`**, a `.pt` with

```python
{'cka': {'toks':   [encoder names],         # row/column order of the kernel
         'kernel': (m, m) float tensor,     # linear CKA in [0, 1], unit diagonal
         'n_cka':  <images used>},          # calibration images
 'meta': {'feature_dir': ..., 'seed': 0, 'n_cka': <images used>,
          'n_images': <images available>}}
```

The file additionally records `cka.cka_to_ref` and `cka.reference`, plus a
top-level `tok_names`; the evaluation reads only `cka.toks` and `cka.kernel`.

Notes that matter for matching the reference numbers:

* the matrix is **symmetric with a unit diagonal**, and CKA is computed as
  $\lVert X_c^\top Y_c\rVert_F^2 / (\lVert X_c^\top X_c\rVert_F \, \lVert Y_c^\top Y_c\rVert_F)$,
  the standard centered-Gram definition of linear CKA; that identity avoids
  materialising an $n\times n$ Gram matrix (peak RAM ≈ 12 GB);
* the row order is whatever `list_feature_tokenizers()` returns (sorted names),
  which is why it is stored in the file — but **the evaluation keys everything by
  encoder name**, so a different order is fine as long as the names are right;
* $X_c$ uses the centred features; the centering is what makes the kernel
  architecture-agnostic.  Do not skip it.

Where to put it: `results/rebuild/` is what `run.sh` looks for (an explicit
`--cka_pt` / `$CKA` wins, and a copy under `artifacts/` is used if present).
A `.npz` mirror next to the kernel (`cka_diverse.npz`, carrying the same `toks`
and `kernel` arrays) is accepted when `torch.load` is unavailable; `.gitignore`
keeps `*.npz` out of the repository.
Then check the build against the reference numbers:

```bash
bash run.sh verify            # -> 254 checks, 0 mismatches (see §6)
```

### 5.2 Step 2 — evaluation and the budget study (needs C, X, labels)

```bash
bash run.sh pipeline            # all of the below, in order
bash run.sh verify              # 254 checks vs results/reference/ -> 0 mismatches
bash run.sh verify --skip-sweep # ~5 s: artifacts + 5-fold only
bash run.sh main                # GBDT, shared 5-fold       -> results/derived/
bash run.sh learners            # learner comparison         -> results/derived/
bash run.sh sweep               # budget-study data          -> results/derived/
bash run.sh figure              # writes figures/budget_sweep.{png,pdf}
```

`verify` reports:

| section | recomputed | diffed against |
|---|---|---|
| **A** | kernel shape / symmetry / unit diagonal / range, token order, pool and label coverage | — (structural) |
| **B** | descriptor $[\mathbf{k}_m,x_m]$, shared 5-fold seed 42, the labeled pool, learners `gbdt` and `krr` | `fullinfo_gbdt.json`, `fullinfo_krr.json` |
| **C** | budget study: `gbdt`, 100 random labeled subsets per $k'$, seed 0, pooled + fold-mean | `budget_sweep.json` |

`bash run.sh verify --out-md results/derived/verify_report.md` writes the full
per-quantity table.

**Budget-study values** (`bash run.sh figure` plots the pooled caliber; the command
also writes `budget_sweep_table.{tex,md}` next to the figure):

| labeled encoders $k'$ | Qwen3-1.7B | Qwen2.5-1.5B | SmolLM2-1.7B |
|---|---|---|---|
| 8 | 0.498 | 0.509 | 0.356 |
| 16 | 0.646 | 0.686 | 0.512 |
| 24 | 0.696 | 0.748 | 0.555 |
| 32 | 0.743 | 0.783 | 0.616 |
| 40 | 0.766 | 0.801 | 0.636 |
| 48 | 0.778 | 0.816 | 0.649 |
| 56 | 0.779 | 0.822 | 0.666 |
| 64 | 0.809 | 0.853 | 0.685 |

(to two decimals: 0.50 / 0.51 / 0.36 rising to 0.81 / 0.85 / 0.69). The curves are
monotone in the pooled caliber, which is the one the caption names.

### 5.3 Step 3 — rebuild X (and C) from feature matrices (GPU)

Same feature directory as §5.1 plus the text-feature file (see §8.3 for what they
cost to rebuild).  `C` is rebuilt by the first command here as well.

```bash
export FEAT_DIR=/path/to/features_diverse
export TXT_PT=/path/to/sample_data/text_features/text_features_qwen25.pt

bash run.sh cka            # -> results/rebuild/cka_diverse.pt
bash run.sh crossmodal     # -> results/rebuild/crossmodal_stats_final_qwen25_full.csv
bash run.sh pipeline-descriptor   # rebuild + evaluate + verify vs the reference
```

`CKA_X_RESULTS_DIR` routes the descriptor writers into `results/rebuild/`, so
the frozen `artifacts/` and the `results/reference/` diff targets stay intact.
`pipeline-descriptor` ends with

```
== verify the REBUILT descriptor against the archive (results/reference/) ==
```

which is the real end-to-end fidelity check: a rebuilt kernel/CSV that still
reproduces every reference number means the CKA/C definitions match, not just
the shipped artifacts.

### 5.4 Step 4 — rebuild the features themselves (GPU, hours)

```bash
export N_IMAGES=<size of your calibration set>
export IMG_DIR=/path/to/images_diverse
export FEAT_DIR=/path/to/features_diverse
export TOKENIZER_WEIGHTS_ROOT=/path/to/tokenizer/continuous
export VTB_CONFIGS_ROOT=/path/to/configs/continuous/vision_encoder
export LMU_DATA=/path/to/LMUData
export LLM_ROOT=/path/to/llm

bash run.sh images              # the calibration set (LMUData + OCR-VQA)
bash run.sh features            # the continuous encoders -> features_diverse/
bash run.sh features-discrete   # the discrete encoders (9 are extractable; the study uses 5)
bash run.sh text                # text_features_qwen25.pt (frozen Qwen2.5-1.5B)
bash run.sh pipeline-all        # descriptor + evaluation + verification
```

### 5.5 Which command produced each reference file

Paths are the in-package forms; the originals were the same commands under the
project's own results directory.

| reference file | command |
|---|---|
| `results/rebuild/cka_diverse.pt` (C, built in §5.1) | `compute_cka_kernel.py --feature_dir features_diverse --full --out results/rebuild/cka_diverse.pt --device cuda` |
| `artifacts/crossmodal_stats_final_qwen25_full.csv` | `compute_crossmodal_stats.py --diverse_dir features_diverse --text_features_pt …/text_features_qwen25.pt --full_cka --out_name crossmodal_stats_qwen25_full --device cuda` then `make_cm_final.py --src … --out_name crossmodal_stats_final_qwen25_full` |
| `results/reference/fullinfo_gbdt.json` | `run_ckax.py --preset main --learner gbdt --cm_csv artifacts/crossmodal_stats_final_qwen25_full.csv --cka_pt results/rebuild/cka_diverse.pt --targets qwen3 qwen25 smollm2 --out results/reference/fullinfo_gbdt.json` |
| `results/reference/fullinfo_krr.json` | the same with `--learner krr` |
| `results/reference/budget_sweep.{json,csv,md}` | `run_budget_sweep.py --out_json … --out_csv … --out_md …` with its protocol defaults (`--k_sweep 8,16,24,32,40,48,56,64 --n_splits 100 --seed 0 --split_mode nested --learner gbdt --self_mode keep`) |

Re-running either full-information command reproduces its reference file
(check **B** of §6).

---

## 6. What the self-check proves

Once C is built (§5.1), `bash run.sh verify` → **254 checks, 0 mismatches,
tolerance 1e-9** (255 when a `cka_tokens.txt` sits next to the kernel, which
adds the row-order check):

| quantity | reference | recomputed |
|---|---|---|
| budget study, pooled Spearman at $k'=8$ | 0.498 / 0.509 / 0.356 | identical |
| budget study, pooled Spearman at $k'=64$ | 0.809 / 0.853 / 0.685 | identical |
| GBDT, full-information 5-fold, pooled ρ (qwen3 / qwen25 / smollm2) | 0.768 / 0.832 / 0.639 | identical |
| kernel ridge, same protocol | 0.717 / 0.813 / 0.737 | identical |

All 24 pooled points, all fold-mean means **and** stds, and both reference runs
reproduce bit-for-bit from the shipped `artifacts/` once C is built (§5.1).

---

## 7. Labels and paths

**Labels.** The label $y_m$ is the MLLM benchmark score of
encoder $E_m$ (mean over the eleven benchmarks, per language backbone). Those
scores are not distributed with this package — only
`ground_truth/ground_truth.example.json`, which documents the format. Supply
them either by writing the file to `ground_truth/ground_truth.json`, or with:

```bash
export CKA_X_GT=/path/to/ground_truth/ground_truth.json
```

Format:

```json
{
  "clip_openai__l14": { "family": "continuous",
                        "scores": { "qwen3": 47.9, "qwen25": 49.8, "smollm2": 46.1 } }
}
```

`qwen3` = Qwen3-1.7B, `qwen25` = Qwen2.5-1.5B, `smollm2` = SmolLM2-1.7B — the
three language backbones of the reference run.  The set an evaluation covers is
read off the label file itself (pass `--targets` to override); scripts intersect
whatever is present, so partial coverage degrades gracefully.

**Paths.** Every path is env-overridable and defaults to a location *inside* the
package, so the only inputs to supply are the labels (§7) and, for the rebuild
stages, the assets of §8.3 — with one exception: the legacy image-directory
probes inside `prepare_images.py` (`~/data/coco/val2017`, `/data/imagenet/val`,
…). Those are off unless you set `LOCAL_IMAGE_FALLBACK`, and `run.sh` never
reaches them. `bash run.sh preflight` resolves and prints the paths it uses.

| variable | used by | default (inside the package) |
|---|---|---|
| `CKA_X_GT` | all stages | `ground_truth/ground_truth.json` |
| `PY`, `DEVICE` | all stages | `python`, `cuda` |
| `FEAT_DIR` | cka / crossmodal / text / features | `./features_diverse` |
| `IMG_DIR` | images / features | `./images_diverse` |
| `SAMPLE_DIR`, `TXT_PT` | text / crossmodal | `./sample_data`, `…/text_features/text_features_qwen25.pt` |
| `N_IMAGES`, `OCR_RATIO` | images / features | size of your calibration set — **required**, no default; `0.3` |
| `TOKENIZER_WEIGHTS_ROOT` | features | `./tokenizer/continuous` |
| `VTB_CONFIGS_ROOT` | features | `./configs/continuous/vision_encoder` |
| `VTB_ROOT` | features / features-discrete | `./UniTok` |
| `DISC_CKPT`, `VQGAN_PATH` | features-discrete | `./tokenizer/discrete` |
| `CLIP_DIR` | `extract_text_features.py` (CLIP branch; not invoked by `run.sh`) | `./tokenizer/continuous/clip-vit-large-patch14` |
| `LLM_ROOT` | text | `./llm` (re-roots all 8 encoder dirs) |
| `LMU_DATA`, `OCR_VQA_CACHE` | images | `./LMUData`, `./ocr-vqa` |
| `CKA_X_RESULTS_DIR` | descriptor writers | `results/reference`; rebuild stages set `results/rebuild` |

Two module-level directories in `scripts/ckax_common.py` drive the defaults:
`RESULTS_DIR` (`results/reference/`, read as inputs) and `DERIVED_DIR`
(`results/derived/`, written by every run). `FEAT_DIR` / `IMG_DIR` /
`SAMPLE_DIR` are resolved inside the package only — put the data there, or set
the variable to an absolute path elsewhere.

---

## 8. Data assets

### 8.1 What ships

| asset | size | role |
|---|---|---|
| `artifacts/crossmodal_stats_final_qwen25_full.csv` | 6 KB | **X** — `cm_cka`, `cm_r2` per encoder (+ LMU/OCR subsets) |
| `results/reference/*` | 29 KB | reference outputs used as diff targets |
| `results/derived/learner_comparison.json` | 9 KB | learner comparison (every learner × every backbone) |
| `scripts/*.py` | 232 KB | the evaluation code + the upstream rebuild stages |

**C** (the CKA kernel, §5.1) and the label table (§7) are not included.

### 8.2 How the candidate pool is defined

The package carries no pool list; the pool is derived at run time:

```
pool = encoders present in the descriptors  ∩  encoders that have a label for
       every language backbone being evaluated
```

The descriptor table may cover more encoders than the labels do, so the pool is
simply whatever survives that intersection — a study that labels a different set
of encoders gets a different pool, and nothing in the code fixes its size.  Pass
`--pool_file <list>` to any evaluation script to pin one explicitly.

The frozen numbers in `results/reference/` were produced on the study's own pool
and calibration set, and the self-check diffs a recomputation against them, so it
looks that pool up under the internal tag recorded in those files.  A different
pool or image set yields different — equally valid — numbers; what the archive
verifies is that the C and X *definitions* that produced it are the ones in this
code.

### 8.3 Assets you supply, and how to rebuild them

| asset | size | how to get it | needed for |
|---|---|---|---|
| **C** — the pairwise CKA kernel | 29 KB | `bash run.sh cka` (§5.1) | every evaluation stage |
| the labels (one score per encoder and language backbone) | ~12 KB | via `CKA_X_GT` (§7) | every stage |
| `features_diverse/` (one row per calibration image, per encoder) | ~6 GB for the single-layer matrices the pipeline reads; the ~59 GB server tree also carries the optional multi-layer `visual_features_layers.pt` | `bash run.sh features` (+ `features-discrete`; add `--multi_layer` for the layer stack) | §5.3 |
| the calibration image set (LMUData + OCR-VQA) | ~2 GB | `bash run.sh images` | §5.4 |
| `text_features_qwen25.pt` (1536-d) | ~6 KB per image | `bash run.sh text` | §5.3 |
| OCR-VQA question/answer text map | ~1.8 MB (7,158 entries) | your own copy; point the text stage at it | `text` |
| discrete-tokenizer wrappers (`$VTB_ROOT/src/discrete/model/tokenizers/*/wrapper.py`) | — | part of the upstream UniTok tree; the continuous encoders work without it | `features-discrete` |
| Qwen2.5-1.5B (and the other text encoders) | ~3 GB each | your own download; `LLM_ROOT` points at it | §5.3 |
| tokenizer weights | up to ~500 GB | your own download; `TOKENIZER_WEIGHTS_ROOT` points at them | §5.4 |

---

## 9. Environment and pitfalls

Verified stack: **Python 3.10, torch 2.6.0+cu124 (CUDA 12.4), 1× A100 80 GB**.
`compute_cka_kernel.py --full` streams pair-by-pair and peaks near 12 GB.

Four things to watch for in a fresh environment:

1. **timm × the UniTok fork** — extracting `unitok_attn` dies with
   `TypeError: GeGluMlp.__init__() got an unexpected keyword argument 'norm_layer'`.
   The code patches `GeGluMlp.__init__` at import time
   (`extract_discrete_features.py::_patch_unitok_geglumlp`); do **not** edit the
   third-party fork, the patch must be applied again after every update.
2. **`torch.load` needs `weights_only=False`** (torch ≥ 2.6 changed the
   default); every `.pt` in this pipeline stores numpy objects.
3. **`open_clip.create_model()` does not accept `image_size`** in the versions
   used here; resolution changes go through `tokenizer_loaders.py`.
4. **the OCR-VQA arrow cache may be unavailable**; text extraction then reads
   `ocr_text_map.json` instead (see §8.3).

---

## 10. Which text encoder feeds X

X is computed with **one frozen Qwen2.5-1.5B** text encoder (1536-d) for every
backbone, so the budget-study curves differ only in the labels; the same encoder
produced the reference numbers of §6.  The CLIP text features of
`extract_text_features.py` are the alternative input (`TXT_PT`).

---

## 11. Files

The package is self-contained: every import is either the standard library, a
third-party package from `requirements.txt`, or a module inside `scripts/`, and
no path points outside the folder unless you set an env var (§7) — the sole
exception is the opt-in image-directory probe in `prepare_images.py`
(`LOCAL_IMAGE_FALLBACK`), which `run.sh` never reaches.

| file | role |
|---|---|
| `config.py` | the eleven benchmark names and the label loader (`CKA_X_GT`) |
| `scripts/ckax_common.py` | where things are read/written (`RESULTS_DIR`, `DERIVED_DIR`), data-dir lookup, feature-dir listing, the backbone names of the label file, `spearman` |
| `scripts/ckax_core.py` | descriptor loading (X, C), the kernel-ridge predictor, the three metrics, the alpha/kernel-weight LOO helpers |
| `scripts/ckax_folds.py` | the shared k-fold split and the ridge / kNN / GBDT / MLP fold fits used by the learner comparison |
| `scripts/run_ckax.py` | the method driver (`--preset main`, `--preset learners`) |
| `scripts/run_budget_sweep.py`, `plot_budget_sweep.py` | budget-study data and figure |
| `scripts/compute_cka_kernel.py`, `compute_crossmodal_stats.py`, `make_cm_final.py` | rebuild C and X from feature matrices |
| `scripts/prepare_images.py`, `extract_features.py`, `tokenizer_loaders.py`, `extract_discrete_features.py`, `extract_text_features.py`, `extract_qwen_text_features.py` | rebuild the feature matrices and the text features |
| `scripts/verify_ckax.py` | the self-check |


---

## 12. Troubleshooting

| symptom | cause / fix |
|---|---|
| `[ERROR] no CKA kernel at …` | the kernel is not included: `bash run.sh cka` (§5.1), or pass `--cka_pt`. |
| `[ERROR] ground truth is empty` | no `ground_truth/ground_truth.json` and no `CKA_X_GT`. |
| `[ERROR] N_IMAGES is not set` | the `images` / `features` stages rebuild the calibration set, so its size is an input with no default: `export N_IMAGES=<number of images>` (§5.4). |
| `ModuleNotFoundError: sklearn` / `scipy` / `matplotlib` | `pip install -r requirements.txt`. |
| `ModuleNotFoundError: open_clip` / `datasets` | only the upstream stages need them. |
| `ModuleNotFoundError: src.discrete…` | the discrete wrappers are external (§8.3); the continuous encoders rebuild without them. |
| check **B** mismatches while **A** passes | the labels differ from those used for the reference files. |
| check **C** mismatches | first check the kernel: a different `torch`/numpy build shifts the CKA values, not the learner (GBDT tie-breaking is `sklearn`'s, with `random_state=0`). Re-run at the default `--n_splits 100` — a smaller `--n_splits` cannot reproduce a 100-split reference. |
| GPU OOM during `run.sh cka` | `--full` peaks near 12 GB; drop `--full` for the subsampled kernel (20 % of the images by default, `--n_cka` to change it — numbers will then differ from the reference). |
| the figure is not byte-identical to another rendering of the same numbers | PDFs embed timestamps and object IDs; compare the plotted values, not the file hash. |
| a stage logged as FAILED | `run.sh` never aborts; read `logs/<stage>.log` and the `FAILED stages:` line in the summary. |
