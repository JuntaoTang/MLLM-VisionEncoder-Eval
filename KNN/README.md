## Evaluation protocols

All KNN variants use exact inner-product search on L2-normalized features,
`k=20`, temperature-weighted voting (`temperature=0.07`) and seed 42.

| Track | Official split | Database shots/class | Query/class |
|---|---|---:|---:|
| `knn_5shot` | ImageNet-1K validation | 5 | 5 |
| `different_shot` | ImageNet-1K training | 5, 10, 20, 45, 95, 195 | 5 |

The shot sets are nested: the 5-shot set is a prefix of the 10-shot set, and
so on. Query samples are fixed across every shot. Feature-extraction FLOPs are
computed for the database samples used by that shot plus the common 5,000
queries; exact-search FLOPs are `2 * N_query * N_database * feature_dim`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the CUDA environment used to generate the checked-in result table:

```bash
conda env create -f environment.yml
conda activate zero-shot-knn
```

Some encoders require their upstream repository or a model-specific environment.
The corresponding file in `knn_5shot/models/` or `different_shot/models/`
documents that dependency beside the model loading code. Keep third-party
repositories outside this repository or under the ignored `third_party/`
directory.

## Build fixed protocols

Five-shot validation protocol:

```bash
python protocols/build_protocol.py \
  --imagenet-root /path/to/imagenet/val \
  --shots 5 \
  --output protocols/imagenet_val_5shot_5query_seed42.json
```

Multi-shot training protocol:

```bash
python protocols/build_protocol.py \
  --imagenet-root /path/to/imagenet/train \
  --shots 5 10 20 45 95 195 \
  --output protocols/imagenet_train_195shot_5query_seed42.json
```

The protocol records a SHA-256 digest of ImageFolder-relative paths. This can
be used to detect a different dataset layout without publishing the licensed
ImageNet file list.

The two fixed, path-sanitized protocol JSON files used by the experiments are
also checked into `protocols/`, so a reproducer need not resample the split.

## Multi-shot evaluation from one feature extraction

Export one `N x D` feature matrix and one `N` label array in the same ordering
as `torchvision.datasets.ImageFolder`. Then run:

```bash
python different_shot/evaluate_features.py \
  --features features/MODEL.npy \
  --labels features/labels.npy \
  --protocol protocols/imagenet_train_195shot_5query_seed42.json \
  --model MODEL \
  --flops-per-image FLOPS \
  --output results/MODEL
```

The same feature matrix is reused for all six shots. The command writes JSON
and CSV and prints the paper-table format:

```text
| Model | Shot | Train | Top1 | TFLOPs |
```

`different_shot/run_preexported_features.py` is retained as a convenience for
the original exported-feature manifest. It is **not a standalone reproducer**:
it assumes 200,000 pre-exported feature rows, a manifest and hard-coded FLOPs.
For independent reproduction, generate the protocol and features first and use
`different_shot/evaluate_features.py`.

## Model-specific extraction code

The `models/` directories contain the original family-specific model loading,
preprocessing and feature-extraction implementations after removing private
machine paths. Set dataset, checkpoint, cache and output constants to paths in
your environment before running a model file. These files cover DINO, DINOv2,
DINOv3, EUPE, I-JEPA, MAE, MetaCLIP, MetaCLIP worldwide, PE-Core, PixIO, RAE,
SigLIP, TokLIP, UniAR-BSQ, UniTok, VILA-U and WebSSL-DINO.

Do not commit downloaded checkpoints, ImageNet, feature arrays or result logs.

The aggregate 70-model table is provided under `results/knn_multishot/` in
CSV, JSON and Markdown formats. It contains no feature vectors or private paths.

## Pre-exported feature artifact

The approximately 60 GB feature matrices are deliberately not stored in Git.
contains the manifest, per-model metadata, labels and source
indices required to interpret the external artifact.

The official model sources are
[`facebookresearch/metaclip`](https://github.com/facebookresearch/metaclip) and
[`google-research/big_vision`](https://github.com/google-research/big_vision).

## Privacy and double-blind review

No author names, email addresses, SSH keys, tokens, institution names, private
hostnames or original `/home/...` paths should be committed. A private GitHub
repository under a personal account is suitable for private development, but
its owner identity is not anonymous to collaborators. Follow the venue's
double-blind code-release policy before sharing access with reviewers.
