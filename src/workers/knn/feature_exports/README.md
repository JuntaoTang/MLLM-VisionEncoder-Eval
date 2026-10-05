# Exported feature metadata

This directory includes the small files required to identify and reproduce the
original 200,000-image feature export:

- `metadata/`: per-model JSON metadata (70 models)
- `labels.npy`: ImageNet-1K labels in export row order
- `source_indices.npy`: row-to-ImageFolder index mapping
- `manifest.tsv`: compact table of exported models
- `export_manifest.json`: full machine-readable manifest

The `features/` directory is intentionally not committed. It contains about
60 GB of generated `.npy` matrices, including individual files larger than
2 GB. Set `FEATURE_EXPORT_ROOT` to a directory containing the complete layout:

```text
FEATURE_EXPORT_ROOT/
├── features/
├── metadata/
├── labels.npy
├── source_indices.npy
├── manifest.tsv
└── export_manifest.json
```

Example:

```bash
export FEATURE_EXPORT_ROOT=/path/to/vae_linear_probing_200shot_features_fp32
python different_shot/run_preexported_features.py
```

The matrices are derived artifacts, not model source code. They should be
placed in an external dataset/artifact store with checksums and downloaded on
demand rather than committed to ordinary Git history.

