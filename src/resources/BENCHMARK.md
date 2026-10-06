# Paper ground truth

[`ground_truth.json`](ground_truth.json) is the authoritative, versioned label
table for **70 vision encoders x 3 language models = 210 trained MLLMs** in
[A Strong Baseline for Evaluating Vision Encoders in Multimodal Large Language Models](https://arxiv.org/abs/2610.05413).
It is included in source distributions and wheels; evaluation requires no
private data paths or trained model downloads.

## Format and units

- `schema_version`: 1.
- `encoders`: keyed by canonical `encoder_id`; includes display name and family.
- `llms`: `qwen3` (Qwen3-1.7B), `qwen25` (Qwen2.5-1.5B-Instruct), and
  `smollm2` (SmolLM2-1.7B-Instruct).
- `encoders[encoder_id].llms[llm].scores`: scores for all 11 downstream tasks.
- `encoders[encoder_id].llms[llm].average`: the original two-decimal average
  used for paper correlations. Preserve it exactly rather than recomputing it
  from the rounded task scores.
- Scores use percentage units; captioning CIDEr is multiplied by 100.
- `order` sorts encoders by mean downstream performance. Prediction templates
  use the canonical encoder registry order. Evaluation joins by IDs, never by
  row position, `rank`, display name, or this sorted order.

The 11 tasks are MMMU, MMBench, VQAv2, ScienceQA, ChartQA, DocVQA, TextVQA,
POPE, GQA, MSCOCO captioning, and Flickr30K captioning. Their exact split IDs
are recorded under `benchmarks`.

## Provenance and maintenance

The labels were selected from the archived downstream result table, whose
SHA-256 is recorded in `provenance.source_sha256`. Run-specific bookkeeping
was removed; all original paper labels were preserved.

The released JSON file has SHA-256:

```text
05a5a1de7a3b2687b2ca49f188230f5d0f95c0829ed19771068d2765d8c0729c
```

Validate the complete encoder/LLM/task coverage and label consistency with:

```bash
vision-encoder-eval data ground-truth
```

For a correction, retain the original result table and its provenance, import
through `vision-encoder-eval data ground-truth --import-from <archived-table.json>
--output <candidate.json>`, and review the label changes before replacing the
released resource. Update this checksum and the release-integrity test together
with an explanation of the correction. Never fill unavailable labels with zero.

CSV exports are generated from this JSON using `benchmark export-ground-truth`;
they are not a second independently maintained label table. Correlation reports
include the SHA-256 of the label file and the prediction CSV.

## Evaluation protocol

The primary comparison computes Spearman and Pearson separately for each LLM
over all 70 encoders, using its original 11-task `average`. Per-task correlations
are also reported. Scores with ties use SciPy's Spearman implementation. The
three LLMs are not pooled, because their downstream performance scales differ.

The evaluator requires every encoder for each selected LLM. Missing, duplicate,
unknown, nonfinite, and constant predictions fail evaluation. `--direction lower`
negates scores before correlation; raw predictions remain in their original
units. A constant per-task ground truth produces null correlations and an
explicit reason in JSON.

For training-free comparisons, compute predictions without reading downstream
labels. Methods fitted to labels should use held-out evaluation and disclose
their training split and label budget; correlations on their training labels
are not comparable to the paper's training-free results.
