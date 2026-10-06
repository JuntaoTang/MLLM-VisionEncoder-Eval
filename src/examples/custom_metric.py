"""Example plugin: RSA on paired features with verified sample identities.

Set VEE_METRIC_FEATURES to a directory containing visual/<encoder_id>.npy and
text/<llm>.npy. Each array needs a .json feature manifest beside it.
Replace the RSA call with your own metric to use the same prediction interface.
"""
import os
from pathlib import Path

import numpy as np

from vision_encoder_eval.data.features import validate_paired_rows
from vision_encoder_eval.methods.rsa import RSA


def predict(*, encoder_id, llm):
    root = os.environ.get("VEE_METRIC_FEATURES")
    if not root:
        raise ValueError("set VEE_METRIC_FEATURES to the paired-feature directory")
    visual_path = Path(root) / "visual" / f"{encoder_id}.npy"
    text_path = Path(root) / "text" / f"{llm}.npy"
    visual = np.load(visual_path, allow_pickle=False)
    text = np.load(text_path, allow_pickle=False)
    validate_paired_rows(
        {"visual_manifest": str(visual_path.with_suffix(".json")),
         "text_manifest": str(text_path.with_suffix(".json"))},
        visual_path, text_path, visual, text, {},
    )
    return RSA().evaluate(visual, text, {})["final_score"]
